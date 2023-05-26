import logging
import os
from pathlib import Path
from typing import List

import cv2
import hydra
import numpy as np
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data.utils import get_file_list
from src.data.utils_sly import FEATURE_MAP, FEATURE_TYPE, get_box_sizes
from src.models.map_fuser import MapFuser
from src.models.mask_processor import MaskProcessor

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def compute_lungs_info(
    mask: np.ndarray,
) -> dict:
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    lung_coords = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        x1, y1, x2, y2 = x, y, x + width, y + height
        lung_coords.append([x1, y1, x2, y2])

    x1_values, y1_values, x2_values, y2_values = zip(*lung_coords)
    x1, y1 = min(x1_values), min(y1_values)
    x2, y2 = max(x2_values), max(y2_values)

    feature_name = 'Lungs'
    lungs_info = {
        'Feature ID': FEATURE_MAP[feature_name],
        'Feature': feature_name,
        'Reference type': FEATURE_TYPE[feature_name],
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2,
    }

    lungs_info.update(get_box_sizes(x1=x1, y1=y1, x2=x2, y2=y2))

    return lungs_info


def process_prob_maps(
    img_paths: List[str],
    save_dir: str,
) -> dict:
    # Fuse segmentation probability maps
    fuser = MapFuser()
    for img_path in img_paths:
        fuser.add_prob_map(img_path)
    fused_map = fuser.conditional_probability_fusion()
    fused_map = (fused_map * 255.0).astype(np.uint8)

    # Process obtained fused map
    processor = MaskProcessor()
    mask_bin = processor.binarize_image(image=fused_map)
    mask_smooth = processor.smooth_mask(mask=mask_bin)
    mask_clean = processor.remove_artifacts(mask=mask_smooth)

    # Save the fused map and its mask
    img_name = Path(img_paths[0]).name
    map_dir = os.path.join(save_dir, 'map')
    mask_dir = os.path.join(save_dir, 'mask')
    os.makedirs(map_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    map_path = os.path.join(map_dir, f'{img_name}')
    mask_path = os.path.join(mask_dir, f'{img_name}')
    cv2.imwrite(map_path, fused_map)
    cv2.imwrite(mask_path, mask_clean)

    # Extract lungs metadata
    lungs_info = compute_lungs_info(mask=mask_clean)

    return lungs_info


def reorder_image_paths(
    input_lists: List[List[str]],
) -> List[List[str]]:
    # Check if all input lists have the same number of elements
    num_elements = len(input_lists[0])
    assert all(
        len(lst) == num_elements for lst in input_lists
    ), 'All input lists must have the same number of elements'

    # Use zip() to iterate over the elements at the same index from the input lists
    output_list = []
    for items in zip(*input_lists):
        # Check if all file paths have the same basename
        basenames = [os.path.basename(path) for path in items]
        assert all(
            basename == basenames[0] for basename in basenames
        ), 'All file paths must have the same basename'
        output_list.append(list(items))

    return output_list


@hydra.main(
    config_path=os.path.join(os.getcwd(), 'configs'),
    config_name='fuse_maps',
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    log.info(f'Config:\n\n{OmegaConf.to_yaml(cfg)}')

    log.info(f'Models used...............: {cfg.model_names}')
    log.info('')

    # Get list of paths to images
    img_path_sets: List[List[str]] = []
    for model_name in cfg.model_names:
        img_dir = os.path.join(cfg.img_dir, f'{model_name}')
        img_paths_ = get_file_list(
            src_dirs=img_dir,
            ext_list=[
                '.png',
                '.jpg',
                '.jpeg',
                '.bmp',
            ],
        )
        img_paths_ = img_paths_[:10]  # TODO: remove after debugging
        # img_path_sets.append(img_paths_)
        img_path_sets.append(img_paths_)
        log.info(f'{model_name}...............: {len(img_paths_)} images')

    # Reorder image paths for multiprocessing
    img_path_sets = reorder_image_paths(img_path_sets)

    # Process segmentation probability maps
    Parallel(n_jobs=-1)(
        delayed(process_prob_maps)(img_path_set, cfg.save_dir)
        for img_path_set in tqdm(img_path_sets, desc='Processing')
    )

    # TODO: remove after debugging
    # Process probability map
    # from src.models.mask_processor import MaskProcessor
    #
    # processor = MaskProcessor()
    # mask_bin = processor.binarize_image(image=fused_map)
    # mask_smooth = processor.smooth_mask(mask=mask_bin)
    # mask_clean = processor.remove_artifacts(mask=mask_smooth)
    # lungs_info = compute_lungs_info(mask=mask_clean)
    # for model_name in cfg.model_names:
    #     log.info(f'Model in use: {model_name}')
    #     print(f'Model in use: {model_name}')
    #     model_dir = os.path.join(cfg.model_dirs, model_name)
    #     map_dir = os.path.join(cfg.save_dir, f'{model_name}', 'map')
    #     mask_dir = os.path.join(cfg.save_dir, f'{model_name}', 'mask')
    #     os.makedirs(map_dir, exist_ok=True)
    #     os.makedirs(mask_dir, exist_ok=True)
    #
    #     metadata = pd.DataFrame(columns=METADATA_COLUMNS)
    #
    #     model = LungSegmenter(
    #         model_dir=model_dir,
    #         device='auto',
    #     )
    #
    #     for img_path in tqdm(img_paths, desc='Lung segmentation', unit='images'):
    #         img_name = Path(img_path).name
    #         img = cv2.imread(img_path)
    #         img_height, img_width = img.shape[:2]
    #         img_ratio = img_height / img_width
    #
    #         # Retrieve and save a probability segmentation map
    #         map_ = model(
    #             img=img,
    #             scale_output=True,
    #         )
    #         map = cv2.resize(map_, (img_width, img_height), interpolation=cv2.INTER_LANCZOS4)
    #         map_path = os.path.join(map_dir, img_name)
    #         cv2.imwrite(map_path, map)
    #
    #         # Retrieve and save a binary segmentation mask
    #         mask_ = model.binarize_map(
    #             map=map,
    #             threshold_method='otsu',
    #         )
    #         mask_smooth = model.smooth_mask(mask=mask_)
    #         mask_lungs = model.remove_artifacts(mask=mask_smooth)
    #         mask_path = os.path.join(mask_dir, img_name)
    #         cv2.imwrite(mask_path, mask_lungs)
    #
    #         # Compute lung coordinates
    #         lungs_info = model.compute_lungs_info(mask=mask_lungs)
    #
    #         # Extract metadata
    #         img_stem = Path(img_path).stem
    #         subject_id, study_id = img_stem.split('_')
    #         obj_info = {
    #             'Image path': img_path,
    #             'Image name': img_name,
    #             'Subject ID': subject_id,
    #             'Study ID': study_id,
    #             'Image width': img_width,
    #             'Image height': img_height,
    #             'Image ratio': img_ratio,
    #             'View': 'Frontal',
    #         }
    #         obj_info.update(lungs_info)
    #         # TODO: add encoded mask to the column "Mask"
    #         metadata = metadata.append(obj_info, ignore_index=True)
    #
    #     # Save metadata
    #     metadata_path = os.path.join(model_dir, 'metadata.xlsx')
    #     log.info(f'Saving metadata to {metadata_path}')
    #     metadata.sort_values(['Image path'], inplace=True)
    #     metadata.reset_index(drop=True, inplace=True)
    #     metadata.index += 1
    #     metadata.to_excel(
    #         metadata_path,
    #         sheet_name='Metadata',
    #         index=True,
    #         index_label='ID',
    #     )
    #
    #     # Empty memory
    #     del model
    #     gc.collect()
    #     torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
