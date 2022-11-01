import os
import logging
import argparse
from pathlib import Path
import pandas as pd
from PIL import Image
import supervisely_lib as sly

from tools.utils import convert_base64_to_image

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
    filename="logs/{:s}.log".format(Path(__file__).stem),
    filemode="w",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def check_dataset_dirs(
    dataset_dir: str, dataset_ann_dir: str, dataset_img_dir: str
) -> None:
    """

    Args:
        dataset_dir:
        dataset_ann_dir:
        dataset_img_dir:

    Returns:

    """
    logger.info("Checking dataset paths")

    if not os.path.isdir(dataset_dir):
        raise OSError(f'"{dataset_dir}" does not exist!')

    if not os.path.isdir(dataset_ann_dir):
        raise OSError(f'"{dataset_ann_dir}" does not exist!')

    if not os.path.isdir(dataset_img_dir):
        raise OSError(f'"{dataset_img_dir}" does not exist!')


def crop_images(dataset_img_dir: str, save_dir: str) -> None:
    """

    Args:
        dataset_img_dir:
        save_dir:

    Returns:

    """
    logger.info("Cropping images")

    save_img_dir = os.path.join(save_dir, "img")
    os.makedirs(save_img_dir, exist_ok=True)

    img_list = os.listdir(dataset_img_dir)

    top = 0
    left = 0
    for img_name in img_list:
        img = Image.open(os.path.join(dataset_img_dir, img_name))
        subject_id, study_id, left_width, right_width, ext = img_name.replace(
            ".", "_"
        ).split("_")
        right = int(left_width)
        bottom = img.height

        img = img.crop((left, top, right, bottom))

        img.save(os.path.join(save_img_dir, f"{subject_id}_{study_id}.{ext}"))


def get_edema_name(ann: dict) -> str:
    """

    Args:
        ann:

    Returns:

    """
    if ann["tags"]:
        check_labeler_login(ann["tags"][0]["labelerLogin"])
        edema_name = ann["tags"][0]["value"]
    else:
        logger.warning("There is no tags!")
        edema_name = ""
    return edema_name


def check_labeler_login(labeler_login: str) -> None:
    """

    Args:
        labeler_login:

    Returns:

    """
    if labeler_login in ("ViacheslavDanilov", "mak_en", "irina.ryndova"):
        logger.error(f"Wrong labeler login: {labeler_login}")


def get_object_rp(obj: dict) -> str:
    """

    Args:
        obj:

    Returns:

    """
    if obj["tags"]:
        rp = obj["tags"][0]["value"]
    else:
        logger.warning("There is info about RP!")
        rp = ""
    return rp


def get_object_sizes(obj: dict) -> dict:
    """

    Args:
        obj:

    Returns:

    """
    if obj["geometryType"] == "bitmap":
        bitmap = convert_base64_to_image(obj["bitmap"]["data"])
        x1, y1 = obj["bitmap"]["origin"][0], obj["bitmap"]["origin"][1]
        x2 = x1 + bitmap.shape[0]
        y2 = y1 + bitmap.shape[1]
    else:
        xs = [x[0] for x in obj["points"]["exterior"]]
        ys = [x[1] for x in obj["points"]["exterior"]]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def get_box_sizes(x1: int, y1: int, x2: int, y2: int) -> dict:
    """

    Args:
        x1:
        y1:
        x2:
        y2:

    Returns:

    """
    box_width, box_height = x2 - x1, y2 - y1
    xc, yc = box_width / 2, box_height / 2

    return {"xc": xc, "yc": yc, "box width": box_width, "box height": box_height}


def prepare_metadata_annotations(dataset_ann_dir: str, save_dir: str) -> None:
    """

    Args:
        dataset_ann_dir:
        save_dir:

    Returns:

    """
    logger.info("Preparing metadata and annotations")

    save_ann_dir = os.path.join(save_dir, "ann")
    os.makedirs(save_ann_dir, exist_ok=True)

    edema_id = {
        "": 0,
        "Vascular congestion": 1,
        "Interstitial edema": 2,
        "Alveolar edema": 3,
    }

    figure_id = {
        "Cephalization": 0,
        "Artery": 1,
        "Heart": 2,
        "Kerley": 3,
        "Bronchus": 4,
        "Effusion": 5,
        "Bat": 6,
        "Infiltrate": 7,
    }

    metadata = pd.DataFrame(
        columns=[
            "image path",
            "subject id",
            "study id",
            "image width",
            "image height",
            "figure",
            "x1",
            "y1",
            "x2",
            "y2",
            "xc",
            "yc",
            "box width",
            "box height",
            "rp",
            "class",
        ]
    )

    ann_list = os.listdir(dataset_ann_dir)
    for ann_name in ann_list:
        logger.info(f"Processing annotation {ann_name}")
        annotation = pd.DataFrame(
            columns=["edema id", "figure id", "x1", "y1", "x2", "y2"]
        )

        ann = sly.io.json.load_json_file(os.path.join(dataset_ann_dir, ann_name))

        edema_name = get_edema_name(ann)

        subject_id, study_id, width, right_width, ext, _ = ann_name.replace(
            ".", "_"
        ).split("_")
        height = ann["size"]["height"]
        cropped_img_path = os.path.join(
            save_dir, "img", f"{subject_id}_{study_id}.{ext}"
        )

        if len(ann["objects"]) == 0:
            logger.warning(f"There is no objects!")
            continue

        for obj in ann["objects"]:
            logger.info(f"Processing object {obj}")
            check_labeler_login(obj["labelerLogin"])

            rp = get_object_rp(obj)
            xy = get_object_sizes(obj)
            box = get_box_sizes(*xy.values())
            figure_name = obj["classTitle"]

            annotation_info = {
                "edema id": edema_id[edema_name],
                "figure id": figure_id[figure_name],
            }
            annotation_info.update(xy)
            annotation = annotation.append(annotation_info, ignore_index=True)

            image_info = {
                "image path": cropped_img_path,
                "subject id": subject_id,
                "study id": study_id,
                "image width": width,
                "image height": height,
                "figure": figure_name,
                "rp": rp,
                "class": edema_name,
            }
            image_info.update(xy)
            image_info.update(box)
            metadata = metadata.append(image_info, ignore_index=True)

        new_annotation_name = f"{subject_id}_{study_id}.csv"
        logging.info(f"Saving annotation {new_annotation_name}")
        annotation.to_csv(
            os.path.join(save_ann_dir, new_annotation_name),
            header=False,
            index=False,
            sep=" ",
        )

    logging.info("Saving metadata.csv")
    metadata.to_csv(os.path.join(save_dir, f"metadata.csv"))


def main(
    dataset_dir: str,
    save_dir: str,
) -> None:
    """

    Args:
        dataset_dir:
        save_dir:

    Returns:

    """
    dataset_ann_dir = os.path.join(dataset_dir, "ann")
    dataset_img_dir = os.path.join(dataset_dir, "img")

    check_dataset_dirs(dataset_dir, dataset_ann_dir, dataset_img_dir)

    crop_images(dataset_img_dir, save_dir)

    prepare_metadata_annotations(dataset_ann_dir, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Supervisely dataset")
    parser.add_argument(
        "--dataset_dir", default="dataset/MIMIC-CXR-Edema-SLY/DS1", type=str
    )
    parser.add_argument(
        "--save_dir", default="dataset/MIMIC-CXR-Edema-Convert", type=str
    )
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        save_dir=args.save_dir,
    )
