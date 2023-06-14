import cv2
import numpy as np


class MapFuser:
    """MaskFuser is a class for fusing multiple probability maps into a single fused mask.

    It supports adding probability maps, calculating the conditional probability fusion,
    and obtaining the fused mask.
    """

    def __init__(self):
        self.prob_maps = []

    def add_prob_map(
        self,
        map_path: str,
    ):
        # Read the mask using OpenCV
        prob_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if prob_map is None:
            raise ValueError(f'Failed to read the mask: {map_path}')

        # Ensure the input mask has the same shape as existing masks
        if self.prob_maps:
            assert (
                prob_map.shape == self.prob_maps[0].shape
            ), 'Input mask must have the same shape as existing masks'

        # Check the scale type of the mask
        if np.max(prob_map) > 1.0:
            prob_map = prob_map / 255.0

        self.prob_maps.append(prob_map)

    def conditional_probability_fusion(
        self,
        scale_output: bool = True,
    ):
        # Ensure at least one map has been added
        assert len(self.prob_maps) > 0, 'No prob_maps have been added'

        if len(self.prob_maps) == 1:
            return self.prob_maps[0]

        # Calculate the conditional probability fusion
        fused_map = np.zeros_like(self.prob_maps[0])

        for y in range(fused_map.shape[0]):
            for x in range(fused_map.shape[1]):
                # Calculate the conditional probability of foreground
                prob_foreground = np.prod([mask[y, x] for mask in self.prob_maps])

                # Calculate the conditional probability of background
                prob_background = 1 - prob_foreground

                # Normalize the probabilities
                total_prob = prob_foreground + prob_background
                if total_prob > 0:
                    prob_foreground /= total_prob
                    prob_background /= total_prob

                # Assign the fused probability to the foreground class
                fused_map[y, x] = prob_foreground

        if scale_output:
            fused_map = (fused_map * 255.0).astype(np.uint8)

        return fused_map


if __name__ == '__main__':
    # Create an instance of MaskFuser
    fuser = MapFuser()

    # Add prob_map paths
    map1_path = 'data/interim_lungs/MAnet/10013643_58785837.png'
    map2_path = 'data/interim_lungs/DeepLabV3/10013643_58785837.png'
    map3_path = 'data/interim_lungs/Unet++/10013643_58785837.png'
    fuser.add_prob_map(map1_path)
    fuser.add_prob_map(map2_path)
    fuser.add_prob_map(map3_path)
    fused_map = fuser.conditional_probability_fusion()
    fused_map = (fused_map * 255.0).astype(np.uint8)

    # Process probability map
    from src.models.mask_processor import MaskProcessor

    processor = MaskProcessor()
    mask_bin = processor.binarize_image(image=fused_map)
    mask_smooth = processor.smooth_mask(mask=mask_bin)
    mask_clean = processor.remove_artifacts(mask=mask_smooth)
    print('text')
