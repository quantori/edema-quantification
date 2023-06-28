import cv2
import numpy as np


class MapFuser:
    """MapFuser is a class for fusing multiple probability maps into a single fused map.

    It supports adding probability maps, calculating the conditional probability fusion,
    and obtaining the fused map.
    """

    def __init__(self):
        self.prob_maps = []

    def add_prob_map(
        self,
        prob_map: np.ndarray,
    ):
        # Ensure the input mask has the same shape as existing masks
        if self.prob_maps:
            assert (
                prob_map.shape == self.prob_maps[0].shape
            ), 'Input mask must have the same shape as existing masks'

        # Check the scale type of the mask
        if np.max(prob_map) > 1.0:
            prob_map = prob_map / 255.0

        self.prob_maps.append(prob_map)

    def conditional_probability_fusion(self, scale_output=True):
        # Ensure at least one map has been added
        assert len(self.prob_maps) > 0, 'No prob_maps have been added'

        if len(self.prob_maps) == 1:
            return self.prob_maps[0]

        # Calculate the conditional probability fusion
        img_height, img_width = self.prob_maps[0].shape[:2]
        prob_product = np.prod(self.prob_maps, axis=0)
        fused_map = np.array(
            [
                [self.calculate_probability(prob_product[y, x]) for x in range(img_width)]
                for y in range(img_height)
            ],
        )

        if scale_output:
            fused_map = (fused_map * 255.0).astype(np.uint8)

        self.prob_maps.clear()

        return fused_map

    @staticmethod
    def calculate_probability(prob_foreground: float) -> float:
        # Calculate the conditional probability of background
        prob_background = 1 - prob_foreground

        # Normalize the probabilities
        total_prob = prob_foreground + prob_background
        if total_prob > 0:
            prob_foreground /= total_prob
            prob_background /= total_prob

        return prob_foreground


if __name__ == '__main__':
    # Create an instance of MaskFuser
    fuser = MapFuser()

    # Add prob_map paths
    map_paths = [
        'data/interim_lungs/DeepLabV3/10013643_58785837.png',
        'data/interim_lungs/FPN/10013643_58785837.png',
        'data/interim_lungs/MAnet/10013643_58785837.png',
    ]
    for map_path in map_paths:
        prob_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        fuser.add_prob_map(prob_map)
    fused_map = fuser.conditional_probability_fusion(scale_output=True)
    print(np.mean(fused_map), np.std(fused_map))

    # Process probability map
    from src.models.mask_processor import MaskProcessor

    processor = MaskProcessor()
    mask_bin = processor.binarize_image(image=fused_map)
    mask_smooth = processor.smooth_mask(mask=mask_bin)
    mask_clean = processor.remove_artifacts(mask=mask_smooth)

    print('Complete')
