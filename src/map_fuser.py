import cv2
import numpy as np


class MapFuser:
    """MaskFuser is a class for fusing multiple probability maps into a single fused mask.

    It supports adding probability maps, calculating the conditional probability fusion,
    and obtaining the fused mask.
    """

    def __init__(self):
        self.maps = []

    def add_prob_map(
        self,
        map_path: str,
    ):
        # Read the mask using OpenCV
        prob_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if prob_map is None:
            raise ValueError(f'Failed to read the mask: {map_path}')

        # Ensure the input mask has the same shape as existing masks
        if self.maps:
            assert (
                prob_map.shape == self.maps[0].shape
            ), 'Input mask must have the same shape as existing masks'

        # Check the scale type of the mask
        if np.max(prob_map) > 1.0:
            prob_map = self.convert_to_map(prob_map)

        self.maps.append(prob_map)

    @staticmethod
    def convert_to_map(
        image: np.ndarray,
    ):
        return image / 255.0

    @staticmethod
    def convert_to_mask(
        image: np.ndarray,
    ):
        return (image * 255.0).astype(np.uint8)

    def conditional_probability_fusion(self):
        # Ensure at least one mask has been added
        assert len(self.maps) > 0, 'No maps have been added'

        # Calculate the conditional probability fusion
        fused_map = np.zeros_like(self.maps[0])

        for y in range(fused_map.shape[0]):
            for x in range(fused_map.shape[1]):
                # Calculate the conditional probability of foreground
                prob_foreground = np.prod([mask[y, x] for mask in self.maps])

                # Calculate the conditional probability of background
                prob_background = 1 - prob_foreground

                # Normalize the probabilities
                total_prob = prob_foreground + prob_background
                if total_prob > 0:
                    prob_foreground /= total_prob
                    prob_background /= total_prob

                # Assign the fused probability to the foreground class
                fused_map[y, x] = prob_foreground

        return fused_map


if __name__ == '__main__':
    # Create an instance of MaskFuser
    fuser = MapFuser()

    # Add prob_map paths
    map1_path = 'data/interim_lungs/MAnet/map/10013643_58785837.png'
    map2_path = 'data/interim_lungs/DeepLabV3/map/10013643_58785837.png'
    map3_path = 'data/interim_lungs/Unet++/map/10013643_58785837.png'
    fuser.add_prob_map(map1_path)
    fuser.add_prob_map(map2_path)
    fuser.add_prob_map(map3_path)

    # Call the conditional_probability_fusion method
    fused_mask = fuser.conditional_probability_fusion()
    cv2.imwrite('fused_mask.png', fused_mask)
    print('text')
