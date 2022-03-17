import numpy as np
import torch

from climatehack import BaseEvaluator
from optical_flow_model import get_flow_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Running on: {}".format(str(device).upper())) #For Debugging

class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""
        if not torch.cuda.is_available():
            print("Warning: If you are running this on a CPU it could take a very long time.")

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)
        data /= 1023.0
        prediction = get_flow_images(data)
        prediction *= 1023

        prediction = prediction.cpu().detach().numpy()            
            
        prediction = prediction.astype(np.float32)
        
        assert prediction.shape == (24, 64, 64)
    
        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()