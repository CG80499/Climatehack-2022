import numpy as np
import torch
import tensorflow as tf

from climatehack import BaseEvaluator
from model3 import Model


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        self.model = Model()
        self.model.load_state_dict(torch.load('modelV11_weightsMAX8.pth', map_location=torch.device('cpu')))
        self.model_LSTM = tf.keras.models.load_model('ConvLSTM_HV11.hdf5', compile=False)
        #self.model.eval()

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
        with torch.no_grad():
            prediction = (
                self.model.predict(torch.from_numpy(data).view(-1, 12, 128, 128))
                .view(24, 64, 64)
                .detach()
                .numpy()
            )
            
            prediction *= 1023.0
            prediction = prediction.astype(np.float32)
            assert prediction.shape == (24, 64, 64)
        
        # (12, 128, 128, 1)
        features = np.expand_dims(data, axis = -1)
        
        Images = features.reshape(-1, 128, 128, 1)

        # Resizing images to 64x64
        frames = tf.image.central_crop(Images, 0.5)
        frames = tf.reshape(frames, [-1, 12, 64, 64, 1])

        pred = self.model_LSTM.predict(frames[:, -3:])
        pred = pred.reshape(3, 64, 64)
        pred *= 1023
        prediction[:2] = pred[:2]
        return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()