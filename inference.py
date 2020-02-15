import json
from models.network import Classifier

with open("idx2classes.json", "r") as f:
    idx2classes = f.read()

idx2classes = json.loads(idx2classes)


class CarClassifier(object):
    """Makes a prediction for a PIL images using your trained model.
    Args:
        model_path: The path to your saved model.
    """

    def __init__(self, model_path="model.pth"):
        self.classifier = Classifier()
        self.classifier.load_checkpoint(model_path)

    def predict(self, image_file):
        """
        Args:
            image_file: path to image file you want to predict.
        Returns:
            string: The predicted class of the car.
        """
       
        return self.classifier.predict(image_file)
