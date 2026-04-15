from networksecurity.constant.training_pipeline import SAVE_MODEL_DIR, MODEL_TRAINER_MODEL_FILE_NAME

import os
import sys

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def predict(self, x):
        try:
            # Step 1: Transform input data
            x_transform = self.preprocessor.transform(x)

            # Step 2: Predict using trained model
            y_hat = self.model.predict(x_transform)

            return y_hat

        except Exception as e:
            raise NetworkSecurityException(e, sys)