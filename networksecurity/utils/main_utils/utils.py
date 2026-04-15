import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os, sys
import numpy as np
# import dill
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    """

    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:

    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Write YAML file
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise NetworkSecurityException(e, sys)    

def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Save numpy array data to a file

    Args:
        file_path (str): Location where file will be saved
        array (np.ndarray): Numpy array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise NetworkSecurityException(e, sys) 
    
def save_object(file_path: str, obj: object):
    """
    Save any Python object as a pickle file

    Args:
        file_path (str): Path where object will be saved
        obj (object): Any Python object (model, scaler, etc.)
    """
    try:
        logging.info("Entered the save_object method in Mainutils class")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method in Mainutils class")

    except Exception as e:
        raise Exception(f"Error saving object: {e}")    


def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")

        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e, sys)

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load numpy array data from file
    Args:
        file_path (str): Location of file to load
    Returns:
        np.ndarray: Loaded numpy array
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")

        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)

    except Exception as e:
        raise NetworkSecurityException(e, sys)    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        for model_name, model in models.items():

            # Correct param access
            param_grid = params.get(model_name, {})

            # GridSearch only if params exist
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3)
                gs.fit(X_train, y_train)
                model.set_params(**gs.best_params_)

            # Train
            model.fit(X_train, y_train)

            # Predict
            y_test_pred = model.predict(X_test)

            # Score (classification)
            test_score = accuracy_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)