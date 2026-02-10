import os
import sys
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    try:
        model_report = {}
        for model_name, model in models.items():
            #Perform Grid Search CV to find the best hyperparameters for the current model
            gs = GridSearchCV(estimator=model, param_grid=params[model_name], cv=5, n_jobs=-1, scoring='r2')
            gs.fit(X_train, y_train)
            best_model = gs.best_estimator_
            y_test_pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_test_pred)
            mse = mean_squared_error(y_test, y_test_pred)
            mae = mean_absolute_error(y_test, y_test_pred)
            model_report[model_name] = (r2_square, mse, mae, best_model)
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)