import os
import sys
from dataclasses import dataclass

# from catboost import CatBoostRegressor  # Commented out - not compatible with Python 3.14
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        try:
            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(),
                "Lasso Regression": Lasso(max_iter=5000),
                "Decision Tree": DecisionTreeRegressor(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor()
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False)  # Commented out - not compatible with Python 3.14
            }

            params={
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },

                "Gradient Boosting": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                },

                "AdaBoost": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.01, 0.1]
                },

                "Linear Regression": {
                    "fit_intercept": [True, False],
                    "n_jobs": [-1]
                },

                "Ridge Regression": {
                    'alpha': [0.1, 1.0, 10.0]
                },

                "Lasso Regression": {
                    'alpha': [0.1, 1.0, 10.0]
                },

                "Decision Tree": {
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },

                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7],
                    'weights': ['uniform', 'distance']
                },

                "XGBRegressor": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1],
                    'max_depth': [3, 5]
                }
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                                 models=models, params=params)  

            # Extract R2 scores from the report (first element of each tuple)
            r2_scores = {name: metrics[0] for name, metrics in model_report.items()}
            
            best_model_score = max(r2_scores.values())
            best_model_name = [name for name, score in r2_scores.items() if score == best_model_score][0]
            # Get the FITTED model (4th element, index 3) from the report
            best_model = model_report[best_model_name][3]

            logging.info("Best model found: %s with R2 score: %f", best_model_name, best_model_score)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            y_pred = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_pred)
            return r2_square
        except Exception as e:
            logging.error("Error occurred during model training: %s", str(e))
            raise CustomException(e, sys)



