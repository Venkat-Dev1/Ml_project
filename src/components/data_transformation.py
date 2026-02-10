import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''



        try:
            numerical_features = ['math_score', 'reading_score', 'writing_score','Total_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                                    'lunch', 'test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical and categorical pipelines created successfully")
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_features),
                    ('cat_pipeline', cat_pipeline, categorical_features)
                ]
            )
            logging.info("Preprocessor created successfully")
            return preprocessor 
        except Exception as e:
            logging.error("Error occurred while creating data transformer object: %s", str(e))
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Initiating data transformation")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data successfully")

            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()
            target_column_name = 'Average_score'
            numerical_columns = ['math_score', 'reading_score', 'writing_score',
                                 'Total_score']
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessor object on training and testing datasets")
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Data transformation completed successfully")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (train_arr, 
                    test_arr, 
                    self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            logging.error("Error occurred during data transformation: %s", str(e))
            raise CustomException(e, sys)