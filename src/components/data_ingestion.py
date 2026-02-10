import os
import sys
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Get project root directory dynamically
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            data_file_path = os.path.join(project_root, 'notebook', 'data', 'student_with_avg_score.csv')
            df = pd.read_csv(data_file_path)
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False,header=True)
            logging.info("Saved the raw data to path: %s", self.ingestion_config.raw_data_path)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False,header=True)
            logging.info("Split the data into train and test sets and saved to paths: %s and %s",
                         self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

            return (self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            logging.error("Error occurred during data ingestion: %s", str(e))
            raise CustomException(e, sys)

if __name__ == "__main__":    
    obj = DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer=ModelTrainer()
    r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr, data_transformation.data_transformation_config.preprocessor_obj_file_path)
    logging.info("R2 score of the best model: %f", r2_score)