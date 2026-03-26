import sys
from src.data_ingestion import DataIngestion
from src.feature_store import RedisFeatureStore
from src.data_processing import DataProcessor
from src.model_training import ModelTraining

from config.database_config import *
from config.paths_config import *

from src.logger import logging
from src.exception import CustomException

class TrainingPipeline:
    def run(self):
        try:
            logging.info("Starting training pipeline...")

            # Step 1: Data Ingestion
            logging.info("Ingesting data...")

            data_ingestion = DataIngestion(db_params=DB_CONFIG, output_dir=RAW_DIR)
            data_ingestion.run()

            # Step 2: Data Processing
            logging.info("Processing data...")

            data_processor = DataProcessor(train_data_path=TRAIN_PATH, test_data_path=TEST_PATH, feature_store=RedisFeatureStore())
            data_processor.run()

            # Step 3: Model Training
            logging.info("Training model...")

            model_trainer = ModelTraining(feature_store=RedisFeatureStore(), model_save_path=MODEL_DIR)
            model_trainer.run()
        except Exception as e:
            logging.error(f"Error occurred while running training pipeline: {str(e)}")
            raise CustomException(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run()
