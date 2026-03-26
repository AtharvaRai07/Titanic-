import os
import sys
import pandas as pd
import sqlalchemy
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
from config.database_config import DB_CONFIG
from config.paths_config import *

class DataIngestion:
    def __init__(self, db_params, ouput_dir):
        self.db_params = db_params
        self.output_dir = ouput_dir

        os.makedirs(self.output_dir, exist_ok=True)


    def connect_to_db(self):
        try:
            engine = sqlalchemy.create_engine(
                f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}"
                f"@{self.db_params['host']}:{self.db_params['port']}/{self.db_params['db_name']}"
            )

            logging.info("Successfully connected to the database.")
            return engine
        except Exception as e:
            logging.error("Error connecting to the database.")
            raise CustomException(e, sys)


    def extract_data(self):
        try:
            engine = self.connect_to_db()
            query = "SELECT * FROM public.titanic"
            df = pd.read_sql(query, engine)

            logging.info("Data extraction successful.")
            return df
        except Exception as e:
            logging.error("Error extracting data from the database.")
            raise CustomException(e, sys)

    def save_data(self, df):
        try:
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            logging.info("Data split into training and testing sets successfully.")

            train_df.to_csv(TRAIN_PATH, index=False)
            test_df.to_csv(TEST_PATH, index=False)

            logging.info(f"Data saved successfully to {TRAIN_PATH} and {TEST_PATH}.")
        except Exception as e:
            logging.error("Error saving data to CSV files.")
            raise CustomException(e, sys)

    def run(self):
        try:
            logging.info("Starting data ingestion process.")
            df = self.extract_data()
            self.save_data(df)
            logging.info("Data ingestion process completed successfully.")

        except Exception as e:
            logging.error("Data ingestion process failed.")
            raise CustomException(e, sys)

if __name__ == "__main__":

    data_ingestion = DataIngestion(db_params=DB_CONFIG, ouput_dir=RAW_DIR)
    data_ingestion.run()
