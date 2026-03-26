import os
import sys
import pandas as pd
from imblearn.over_sampling import SMOTE
from src.feature_store import RedisFeatureStore
from src.logger import logging
from src.exception import CustomException
from config.paths_config import *

class DataProcessor:
    def __init__(self, train_data_path, test_data_path, feature_store: RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.feature_store = feature_store
        self.train_data = None
        self.test_data = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.X_resampled = None
        self.y_resampled = None

    def load_data(self):
        try:
            self.train_data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)

            logging.info("Data loaded successfully.")

        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise CustomException(e, sys)


    def preprocess_data(self):
        try:

            logging.info("Doing data preprocessing...")

            self.train_data['Age'] = self.train_data['Age'].fillna(self.train_data['Age'].median())
            self.train_data['Embarked'] = self.train_data['Embarked'].fillna(self.train_data['Embarked'].mode()[0])
            self.train_data['Fare'] = self.train_data['Fare'].fillna(self.train_data['Fare'].median())
            self.train_data['Sex'] = self.train_data['Sex'].map({"male": 0, "female": 1})
            self.train_data['Embarked'] = self.train_data['Embarked'].astype('category').cat.codes

            logging.info("Doing feature engineering...")

            self.train_data['FamilySize'] = self.train_data['SibSp'] + self.train_data['Parch'] + 1
            self.train_data['IsAlone'] = (self.train_data['FamilySize'] == 1).astype(int)
            self.train_data['HasCabin'] = self.train_data['Cabin'].apply(lambda x: 0 if pd.isnull(x) else 1)
            self.train_data['Title'] = self.train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False).map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}).fillna(4).astype(int)
            self.train_data['PclassFare'] = self.train_data['Pclass'] * self.train_data['Fare']
            self.train_data['AgeFare'] = self.train_data['Age'] * self.train_data['Fare']

            logging.info("Removing columns that are not needed for modeling...")

            self.train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

            logging.info("Data preprocessing completed successfully.")

        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise CustomException(e, sys)

    def handle_imbalance_data(self):
        try:
            X = self.train_data.drop(['Survived'], axis=1)
            y = self.train_data['Survived']

            smote = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = smote.fit_resample(X, y)

            logging.info("Handling imbalanced data using SMOTE completed successfully.")

        except Exception as e:
            logging.error(f"Error during handling imbalanced data: {e}")
            raise CustomException(e, sys)

    def store_features_in_redis(self):
        try:
            batch_data = {}

            for index, row in self.train_data.iterrows():
                entity_id = row['PassengerId']
                features = {
                    "Age": row['Age'],
                    "Fare": row['Fare'],
                    "Pclass": row['Pclass'],
                    "Sex": row['Sex'],
                    "Parch": row['Parch'],
                    "SibSp": row["SibSp"],
                    "Embarked": row['Embarked'],
                    "FamilySize": row['FamilySize'],
                    "IsAlone": row['IsAlone'],
                    "HasCabin": row['HasCabin'],
                    "Title": row['Title'],
                    "PclassFare": row['PclassFare'],
                    "AgeFare": row['AgeFare'],
                    "Survived": row['Survived']
                }
                batch_data[entity_id] = features

            self.feature_store.store_batch_feature(batch_data)

            logging.info("Storing features in Redis completed successfully.")

        except Exception as e:
            logging.error(f"Error during storing features in Redis: {e}")
            raise CustomException(e, sys)

    def retrieve_features_redis_store(self, entity_id):
        try:
            features = self.feature_store.retrieve_feature(entity_id)
            return features
        except Exception as e:
            logging.error(f"Error during retrieving features from Redis: {e}")
            raise CustomException(e, sys)

    def run(self):
        try:
            logging.info("Starting data processing pipeline...")

            self.load_data()
            self.preprocess_data()
            self.handle_imbalance_data()
            self.store_features_in_redis()

            logging.info("Data Processing Completed Successfully.")

        except Exception as e:
            logging.error(f"Error during data processing: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    train_data_path = TRAIN_PATH
    test_data_path = TEST_PATH
    feature_store = RedisFeatureStore()

    data_processor = DataProcessor(train_data_path, test_data_path, feature_store)
    data_processor.run()

    print(data_processor.retrieve_features_redis_store(332))
