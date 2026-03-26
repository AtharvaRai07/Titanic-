import os
import sys
import pickle
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from config.paths_config import *

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.feature_store import RedisFeatureStore
from config.params_config import params
from sklearn.ensemble import RandomForestClassifier

class ModelTraining:
    def __init__(self, feature_store: RedisFeatureStore, model_save_path: str):
        self.feature_store = feature_store
        self.model_save_path = model_save_path
        self.model = None

        os.makedirs(model_save_path, exist_ok=True)

    def load_data_from_feature_store(self, entity_ids):
        try:
            logging.info("Loading data from feature store...")

            data = []
            for entity_id in entity_ids:
                feature = self.feature_store.retrieve_feature(entity_id)
                data.append(feature)

            return pd.DataFrame(data)

        except Exception as e:
            logging.error(f"Error loading data from feature store: {e}")
            raise CustomException(e, sys)

    def prepare_data(self):
        try:
            entity_ids = self.feature_store.get_all_entity_ids()

            train_entity_ids, test_entity_ids = train_test_split(entity_ids, test_size=0.2, random_state=42)

            train_data = self.load_data_from_feature_store(train_entity_ids)
            test_data = self.load_data_from_feature_store(test_entity_ids)

            X_train = train_data.drop("Survived", axis=1)
            y_train = train_data["Survived"]

            X_test = test_data.drop("Survived", axis=1)
            y_test = test_data["Survived"]

            logging.info("Columns in training data: " + ", ".join(X_train.columns))
            logging.info("Columns in test data: " + ", ".join(test_data.columns))

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise CustomException(e, sys)

    def hyperparameter_tuning(self, X_train, y_train):
        try:
            logging.info("Starting hyperparameter tuning...")

            self.model = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(estimator=self.model, param_distributions=params, n_iter=10, cv=5, scoring='accuracy')
            random_search.fit(X_train, y_train)
            self.model = random_search.best_estimator_

        except Exception as e:
            logging.error(f"Error during hyperparameter tuning: {e}")
            raise CustomException(e, sys)

    def evaluate_and_save(self,X_test, y_test):
        try:
            logging.info("Training the model...")

            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logging.info(f"Model evaluation - Accuracy: {accuracy}, Recall: {recall}, Precision: {precision}, F1 Score: {f1}")

            with open(os.path.join(self.model_save_path, "model_metrics.txt"), 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"F1 Score: {f1}\n")

            logging.info("Saving the model...")

            model_filename = os.path.join(self.model_save_path, "random_forest_model.pkl")

            with open(model_filename, 'wb') as f:
                pickle.dump(self.model, f)

            logging.info(f"Model saved successfully at {model_filename}")

        except Exception as e:
            logging.error(f"Error during model evaluation and saving: {e}")
            raise CustomException(e, sys)

    def run(self):
        try:
            logging.info("Starting model training pipeline...")

            X_train, X_test, y_train, y_test = self.prepare_data()
            self.hyperparameter_tuning(X_train, y_train)
            self.evaluate_and_save(X_test, y_test)

        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    feature_store = RedisFeatureStore()
    model_save_path = MODEL_DIR

    model_trainer = ModelTraining(feature_store, model_save_path)
    model_trainer.run()
