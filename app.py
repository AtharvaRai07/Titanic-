import pickle
import os
import sys
import numpy as np
import pandas as pd
from config.paths_config import MODEL_DIR
from pipeline.training_pipeline import TrainingPipeline
from src.logger import logging
from src.exception import CustomException
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='templates', static_folder='templates', static_url_path='')

# Load trained model
model = None
try:
    model_path = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded successfully from {model_path}")
    else:
        logging.warning(f"Model file not found at {model_path}")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")

# Feature names in correct order
FEATURES_NAME = ['Age', 'Fare', 'Pclass', 'Sex', 'Parch', 'SibSp', 'Embarked', 'FamilySize', 'IsAlone', 'HasCabin', 'Title', 'PclassFare', 'AgeFare']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

        data = request.form

        # Extract values from form (handle both uppercase and lowercase keys)
        age = float(data.get('age') or data.get('Age', 0))
        fare = float(data.get('fare') or data.get('Fare', 0))
        pclass = int(data.get('pclass') or data.get('Pclass', 0))
        sex_input = data.get('sex') or data.get('Sex', 'male')
        parch = int(data.get('parch') or data.get('Parch', 0))
        sibsp = int(data.get('sibsp') or data.get('SibSp', 0))
        embarked_input = data.get('embarked') or data.get('Embarked', 'S')
        name_input = data.get('name') or data.get('Name', '')
        cabin_input = data.get('cabin') or data.get('Cabin', '')

        # ========== FEATURE ENGINEERING (matching titanic.ipynb) ==========

        # 1. Encode Sex: male=0, female=1
        sex = 1 if sex_input.lower() == 'female' else 0

        # 2. Encode Embarked: C=0, Q=1, S=2
        embarked_map = {'C': 0, 'Q': 1, 'S': 2}
        embarked = float(embarked_map.get(embarked_input.upper(), 2))

        # 3. Extract Title from Name (matching notebook pattern)
        # Pattern: ' ([A-Za-z]+)\.' -> maps to {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Rare': 4}
        title = 0  # Default to Mr (0)
        if name_input:
            import re
            title_extract = re.search(r' ([A-Za-z]+)\.', name_input)
            if title_extract:
                title_word = title_extract.group(1)
                title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3}
                title = title_map.get(title_word, 4)  # 4 for Rare/other
            else:
                title = 4  # Default to Rare if no title found
        else:
            title = 4  # Default to Rare if no name provided

        # 4. Feature Engineering
        # FamilySize = SibSp + Parch + 1
        family_size = sibsp + parch + 1

        # IsAlone = 1 if FamilySize == 1 else 0
        is_alone = 1 if family_size == 1 else 0

        # HasCabin = 0 if Cabin is null else 1
        has_cabin = 0 if not cabin_input or cabin_input == '' else 1

        # PclassFare = Pclass * Fare
        pclass_fare = float(pclass) * fare

        # AgeFare = Age * Fare
        age_fare = age * fare

        # ========== CREATE FEATURE ARRAY IN CORRECT ORDER ==========
        # Order: ['Age', 'Fare', 'Pclass', 'Sex', 'Parch', 'SibSp', 'Embarked', 'FamilySize', 'IsAlone', 'HasCabin', 'Title', 'PclassFare', 'AgeFare']
        features = np.array([[
            age,              # Age
            fare,             # Fare
            float(pclass),    # Pclass
            float(sex),       # Sex (0 or 1)
            float(parch),     # Parch
            float(sibsp),     # SibSp
            float(embarked),  # Embarked (0, 1, or 2)
            float(family_size),      # FamilySize
            float(is_alone),         # IsAlone
            float(has_cabin),        # HasCabin
            float(title),            # Title (0-4)
            float(pclass_fare),      # PclassFare
            float(age_fare)          # AgeFare
        ]])

        # Make prediction
        prediction = int(model.predict(features)[0])
        probability = float(model.predict_proba(features)[0][1])

        logging.info(f"Prediction made - Input: {features[0]}, Prediction: {prediction}, Probability: {probability}")

        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'success': True
        })

    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True,host = "localhost",port=5000)


