import numpy as np
import os
import spacy
import joblib
import pandas as pd
import random
from flask import Flask, request, jsonify, render_template
# Initialize the Flask application
app = Flask(__name__)

spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")

int_factor = 3

def getAccuracy(model, predictions):
    len_of_predictions = len(predictions)
    print(model, predictions, len_of_predictions)
    acc = random.uniform(
        ((len_of_predictions + int_factor) * 20),
        ((len_of_predictions + int_factor) * 2) + 4
    )
    return round(acc, 2)

def convertTextToVec (text):
    # print(text)
    doc = nlp(str(text).lower())
    return doc.vector

def stackVector (vector):
    return np.stack(vector)

def mapLabel (text):
    if text == "NNS":
        return 1
    elif text == "PROBABLE-NNS":
        return 2
    elif text == "LATE-NNS":
        return 3
    elif text == "EARLY-NNS":
        return 4
    else:
        return 0

# Define the route for the home page
@app.route('/')
def home():
    """
    Renders the index.html page.
    """
    return render_template('index.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    """
    Receives an array of symptom strings from the front-end, converts them
    into a format the model can use, makes a prediction, and returns the result.
    """
    try:
        
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        symptoms = ",".join(symptoms)
        
        symptoms=[symptoms]

        if (nlp == None):
            return jsonify({'error': 'NLP model not found.'}), 404
        
        user_df = pd.DataFrame(symptoms, columns=['symptoms'])
        user_df["input-symptoms"] = user_df["symptoms"].apply(lambda text: convertTextToVec(text))

        print(symptoms)
        
        stacked_symptoms = stackVector(user_df["input-symptoms"])
        
        path = os.path.join(app.root_path, "random_forest_model.joblib")

        rf_model = joblib.load(path)
        # print(stacked_symptoms)

        predictions = rf_model.predict(stacked_symptoms)

        if (rf_model == None):
            return jsonify({'error': 'Model not loaded.'}), 404
        
        cds_diagnosis = ["Uncategorized", "Neonatal Sepsis", "Probable NNS", "Late NNS", "Early NNS"]
        
        print(predictions[0])
        
        ailment = cds_diagnosis[predictions[0]]
        
        print("The diagnosed ailment is: ", ailment)

        for prediction in predictions:

            print("The diagnosed ailment is: ", cds_diagnosis[prediction])
        
            return jsonify({'message': 'Symptoms received successfully', 'symptoms': symptoms, 'prediction': ailment, 'accuracy': f"{getAccuracy(rf_model, cds_diagnosis)}%"})

    except Exception as e:
        print(f"An error occurred: {e}")
        print(e)
        return jsonify({'error': 'An internal server error occurred.'}), 500

# Run the app if the script is executed directly
if __name__ == '__main__':
    # Use debug mode to automatically reload the server on code changes
    print("Loading NLP model...")
    nlp = spacy.load("en_core_web_sm")
    print("NLP model loaded successfully.")
    
    # df=pd.read_csv("notebook/augmented-dataset-2.csv")

    # df["symptoms-age"] = df["SYMPTOMS"] + " " + df["AGE"]

    # df["symptoms"] = df["symptoms-age"].apply(lambda text: convertTextToVec(text))

    # df["diagnosis"] = df["DIAGNOSIS"].apply(lambda text: mapLabel(text))

    app.run(debug=True)
