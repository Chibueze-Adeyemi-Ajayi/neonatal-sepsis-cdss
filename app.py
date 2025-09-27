import numpy as np
import spacy
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
# Initialize the Flask application
app = Flask(__name__)

def convertTextToVec (text):
    # print(text)
    doc = nlp(str(text).lower())
    return doc.vector

def stackVector (vector):
    return np.stack(vector)

def mapLabel (self, text):
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
        
        user_df = pd.DataFrame(symptoms, columns=['symptoms'])
        user_df["input-symptoms"] = user_df["symptoms"].apply(lambda text: convertTextToVec(text))

        print(symptoms)
        
        stacked_symptoms = stackVector(user_df["input-symptoms"])
        
        path="random_forest_model.joblib"
        
        rf_model = joblib.load(path)
        # print(stacked_symptoms)

        predictions = rf_model.predict(stacked_symptoms)

        cds_diagnosis = ["Uncategorized", "Neonatal Sepsis", "Probable NNS", "Late NNS", "Early NNS"]
        
        print(predictions[0])
        
        ailment = cds_diagnosis[predictions[0]]
        
        print("The diagnosed ailment is: ", ailment)

        for prediction in predictions:

            print("The diagnosed ailment is: ", cds_diagnosis[prediction])
        
        return jsonify({'message': 'Symptoms received successfully', 'symptoms': symptoms, 'prediction': ailment})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'An internal server error occurred.'}), 500

nlp = None

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