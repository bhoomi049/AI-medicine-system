import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------------------------------------------------------------
# 1. Load dataset from the same directory (remove "./backend/")
# ---------------------------------------------------------------------
dataset = pd.read_csv('Training.csv')  # <--- Was './backend/Training.csv'

# Prepare features (X) and labels (y)
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

# Encode labels
le = LabelEncoder()
Y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=20
)

# ---------------------------------------------------------------------
# 2. Train multiple models & check accuracy
# ---------------------------------------------------------------------
models = {
    'SVC': SVC(kernel='linear'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'MultinomialNB': MultinomialNB()
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, predictions)
    print(f"{model_name} Confusion Matrix:\n{cm}\n")

# ---------------------------------------------------------------------
# 3. Train & save the "best" model (example: SVC)
# ---------------------------------------------------------------------
best_model = SVC(kernel='linear')
best_model.fit(X_train, y_train)

model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

joblib.dump(best_model, os.path.join(model_dir, "treatment_model.pkl"))

# ---------------------------------------------------------------------
# 4. Load supporting CSV files from the same directory
# ---------------------------------------------------------------------
sym_des = pd.read_csv("symtoms_df.csv")       # <--- Was "./backend/symtoms_df.csv"
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")

# Symptom & disease mappings
symptoms_dict = {symptom: i for i, symptom in enumerate(X.columns)}
diseases_list = {i: disease for i, disease in enumerate(le.classes_)}

# ---------------------------------------------------------------------
# 5. Functions for disease prediction & treatment recommendation
# ---------------------------------------------------------------------

def get_predicted_disease(patient_symptoms):
    """
    Convert a list of symptoms into an input vector for the model,
    then predict the disease using 'treatment_model.pkl'.
    """
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    model_path = os.path.join(model_dir, "treatment_model.pkl")
    model = joblib.load(model_path)
    predicted_disease_index = model.predict([input_vector])[0]
    predicted_disease = diseases_list[predicted_disease_index]
    return predicted_disease

def predict_treatment(features):
    """
    Use 'features' dict (with key "symptoms") to predict the disease,
    then fetch description, precautions, medications, diet, and workout.
    """
    print("DEBUG: Received features type:", type(features))
    print("DEBUG: Features content:", features)

    if not isinstance(features, dict):
        raise TypeError(f"Expected a dictionary for features, but got {type(features)}")

    patient_symptoms = features.get("symptoms", [])
    print("DEBUG: Extracted symptoms:", patient_symptoms)

    if not isinstance(patient_symptoms, list):
        raise TypeError("Expected a list of symptoms")

    # Predict disease
    predicted_disease = get_predicted_disease(patient_symptoms)

    # Fetch data from CSVs
    desc = description.loc[description['Disease'] == predicted_disease, 'Description'].values
    pre = precautions.loc[precautions['Disease'] == predicted_disease, [
        'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'
    ]].values
    med = medications.loc[medications['Disease'] == predicted_disease, 'Medication'].values
    diet = diets.loc[diets['Disease'] == predicted_disease, 'Diet'].values
    wrkout = workout.loc[workout['disease'] == predicted_disease, 'workout'].values

    return {
        "Disease": predicted_disease,
        "Description": desc[0] if len(desc) > 0 else "No description available",
        "Precautions": pre.tolist() if len(pre) > 0 else ["No precautions available"],
        "Medications": med[0] if len(med) > 0 else "No medication available",
        "Diet": diet[0] if len(diet) > 0 else "No diet available",
        "Workout": wrkout[0] if len(wrkout) > 0 else "No workout available"
    }

def get_predicted_value(patient_symptoms):
    """
    (Optional) Another function that uses svc.pkl instead of the best model.
    """
    input_vector = np.zeros(len(symptoms_dict))
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    with open('svc.pkl', 'rb') as f:
        svc_model = pickle.load(f)

    predicted_disease_index = svc_model.predict([input_vector])[0]
    predicted_disease = diseases_list[predicted_disease_index]
    return predicted_disease

def helper(disease):
    """
    Get additional info (desc, pre, med, diet, wrkout) about a predicted disease.
    """
    desc = description.loc[description['Disease'] == disease, 'Description'].values
    pre = precautions.loc[precautions['Disease'] == disease, [
        'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'
    ]].values
    med = medications.loc[medications['Disease'] == disease, 'Medication'].values
    diet = diets.loc[diets['Disease'] == disease, 'Diet'].values
    wrkout = workout.loc[workout['disease'] == disease, 'workout'].values

    return desc, pre, med, diet, wrkout

# Make these functions importable in other files
__all__ = ['get_predicted_value', 'helper', 'predict_treatment']
