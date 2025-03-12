import pandas as pd
import numpy as np
import os
import pickle
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Import XGBoost (ensure it's installed: pip install xgboost)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------
# 1. Load Dataset (Using Training1.csv)
# ---------------------------------------------------------------------
dataset = pd.read_csv('Training.csv')  # Use your new file name

# ---------------------------------------------------------------------
# 2. Basic Data Cleaning (Optional)
# ---------------------------------------------------------------------
print("Missing values per column before cleaning:")
print(dataset.isnull().sum())
dataset = dataset.fillna(0)

# ---------------------------------------------------------------------
# 3. Prepare Features (X) and Labels (y) with Manual Remapping
# ---------------------------------------------------------------------
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

# Instead of using LabelEncoder, create a manual mapping
unique_diseases = sorted(y.unique())
mapping = {disease: idx for idx, disease in enumerate(unique_diseases)}
Y = y.map(mapping).values  # Now Y contains consecutive integers 0, 1, 2, ...
print("Label mapping:", mapping)

# Create a reverse mapping for later use in predictions
diseases_list = {idx: disease for disease, idx in mapping.items()}

# ---------------------------------------------------------------------
# 4. Split dataset
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=20
)

# ---------------------------------------------------------------------
# 5. Train Multiple Models & Evaluate
# ---------------------------------------------------------------------
models = {
    'SVC': SVC(kernel='linear'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'MultinomialNB': MultinomialNB(),
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
}

best_model_name = None
best_accuracy = 0.0
best_model_obj = None

for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(y_test, predictions)
    print(f"{model_name} Confusion Matrix:\n{cm}\n")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model_obj = model

print(f"Best model is {best_model_name} with accuracy {best_accuracy:.4f}")

# ---------------------------------------------------------------------
# 6. Save the Best Model
# ---------------------------------------------------------------------
model_dir = "model"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model_path = os.path.join(model_dir, "treatment_model.pkl")
joblib.dump(best_model_obj, model_path)
print(f"Best model saved to {model_path}")

# ---------------------------------------------------------------------
# 7. Load Supporting CSV Files for Treatment Recommendations
# ---------------------------------------------------------------------
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv("medications.csv")
diets = pd.read_csv("diets.csv")

# ---------------------------------------------------------------------
# 8. Functions for Disease Prediction & Treatment Recommendation
# ---------------------------------------------------------------------
def get_predicted_disease(patient_symptoms):
    """
    Convert a list of symptoms into an input vector,
    then predict the disease using the saved best model.
    """
    input_vector = np.zeros(len(X.columns))
    # Use the columns from the training data for mapping symptoms
    symptoms_dict = {symptom: i for i, symptom in enumerate(X.columns)}
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    model_loaded = joblib.load(model_path)
    predicted_index = model_loaded.predict([input_vector])[0]
    predicted_disease = diseases_list[predicted_index]
    return predicted_disease

def predict_treatment(features):
    """
    Use a 'features' dict (with key "symptoms") to predict the disease,
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
    
    # Fetch treatment details from CSVs
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
    (Optional) Alternate function using the saved model.
    """
    input_vector = np.zeros(len(X.columns))
    symptoms_dict = {symptom: i for i, symptom in enumerate(X.columns)}
    for symptom in patient_symptoms:
        if symptom in symptoms_dict:
            input_vector[symptoms_dict[symptom]] = 1

    best_model_loaded = joblib.load(model_path)
    predicted_index = best_model_loaded.predict([input_vector])[0]
    predicted_disease = diseases_list[predicted_index]
    return predicted_disease

def helper(disease):
    """
    Get additional info (desc, precautions, medications, diet, workout) about a predicted disease.
    """
    desc = description.loc[description['Disease'] == disease, 'Description'].values
    pre = precautions.loc[precautions['Disease'] == disease, [
        'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'
    ]].values
    med = medications.loc[medications['Disease'] == disease, 'Medication'].values
    diet = diets.loc[diets['Disease'] == disease, 'Diet'].values
    wrkout = workout.loc[workout['disease'] == disease, 'workout'].values
    return desc, pre, med, diet, wrkout

# Make these functions importable from this module
__all__ = ['get_predicted_value', 'helper', 'predict_treatment']
