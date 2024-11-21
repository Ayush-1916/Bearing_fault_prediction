import pickle
import numpy as np
import os
file_path = '/Users/ayushkoge/Deployment/bearing_randomforest_classification.pkl'

# Load pre-trained models
pca = pickle.load(open('Users/ayushkoge/Deployment/bearing_randomforest_classification.pkl', 'rb'))  # Adjust path if needed
rf_model = pickle.load(open('Users/ayushkoge/Deployment/bearing_PCA.pkl', 'rb'))  # Adjust path if needed
#/Users/ayushkoge/Deployment/bearing_randomforest_classificaion.pkl
def predict(values):
    """
    Function to predict the status of the engine based on input features.
    Args:
        values (list): List of input features (RMS, Max, Min, etc.).
    Returns:
        str: Prediction result ("Engine Running Fine!" or "Engine About to Fail!").
    """
    # Convert user input to NumPy array
    input_data = np.array(values).reshape(1, -1)
    
    # Apply PCA for dimensionality reduction
    pca_data = pca.transform(input_data)
    
    # Make prediction using Random Forest
    prediction = rf_model.predict(pca_data)
    
    # Convert numerical prediction to a readable format
    return "Engine Running Fine!" if prediction[0] == 0 else "Engine About to Fail!"