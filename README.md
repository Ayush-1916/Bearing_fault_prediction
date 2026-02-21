# Bearing Fault Diagnosis Readme

## Project Overview

This project focuses on detecting bearing faults in rotating machinery using vibration signal analysis and machine learning techniques.

The system classifies bearing conditions into different fault categories based on extracted features from vibration data. It can help in predictive maintenance by identifying potential failures before breakdown occurs.

## Objective:

1. Analyze vibration signal data from bearings
2. Perform feature extraction
3. Train machine learning models for fault classification
4. Evaluate model performance
5. Provide clear fault status output

## Technologies Used:

1. Python
2. NumPy
3. Pandas
4. Scikit-learn
5. Matplotlib / Seaborn
6. Jupyter Notebook

## Dataset

The project uses vibration signal data collected from rotating machinery under different bearing conditions such as:

 1. Normal condition
 2. Inner race fault
 3. Outer race fault
 4. Bearing fault
 
Link to the dataset : https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset

## Methodology:

	1.	Data Loading and Cleaning
  
	2.	Exploratory Data Analysis (EDA)
  
	3.	Feature Engineering
	  •	Statistical features
	  •	Time-domain analysis
    
	4.	Model Training
	  • Logistic Regression
	  • Random Forest
	  • Support Vector Machine
  
	5.	Model Evaluation
	  • Accuracy
	  • Confusion Matrix
	  • Classification Report
  
	6.	Final Model Selection

## Model Output

The trained model predicts the bearing condition and outputs clear status labels such as:

	•	Bearing Running Normally
	•	Inner Race Fault Detected
	•	Outer Race Fault Detected
	•	Ball Fault Detected

## How to run:

1. Run requirements.txt in terminal
2. run app.py inside deployment folder

## Key Learning Outcomes
1. Signal processing basics
2. Feature extraction from vibration data
3. Supervised machine learning for classification
4. Model evaluation techniques
5. Industrial predictive maintenance workflow

## Future Improvements
1. Deep learning-based fault detection
2. Real-time streaming data integration
3. Deployment using Flask or FastAPI
4. Dashboard visualization

Feature extraction techniques are applied to convert raw signals into meaningful numerical inputs for machine learning models.

The folder bearing_dataset contains the whole bearing dataset which i have downloaded from kaggle and files from jupyter notebook on which i have performed Feature engineering , PCA and Model Training. You can access these .ipynp files using jupyter notebook.

Another folder named Deployment contains the python files which i have used to do deployment using flask.


