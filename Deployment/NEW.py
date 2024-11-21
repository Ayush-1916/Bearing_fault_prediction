import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
# Load and prepare data
df1 = pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Normal_Bearing.csv')
df2 = pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/outer_race_fault_test_2.csv')
df3 = pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/outer_race_fault_test_3.csv')
df4 = pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/inner_race_fault.csv')
df5 = pd.read_csv('/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/roller_element_fault.csv')

df = pd.concat([df1, df2, df3, df4, df5])

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
rf_model = RandomForestClassifier().fit(X_train, y_train)

# Flask function to handle dynamic inputs
def process_and_plot(test_no, bearing_no):
    # Load specific test file based on user input
    file_path = f"/Users/ayushkoge/predictive_maintainance_project/bearing_dataset/archive/Time_feature_matrix_Bearing_{bearing_no}_Test_{test_no}.csv"
    test_data = pd.read_csv(file_path, index_col='Unnamed: 0')
    
    # Predict faults
    y_pred_test = rf_model.predict(test_data)
    test_data['Fault'] = y_pred_test
    test_data.index = pd.to_datetime(test_data.index)

    fault_counts = test_data['Fault'].value_counts()

    if len(fault_counts) == 1 and 'Normal' in fault_counts:
        fault_status = "No Fault Detected"
    else:
        detected_faults = [fault for fault in fault_counts.index if fault != 'Normal']
        fault_status = f"Fault Detected: {', '.join(detected_faults)}"
    
    # Subset data by fault type
    norm = test_data[test_data['Fault'] == 'Normal']
    out_race = test_data[test_data['Fault'] == 'Outer Race']
    inner_race = test_data[test_data['Fault'] == 'Inner Race']
    roll_elem = test_data[test_data['Fault'] == 'Roller Element']

    # Generate scatter plot
    col = 'Max'  # Can be changed to any column
    plt.figure(figsize=(10, 5))
    plt.scatter(norm.index, norm[col], label='Normal')
    plt.scatter(out_race.index, out_race[col], label='Outer Race')
    plt.scatter(inner_race.index, inner_race[col], label='Inner Race')
    plt.scatter(roll_elem.index, roll_elem[col], label='Roller Element')
    plt.legend()
    plt.title(f"{col} Scatter Plot for Test {test_no}, Bearing {bearing_no}")
    plt.savefig("static/scatter_plot.png")  # Save the plot as an image

    return fault_status 