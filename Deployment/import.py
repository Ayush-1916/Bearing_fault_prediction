import pickle
from bearing_PCA import pca_3
from bearing_randomforest_classification import rf_model
# Save PCA model
with open('/Users/ayushkoge/Deployment/bearing_PCA.pkl', 'wb') as file:
    pickle.dump(pca_3, file)

# Save Random Forest classifier
with open('/Users/ayushkoge/Deployment/bearing_randomforest_classification.pkl', 'wb') as file:
    pickle.dump(rf_model, file)