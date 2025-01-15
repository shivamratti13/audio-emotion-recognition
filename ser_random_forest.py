# ser_random_forest.py

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
X_train = np.load("data/X_train.npy")
X_val = np.load("data/X_val.npy")
X_test = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_val = np.load("data/y_val.npy")
y_test = np.load("data/y_test.npy")
le_classes = np.load("data/le_classes.npy", allow_pickle=True)  # Modified line

# Initialize and train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Validate the model
y_val_pred = rf.predict(X_val)
print("Validation Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=le_classes))

# Test the model
y_test_pred = rf.predict(X_test)
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=le_classes))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_classes, yticklabels=le_classes, cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix - Random Forest')
plt.savefig("confusion_matrix_rf.png")
plt.show()

# Save the trained model
joblib.dump(rf, "models/rf_model.joblib")