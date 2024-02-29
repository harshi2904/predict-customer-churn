import pandas as pd
import zipfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Step 1: Extract the dataset from the ZIP archive
with zipfile.ZipFile(r"C:\Users\harsh\OneDrive\Desktop\New folder\archive (3).zip", "r") as zip_ref:
    zip_ref.extractall("customer_churn1")  # Destination folder for extracted files

# Step 2: Load the dataset
data = pd.read_csv("customer_churn1/Churn_Modelling.csv")

# Step 2: Data Preprocessing
data_numeric = data.select_dtypes(include=['number'])

X = data_numeric.drop('Exited', axis=1)
y = data_numeric['Exited']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_testsc = scaler.transform(X_test)

# Step 3: Model Training
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Step 4: Model Evaluation
def evaluate_model(model, X_test, y_test):
    y_predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc_auc = roc_auc_score(y_test, y_predicted)
    
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("ROC AUC Score:", roc_auc)
    print(classification_report(y_test, y_predicted))

print("Logistic Regression Metrics:")
evaluate_model(logistic_model, X_testsc, y_test)

