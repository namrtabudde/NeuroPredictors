import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df_dementia = pd.read_csv(r"C:\Users\DELL\Desktop\neuroimaging\oasis_longitudinal.csv")

# Data cleaning and preprocessing
df_cleaned = df_dementia.drop(columns=["Subject ID", "MRI ID", "Visit", "MR Delay", "Hand"])

# Fill missing values in 'SES' using .loc[] to avoid chained assignment warning
df_cleaned.loc[:, 'SES'] = df_cleaned['SES'].fillna(df_cleaned['SES'].median())

# Dropping rows with missing 'MMSE' values
df_cleaned.dropna(subset=['MMSE'], inplace=True)

# Features and target variable
X = df_cleaned[['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF']]
y = df_cleaned['Group']

# Encode target variable (0 = Non-demented, 1 = Demented, 2 = Converted)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model to a file for later use
model_filename = r"C:\Users\DELL\Desktop\neuroimaging\rf_dementia_model.pkl"
joblib.dump(rf_model, model_filename)

print("Random Forest model trained and saved successfully!")
