
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the schizophrenia dataset
df_schizophrenia = pd.read_csv(r"C:\Users\DELL\Desktop\neuroimaging\SchizophreniaSymptomnsData.csv")

# Data cleaning and preprocessing
# Drop non-numeric columns, leaving only the features and the target variable
df_cleaned = df_schizophrenia.drop(columns=["Name", "Gender", "Marital_Status"])

# Identify numeric columns for filling NaNs
numeric_cols = df_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Fill missing values only for numeric columns
for col in numeric_cols:
    if df_cleaned[col].isnull().any():
        median_value = df_cleaned[col].median()  # Calculate median for the numeric column
        df_cleaned[col].fillna(median_value, inplace=True)  # Fill NaNs with the median value

# Check for missing values in numeric columns after filling
print("Missing values after filling:")
print(df_cleaned[numeric_cols].isna().sum())

# Encode the target variable 'Schizophrenia'
y = df_cleaned['Schizophrenia']  # Target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Select features (numeric columns) for training
X = df_cleaned[['Age', 'Fatigue', 'Slowing', 'Pain', 'Hygiene', 'Movement']]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model to a file for later use
model_filename = r"C:\Users\DELL\Desktop\neuroimaging\rf_schizophrenia_model.pkl"
joblib.dump(rf_model, model_filename)

print("Random Forest model for schizophrenia trained and saved successfully!")
