import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
dataset = pd.read_csv(r"C:\Users\praha\OneDrive\Desktop\House_rent_price\House_Rent_Dataset.csv")

# Drop unwanted columns
dataset = dataset.drop(columns=["Posted On", "Area Locality", "Tenant Preferred", "Point of Contact"])

# Remove outliers in BHK
q1 = dataset["BHK"].quantile(0.25)
q3 = dataset["BHK"].quantile(0.75)
IQR = q3 - q1
max_rng = q3 + (1.5 * IQR)
dataset = dataset[dataset["BHK"] < max_rng]

# Remove outliers in Rent
q1 = dataset["Rent"].quantile(0.25)
q3 = dataset["Rent"].quantile(0.75)
IQR = q3 - q1
max_rng = q3 + (1.5 * IQR)
dataset = dataset[dataset["Rent"] < max_rng]

# Remove outliers in Size
q1 = dataset["Size"].quantile(0.25)
q3 = dataset["Size"].quantile(0.75)
IQR = q3 - q1
max_rng = q3 + (1.5 * IQR)
dataset = dataset[dataset["Size"] < max_rng]

# Remove outliers in Bathroom
q1 = dataset["Bathroom"].quantile(0.25)
q3 = dataset["Bathroom"].quantile(0.75)
IQR = q3 - q1
max_rng = q3 + (1.5 * IQR)
dataset = dataset[dataset["Bathroom"] < max_rng]

# Remove rare categories in City
threshold = 0.01 * len(dataset)
rare_categories = dataset["City"].value_counts()[dataset["City"].value_counts() < threshold].index
dataset["City"] = dataset["City"].replace(rare_categories, "Other")

# Extract numerical floor level
def extract_floor(value):
    if isinstance(value, str):
        match = re.search(r'(\d+)', value)
        if match:
            return int(match.group(1))
        elif "Ground" in value or "Basement" in value:
            return 0
    return np.nan

dataset["Floor_Level"] = dataset["Floor"].apply(extract_floor)
dataset = dataset.dropna(subset=["Floor_Level"])
dataset["Floor_Level"] = dataset["Floor_Level"].astype(int)

# Remove outliers in Floor_Level
q1 = dataset["Floor_Level"].quantile(0.25)
q3 = dataset["Floor_Level"].quantile(0.75)
IQR = q3 - q1
max_rng = q3 + (1.5 * IQR)
dataset = dataset[dataset["Floor_Level"] < max_rng]

# Drop original Floor column
dataset = dataset.drop(columns=["Floor"])

# Ordinal encode Area Type
Oe = OrdinalEncoder(categories=[["Carpet Area", "Built Area", "Super Area"]], handle_unknown="use_encoded_value", unknown_value=-1)
dataset["Area Type"] = Oe.fit_transform(dataset[["Area Type"]])

# Ordinal encode City
oe_city = OrdinalEncoder(categories=[["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Kolkata", "Other"]], handle_unknown="use_encoded_value", unknown_value=-1)
dataset["City"] = oe_city.fit_transform(dataset[["City"]])

# Ordinal encode Furnishing Status
oe_furnish = OrdinalEncoder(categories=[["Unfurnished", "Semi-Furnished", "Furnished"]])
dataset["Furnishing Status"] = oe_furnish.fit_transform(dataset[["Furnishing Status"]])

# Feature Engineering
dataset["Total Rooms"] = dataset["BHK"] + dataset["Bathroom"]
dataset["Log Rent"] = np.log1p(dataset["Rent"])

# One-hot encoding for categorical features
categorical_cols = ["City", "Area Type", "Furnishing Status"]
x = pd.get_dummies(dataset, columns=categorical_cols, drop_first=True)
X = x.drop(columns=["Rent", "Log Rent"])
y = dataset["Log Rent"]

# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=71)

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=500, 
                               max_depth=20, 
                               min_samples_split=4, 
                               random_state=42)
rf_model.fit(x_train, y_train)

print("Random Forest Score:", rf_model.score(x_test, y_test) * 100)

# Save model and scaler
with open("house_rent_rf_model.pkl", "wb") as file:
    pickle.dump(rf_model, file)

with open("house_rent_scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

# Save final columns list to ensure consistency during prediction
final_columns = X.columns.tolist()
with open("final_columns.pkl", "wb") as f:
    pickle.dump(final_columns, f)

print("✅ Random Forest model, scaler, and final columns saved successfully!")
