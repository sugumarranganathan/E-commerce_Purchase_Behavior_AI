import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

# Load original dataset
df = pd.read_csv("fashion_customer_churn.csv")

# Drop CustomerID if exists
if "CustomerID" in df.columns:
    df = df.drop("CustomerID", axis=1)

# Keep only 12 selected fields + target
selected_features = [
    "Gender",
    "Age",
    "MembershipType",
    "PreferredCategory",
    "TotalOrders",
    "TotalSpent",
    "LastPurchaseDaysAgo",
    "PurchaseFrequencyPerMonth",
    "AppLoginFrequency",
    "CouponUsageCount",
    "ReturnCount",
    "SatisfactionScore"
]

target_col = "Churn"

df = df[selected_features + [target_col]].copy()

# Encode categorical columns like your notebook style
label_encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

if target_col in categorical_cols:
    categorical_cols.remove(target_col)

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df[target_col] = target_encoder.fit_transform(df[target_col].astype(str))

# Save encoders
pickle.dump(label_encoders, open("fashion_rf_label_encoders.pkl", "wb"))
pickle.dump(target_encoder, open("fashion_rf_target_encoder.pkl", "wb"))

# Save preprocessed dataset (optional, mentor-friendly)
df.to_csv("preprocessed_fashion_customer_churn_12fields.csv", index=False)

# Train model
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("fashion_churn_rf_model.pkl", "wb"))

print("\nSaved files:")
print("- fashion_churn_rf_model.pkl")
print("- fashion_rf_label_encoders.pkl")
print("- fashion_rf_target_encoder.pkl")
print("- preprocessed_fashion_customer_churn_12fields.csv")

