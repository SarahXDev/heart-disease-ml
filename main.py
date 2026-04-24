
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("heart.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ==============================
# 3. Data Visualization (EDA)
# ==============================

plt.figure()
sns.countplot(x="target", data=df)
plt.title("Target Distribution (Heart Disease)")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.show()

# Correlation heatmap
plt.figure()
corr = df.corr()
sns.heatmap(corr, annot=False)
plt.title("Feature Correlation Heatmap")
plt.show()

# ==============================
# 4. Features & Target
# ==============================
X = df.drop("target", axis=1)
y = df["target"]

# ==============================
# 5. Train/Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. Train Logistic Regression Model
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ==============================
# 7. Predictions
# ==============================
y_pred = model.predict(X_test)

# ==============================
# 8. Evaluation
# ==============================
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# ==============================
# 9. Visualization of Results
# ==============================

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 10. Feature Importance (Coefficients)
# ==============================
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients = coefficients.sort_values(by="Coefficient", ascending=False)

plt.figure()
sns.barplot(x="Coefficient", y="Feature", data=coefficients)
plt.title("Feature Importance (Logistic Regression)")
plt.show()
