from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

app = Flask(__name__)

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv("House Price Prediction Dataset.csv")

# Remove duplicate rows
df = df.drop_duplicates()

# ============================================================
# FEATURES
# Same numeric features used in your Colab
# ============================================================

features = [
    "Id",
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors",
    "YearBuilt"
]

X = df[features]
y = df["Price"]

# ============================================================
# CLASSIFICATION
# Low Price = 0
# High Price = 1
# ============================================================

median_price = y.median()

y_class = (y >= median_price).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)

# Scaling for SVM
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# SVM
# ============================================================

svm_model = SVC(
    kernel="rbf",
    probability=True,
    random_state=42
)

svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)

# ============================================================
# RANDOM FOREST
# ============================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# ============================================================
# DECISION TREE
# ============================================================

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# ============================================================
# CLASSIFICATION RESULTS
# ============================================================

results = {
    "SVM": {
        "accuracy": accuracy_score(y_test, y_pred_svm),
        "precision": precision_score(y_test, y_pred_svm),
        "recall": recall_score(y_test, y_pred_svm),
        "f1": f1_score(y_test, y_pred_svm)
    },

    "Random Forest": {
        "accuracy": accuracy_score(y_test, y_pred_rf),
        "precision": precision_score(y_test, y_pred_rf),
        "recall": recall_score(y_test, y_pred_rf),
        "f1": f1_score(y_test, y_pred_rf)
    },

    "Decision Tree": {
        "accuracy": accuracy_score(y_test, y_pred_dt),
        "precision": precision_score(y_test, y_pred_dt),
        "recall": recall_score(y_test, y_pred_dt),
        "f1": f1_score(y_test, y_pred_dt)
    }
}

# ============================================================
# LINEAR REGRESSION
# ============================================================

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

linear_model = LinearRegression()

linear_model.fit(X_train_reg, y_train_reg)

y_pred_linear = linear_model.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, y_pred_linear)
mse = mean_squared_error(y_test_reg, y_pred_linear)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_reg, y_pred_linear)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        median_price=median_price,
        results=results,
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # Get data from HTML form

    house_id = float(request.form["Id"])
    area = float(request.form["Area"])
    bedrooms = float(request.form["Bedrooms"])
    bathrooms = float(request.form["Bathrooms"])
    floors = float(request.form["Floors"])
    year_built = float(request.form["YearBuilt"])

    # Create input dataframe

    input_data = pd.DataFrame([[
        house_id,
        area,
        bedrooms,
        bathrooms,
        floors,
        year_built
    ]], columns=features)

    # ========================================================
    # SVM PREDICTION
    # ========================================================

    input_scaled = scaler.transform(input_data)

    svm_prediction = svm_model.predict(input_scaled)[0]

    svm_probability = svm_model.predict_proba(input_scaled)[0][1] * 100

    if svm_prediction == 1:
        svm_result = "High Price"
    else:
        svm_result = "Low Price"

    # ========================================================
    # RANDOM FOREST
    # ========================================================

    rf_prediction = rf_model.predict(input_data)[0]

    rf_probability = rf_model.predict_proba(input_data)[0][1] * 100

    if rf_prediction == 1:
        rf_result = "High Price"
    else:
        rf_result = "Low Price"

    # ========================================================
    # DECISION TREE
    # ========================================================

    dt_prediction = dt_model.predict(input_data)[0]

    dt_probability = dt_model.predict_proba(input_data)[0][1] * 100

    if dt_prediction == 1:
        dt_result = "High Price"
    else:
        dt_result = "Low Price"

    # ========================================================
    # LINEAR REGRESSION PRICE
    # ========================================================

    predicted_price = linear_model.predict(input_data)[0]

    # ========================================================
    # ACTUAL PRICE COMPARISON
    # ========================================================

    actual_price = request.form.get("ActualPrice")

    if actual_price and actual_price.strip():

        actual_price = float(actual_price)

        difference = predicted_price - actual_price

        percentage_error = (
            abs(difference) / actual_price
        ) * 100

    else:

        actual_price = None
        difference = None
        percentage_error = None

    return render_template(
        "index.html",

        median_price=median_price,

        results=results,

        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,

        predicted=True,

        svm_result=svm_result,
        svm_probability=svm_probability,

        rf_result=rf_result,
        rf_probability=rf_probability,

        dt_result=dt_result,
        dt_probability=dt_probability,

        predicted_price=predicted_price,

        actual_price=actual_price,
        difference=difference,
        percentage_error=percentage_error,

        form_data=request.form
    )


if __name__ == "__main__":
    app.run(debug=True)