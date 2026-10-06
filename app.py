from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
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
# We only use the features the user enters
# ============================================================

features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Floors"
]

X = df[features]
y = df["Price"]


# ============================================================
# TRAIN / TEST DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        predicted=False,
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        total_records=len(df)
    )


# ============================================================
# HOUSE PRICE PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ----------------------------------------------------
        # GET USER INPUT
        # ----------------------------------------------------

        area = float(request.form["Area"])
        bedrooms = float(request.form["Bedrooms"])
        bathrooms = float(request.form["Bathrooms"])
        floors = float(request.form["Floors"])


        # ----------------------------------------------------
        # BASIC VALIDATION
        # ----------------------------------------------------

        if area <= 0:
            raise ValueError("Area must be greater than 0.")

        if bedrooms <= 0:
            raise ValueError("Bedrooms must be greater than 0.")

        if bathrooms <= 0:
            raise ValueError("Bathrooms must be greater than 0.")

        if floors <= 0:
            raise ValueError("Floors must be greater than 0.")


        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [[
                area,
                bedrooms,
                bathrooms,
                floors
            ]],
            columns=features
        )


        # ----------------------------------------------------
        # PREDICT PRICE
        # ----------------------------------------------------

        predicted_price = model.predict(input_data)[0]


        # Prevent negative prediction
        predicted_price = max(0, predicted_price)


        # ----------------------------------------------------
        # PRICE PER SQ FT
        # ----------------------------------------------------

        price_per_sqft = predicted_price / area


        # ----------------------------------------------------
        # RETURN RESULT
        # ----------------------------------------------------

        return render_template(
            "index.html",

            predicted=True,

            predicted_price=predicted_price,
            price_per_sqft=price_per_sqft,

            area=area,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            floors=floors,

            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,

            total_records=len(df)
        )

    except Exception as e:

        return render_template(
            "index.html",
            predicted=False,
            error=str(e),
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            total_records=len(df)
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)