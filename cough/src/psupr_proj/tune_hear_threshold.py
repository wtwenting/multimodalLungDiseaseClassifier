import numpy as np
import joblib
import tensorflow as tf

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# --------------------------------------------------
# 1. Load VALIDATION embeddings
# --------------------------------------------------

data = np.load("data/hear_embeddings/hear_validation.npz")

X_val = data["embeddings"].astype(np.float32)
labels = data["label"].astype(str)

# Healthy = 0
# COVID-19 = 1
y_val = (labels == "COVID-19").astype(int)


# --------------------------------------------------
# 2. Load scaler
# --------------------------------------------------

scaler = joblib.load(
    "artifacts/hear_dense/scaler.joblib"
)

X_val_scaled = scaler.transform(X_val)


# --------------------------------------------------
# 3. Load trained Dense model
# --------------------------------------------------

model = tf.keras.models.load_model(
    "artifacts/hear_dense/best_model.keras"
)


# --------------------------------------------------
# 4. Predict probabilities ONCE
# --------------------------------------------------

prob = model.predict(
    X_val_scaled,
    verbose=0
).reshape(-1)


# --------------------------------------------------
# 5. Try different thresholds
# --------------------------------------------------

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'COVID-F1':<12}"
    f"{'Macro-F1':<12}"
)

print("-" * 60)


best_threshold = None
best_macro_f1 = -1


for threshold in np.arange(0.30, 0.81, 0.05):

    # Convert probability → class
    y_pred = (prob >= threshold).astype(int)

    # COVID metrics
    precision = precision_score(
        y_val,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        y_pred,
        zero_division=0
    )

    covid_f1 = f1_score(
        y_val,
        y_pred,
        zero_division=0
    )

    # Overall balance between both classes
    macro_f1 = f1_score(
        y_val,
        y_pred,
        average="macro"
    )


    print(
        f"{threshold:<12.2f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{covid_f1:<12.4f}"
        f"{macro_f1:<12.4f}"
    )


    # Find threshold with highest validation Macro-F1
    if macro_f1 > best_macro_f1:

        best_macro_f1 = macro_f1
        best_threshold = threshold


# --------------------------------------------------
# 6. Best threshold
# --------------------------------------------------

print("\n====================================")
print("BEST VALIDATION THRESHOLD")
print("====================================")

print(f"Threshold: {best_threshold:.2f}")
print(f"Macro-F1:  {best_macro_f1:.4f}")


# --------------------------------------------------
# 7. Confusion matrix at selected threshold
# --------------------------------------------------

best_pred = (
    prob >= best_threshold
).astype(int)

cm = confusion_matrix(
    y_val,
    best_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix")
print("[Healthy, COVID-19]")
print(cm)