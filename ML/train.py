
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score


def load_dataset(path=r"C:\Users\Vishnupriya\Downloads\jithu\ML project\Agentic Ai\CartData.csv"):
 df = pd.read_csv(path)
 df["recovered"] = (
        (df["cart_value"] < 400).astype(int) +
        (df["previous_purchases"] >= 3).astype(int) +
        (df["cart_adds_last_month"] >= 5).astype(int) +
        (df["all_items_in_stock"]).astype(int)
    ) >= 2
 df["recovered"] = df["recovered"].astype(int)
 df["value_per_item"] = df["cart_value"] / df["num_items"].replace(0, 1)
 df["engagement_ratio"] = df["cart_adds_last_month"] / (df["previous_purchases"] + 1)
 df["discount_sensitivity"] = (  df["cart_value"] / (df["previous_purchases"] + 2))
 return df
def train():
 df = load_dataset()
 feature_cols = [
        "cart_value",
        "time_since_abandonment_hours",
        "num_items",
        "value_per_item",
        "previous_purchases",
        "cart_adds_last_month",
        "engagement_ratio",
        "discount_sensitivity",
        "customer_segment",
        "customer_tone",
        "all_items_in_stock" ]
 category_column = ["customer_segment", "customer_tone"]
 X = df[feature_cols]
 y = df["recovered"]
 X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42 )
 model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=8,
        eval_metric="AUC",
        loss_function="Logloss",
        random_seed=42,
        verbose=100
    )
 model.fit( X_train,y_train,eval_set=(X_test, y_test),cat_features=[feature_cols.index(col) for col in category_column])
 preds = model.predict_proba(X_test)[:, 1]
 auc = roc_auc_score(y_test, preds)
 acc = accuracy_score(y_test, (preds > 0.5).astype(int))
 print(f"Model Trained")
 print(f" • Accuracy = {acc:.2f}")
 print(f" • AUC = {auc:.2f}")  
 model_path = "model.joblib"
 meta_path = "feature_metadata.joblib"
 joblib.dump(model, model_path)
 joblib.dump({"features": feature_cols,"categorical": category_column,}, meta_path)
 print(f"saved model to {model_path}")
 print(f"Saved feature metadata to {meta_path}")


if __name__ == "__main__":
    train()
