import os
import pickle
import pandas as pd
import numpy as np
import shap

class ChurnPredictor:
    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir
        self.model = None
        self.features = None
        self.encoders = None
        self.explainer = None

    def load(self):
        """Loads all model artifacts and initializes the SHAP explainer."""
        with open(os.path.join(self.model_dir, "churn_model.pkl"), "rb") as f:
            self.model = pickle.load(f)
        with open(os.path.join(self.model_dir, "features.pkl"), "rb") as f:
            self.features = pickle.load(f)
        with open(os.path.join(self.model_dir, "encoders.pkl"), "rb") as f:
            self.encoders = pickle.load(f)
        
        # Initialize SHAP TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(self.model)

    def preprocess_input(self, raw_df):
        """Ensures incoming UI or Batch CSV data matches training transformations exactly."""
        df = raw_df.copy()
        if 'CustomerID' in df.columns:
            df = df.drop(columns=['CustomerID'])
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])

        # Align columns & handle missing entries
        for col in self.features:
            if col not in df.columns:
                df[col] = np.nan
            
            if col in self.encoders:
                df[col] = df[col].fillna(str(df[col].mode()[0] if not df[col].mode().empty else "Unknown"))
                df[col] = self.encoders[col].transform(df[col].astype(str))
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0.0)

        return df[self.features]

    def predict(self, processed_df):
        """Returns predictions and raw churn probabilities."""
        preds = self.model.predict(processed_df)
        probs = self.model.predict_proba(processed_df)[:, 1]
        return preds, probs

    def explain(self, processed_df):
        """Generates SHAP values for an inference row."""
        shap_values = self.explainer(processed_df)
        return shap_values