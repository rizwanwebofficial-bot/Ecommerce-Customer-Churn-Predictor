import os
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Advanced ML Additions
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def main():
    print("🚀 Initiating Advanced Machine Learning Pipeline...")
    
    data_path = "data/ecommerce_churn.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Could not find dataset at {data_path}.")
        
    try:
        xl = pd.ExcelFile(data_path)
        df = pd.read_excel(data_path, sheet_name='E Comm' if 'E Comm' in xl.sheet_names else xl.sheet_names[-1])
    except Exception:
        df = pd.read_csv(data_path)
        
    if 'CustomerID' in df.columns:
        df = df.drop(columns=['CustomerID'])
        
    print(f"📊 Initial Dataset Class Balance:\n{df['Churn'].value_counts(normalize=True)}")
    
    # 1. Clean Data Types & Fill Missing Fields
    for col in df.columns:
        if df[col].dtype == 'object' or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
            
    # 2. Categorical Encoding
    categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    X = df.drop(columns=['Churn'])
    y = df['Churn'].astype(int)
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Apply SMOTE to the Training Partition
    print("⚖️ Balancing training features using SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"✅ Balanced Training Class Target Counts:\n{y_train_res.value_counts()}")
    
    # 4. Benchmarking Model Options
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost Classifier": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    results_summary = []
    
    for name, model in models.items():
        print(f"\n🏋️‍♂️ Evaluating {name}...")
        
        # 5-Fold Stratified Cross-Validation on Balanced Data
        cv_results = cross_validate(model, X_train_res, y_train_res, cv=5, scoring=['accuracy', 'f1'])
        cv_acc = cv_results['test_accuracy'].mean()
        cv_f1 = cv_results['test_f1'].mean()
        
        # Train on entire Training block to test Holdout Set
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test)
        
        results_summary.append({
            "Model Name": name,
            "5-Fold CV Accuracy": f"{cv_acc:.4f}",
            "5-Fold CV F1-Score": f"{cv_f1:.4f}",
            "Holdout Accuracy": f"{accuracy_score(y_test, y_pred):.4f}",
            "Holdout Precision": f"{precision_score(y_test, y_pred):.4f}",
            "Holdout Recall": f"{recall_score(y_test, y_pred):.4f}",
            "Holdout F1-Score": f"{f1_score(y_test, y_pred):.4f}"
        })
        
        # Save XGBoost as our primary runtime production driver
        if name == "XGBoost Classifier":
            os.makedirs("saved_models", exist_ok=True)
            with open("saved_models/churn_model.pkl", "wb") as f:
                pickle.dump(model, f)
                
    # Display Benchmark Metrics Matrix
    print("\n📈 ====== ALL MODEL PERFORMANCE COMPARISON MATRIX ======")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    with open("saved_models/features.pkl", "wb") as f:
        pickle.dump(list(X.columns), f)
    with open("saved_models/encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)
        
    print("\n💾 Core production model artifacts stored inside 'saved_models/'!")

if __name__ == "__main__":
    main()