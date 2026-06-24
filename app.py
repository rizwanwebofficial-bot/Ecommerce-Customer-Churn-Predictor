import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from src.predictor import ChurnPredictor

st.set_page_config(page_title="Enterprise Churn Core Engine", layout="wide")

@st.cache_resource
def get_predictor():
    predictor = ChurnPredictor()
    try:
        predictor.load()
        return predictor
    except Exception:
        return None

agent = get_predictor()

if agent is None:
    st.error("❌ Pre-trained components are missing. Run `python train.py` from your terminal first.")
    st.stop()

st.title("🎯 Enterprise E-Commerce Customer Churn Engine")
st.write("Full-spectrum prediction framework featuring SMOTE training alignment and SHAP feature explainability matrices.")

tab1, tab2 = st.tabs(["👤 Individual Customer Real-Time Profiler", "📁 Bulk Batch CSV Data Ingestion"])

# ==================== TAB 1: INDIVIDUAL PROFILER ====================
with tab1:
    st.subheader("Customer Characteristics Inputs (All 19 Features)")
    
    with st.form("individual_form"):
        c1, c2, c3 = st.columns(3)
        
        with c1:
            tenure = c1.number_input("Tenure (Months)", min_value=0, max_value=100, value=15)
            pref_login = c1.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
            city_tier = c1.selectbox("City Tier", [1, 2, 3])
            warehouse_dist = c1.number_input("Warehouse Distance to House", min_value=0, max_value=200, value=15)
            pref_payment = c1.selectbox("Preferred Payment Mode", ["Debit Card", "Credit Card", "E Wallet", "UPI", "Cash on Delivery"])
            gender = c1.selectbox("Gender", ["Female", "Male"])
            
        with c2:
            hours_app = c2.number_input("Hours Spent on App", min_value=0, max_value=24, value=3)
            devices = c2.number_input("Number of Devices Registered", min_value=1, max_value=10, value=3)
            pref_order = c2.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
            satisfaction = c2.slider("Satisfaction Score Rating", 1, 5, 3)
            marital = c2.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            addresses = c2.number_input("Number of Saved Addresses", min_value=1, max_value=50, value=4)
            
        with c3:
            complain = c3.selectbox("Has Active Complaint?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            hike = c3.number_input("Order Amount Hike From Last Year (%)", min_value=0, max_value=100, value=12)
            coupon = c3.number_input("Coupons Used This Month", min_value=0, max_value=50, value=2)
            order_cnt = c3.number_input("Total Orders Placed", min_value=1, max_value=100, value=5)
            day_last_order = c3.number_input("Days Since Last Order", min_value=0, max_value=365, value=4)
            cashback = c3.number_input("Average Cashback Amount", min_value=0, max_value=1000, value=150)
            gift_card = c3.number_input("Gift Cards Redeemed Count", min_value=0, max_value=50, value=1)

        submitted = st.form_submit_button("Generate Predictive Analysis & SHAP Breakdown", type="primary")

    if submitted:
        raw_input_data = {
            'Tenure': tenure, 'PreferredLoginDevice': pref_login, 'CityTier': city_tier,
            'WarehouseToHome': warehouse_dist, 'PreferredPaymentMode': pref_payment, 'Gender': gender,
            'HourSpendOnApp': hours_app, 'NumberOfDeviceRegistered': devices, 'PreferedOrderCat': pref_order,
            'SatisfactionScore': satisfaction, 'MaritalStatus': marital, 'NumberOfAddress': addresses,
            'Complain': complain, 'OrderAmountHikeFromLastYear': hike, 'CouponUsed': coupon,
            'OrderCount': order_cnt, 'DaySinceLastOrder': day_last_order, 'CashbackAmount': cashback,
            'GiftCardCount': gift_card
        }
        
        input_df = pd.DataFrame([raw_input_data])
        processed_df = agent.preprocess_input(input_df)
        pred, prob = agent.predict(processed_df)
        
        st.markdown("### Executive System Determination")
        if pred[0] == 1:
            st.error(f"⚠️ **High Churn Vulnerability Flagged** (Risk Score Check: {prob[0]:.2%})")
        else:
            st.success(f"✅ **Account Profile Classified Safe** (Risk Score Check: {prob[0]:.2%})")
            
        # Generate SHAP Explanation Graphical Layout
        st.markdown("### SHAP Explainability Vector (Why the Model reached this Decision)")
        shap_values = agent.explain(processed_df)
        
        fig, ax = plt.subplots(figsize=(10, 3))
        shap.plots.bar(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig)

# ==================== TAB 2: BATCH INGESTION ====================
with tab2:
    st.subheader("Process Batch File Infrastructure")
    st.write("Upload a `.csv` or `.xlsx` template containing multiple client row indices to get a bulk output report.")
    
    uploaded_file = st.file_uploader("Upload Target Data Document File", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                batch_df = pd.read_excel(uploaded_file)
            else:
                batch_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
            
        st.write(f"📂 File parsing successful. Processing `{batch_df.shape[0]}` customer profiles...")
        
        # Execute Class Pipeline Transformation
        proc_batch = agent.preprocess_input(batch_df)
        batch_preds, batch_probs = agent.predict(proc_batch)
        
        # Append target data predictions
        batch_df['Churn_Prediction'] = np.where(batch_preds == 1, 'Churn Risk', 'Retained')
        batch_df['Churn_Probability'] = batch_probs
        
        st.markdown("### Execution Analytics Overview")
        metrics_col1, metrics_col2 = st.columns(2)
        total_flagged = int(np.sum(batch_preds))
        
        metrics_col1.metric("Total Flagged Risky Profiles", value=total_flagged)
        metrics_col2.metric("Base Percentage Base Influx Risk", value=f"{(total_flagged/len(batch_df)):.1%}")
        
        st.dataframe(batch_df[['CustomerID', 'Churn_Prediction', 'Churn_Probability'] + list(proc_batch.columns[:4])].head(100))
        
        # Export as clean file link
        csv_buffer = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Batch Evaluation Report File",
            data=csv_buffer,
            file_name="Customer_Churn_Inference_Report.csv",
            mime="text/csv"
        )