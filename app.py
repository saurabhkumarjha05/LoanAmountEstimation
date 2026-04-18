import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import warnings
import json
import catboost

# Suppress warnings
warnings.filterwarnings('ignore')

# Set Page Config
st.set_page_config(
    page_title="Loan Amount Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI/UX improvements
st.markdown("""
<style>
    /* Gradient Title */
    .gradient-text {
        background: linear-gradient(45deg, #FF4B4B, #FF904F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Card-style Containers */
    div[data-testid="stExpander"] {
        background-color: #1E1E2E;
        border-radius: 12px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Prominent predict button */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF904F 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        font-size: 1.1rem;
        width: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Hide Streamlit components for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load model artifacts with error handling and caching."""
    model_dir = Path('./models')
    
    if not model_dir.exists():
        st.error(f"Model files not found! Please ensure the 'models' folder exists at {model_dir.absolute()}")
        st.stop()
        
    try:
        model = joblib.load(model_dir / 'best_loan_model.pkl')
        preprocessor = joblib.load(model_dir / 'preprocessor.pkl')
        feature_names = joblib.load(model_dir / 'feature_names.pkl')
        
        best_params = {}
        params_path = model_dir / 'best_params.json'
        if params_path.exists():
            with open(params_path, 'r') as f:
                best_params = json.load(f)
                
        return model, preprocessor, feature_names, best_params
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()

# Load artifacts
model, preprocessor, feature_names, best_params = load_models()

def calculate_eligibility(applicant_income, coapplicant_income, tenure, interest_rate=0.1):
    # Step 1: Total Monthly Income
    total_income = applicant_income + coapplicant_income
    
    # Step 2: FOIR Rule (40% rule)
    max_emi = total_income * 0.4
    
    # Step 3: EMI to Loan Conversion (Approximate formula)
    loan_eligibility = max_emi * tenure * 0.7
    
    # Step 4: Income-Based Caps
    if total_income < 10000:
        loan_cap = 100000
    elif total_income < 50000:
        loan_cap = 500000
    elif total_income < 200000:
        loan_cap = 2500000
    elif total_income < 1000000:
        loan_cap = 20000000
    else:
        loan_cap = 50000000
        
    # Step 5: Final Rule-Based Loan
    rule_based_loan = min(loan_eligibility, loan_cap)
    return rule_based_loan

def calculate_emi(loan_amount, tenure_months, annual_rate=0.1):
    if tenure_months == 0 or loan_amount == 0:
        return 0
    monthly_rate = annual_rate / 12
    emi = loan_amount * monthly_rate * (1 + monthly_rate)**tenure_months / ((1 + monthly_rate)**tenure_months - 1)
    return emi

# --- Sidebar Inputs ---
st.sidebar.markdown("### 📝 Enter Details")

# Group 1: Personal Details
with st.sidebar.expander("👤 Personal Details", expanded=True):
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    married = st.radio("Married", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.radio("Self Employed", ["No", "Yes"])

# Group 2: Financial Details
with st.sidebar.expander("💼 Financial Details", expanded=True):
    applicant_income = st.number_input("Applicant Income (Monthly)", min_value=0.0, value=5000.0, step=500.0, placeholder="e.g. 5000")
    coapplicant_income = st.number_input("Coapplicant Income (Monthly)", min_value=0.0, value=0.0, step=500.0, placeholder="e.g. 2000")

# Group 3: Loan Details
with st.sidebar.expander("🏠 Loan Details", expanded=True):
    requested_loan_amount = st.number_input("Requested Loan Amount", min_value=1.0, value=150.0, step=10.0, help="Required to calculate Income-to-Loan Ratio for the model.")
    loan_amount_term = st.selectbox("Loan Amount Term (Months)", [12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=8)
    credit_history = st.radio("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)")
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.sidebar.button("🚀 Predict Loan Amount")

# --- Main Dashboard ---
st.markdown('<div class="gradient-text">🏦 Smart Loan Amount Estimator</div>', unsafe_allow_html=True)

st.markdown("""
Welcome to the intuitive **Smart Loan Amount Estimator**. This tool uses an advanced **CatBoost Regressor** machine learning model to estimate your eligible loan amount based on your profile inputs.

👈 Please provide your details in the sidebar and click **Predict Loan Amount** to begin.
""")

st.divider()

if predict_btn:
    with st.spinner('Analyzing Profile & Running Inference...'):
        # Prepare input dictionary (matching typical feature names)
        input_data = {
            'Gender': gender,
            'Married': married,
            'Dependents': str(dependents).replace('3+', '3'),
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': applicant_income,
            'CoapplicantIncome': coapplicant_income,
            'Loan_Amount_Term': float(loan_amount_term),
            'Credit_History': float(credit_history),
            'Property_Area': property_area,
            'TotalIncome': applicant_income + coapplicant_income,
            'IncomeToLoanRatio': (applicant_income + coapplicant_income) / requested_loan_amount
        }
        
        # Fallback to keys of input_data
        is_valid_features = feature_names is not None and len(feature_names) > 0
        cols = list(feature_names) if is_valid_features else list(input_data.keys())
        input_df = pd.DataFrame([input_data])
        
        try:
            # Reorder columns just in case
            if is_valid_features and all(f in input_df.columns for f in feature_names):
                input_df = input_df[feature_names]

            # Preprocess
            processed_data = preprocessor.transform(input_df)
            
            # Predict
            prediction = model.predict(processed_data)[0]
            
            # Ensure prediction makes sense
            # The standard dataset target variable is usually in Thousands. We scale it up by 1000 for display.
            predicted_amount = max(0, prediction) * 1000
            
            # Calculate Rule-Based Loan limit
            rule_based_loan = calculate_eligibility(applicant_income, coapplicant_income, float(loan_amount_term))
            
            # Intelligent Hybrid Decision System
            ml_prediction = predicted_amount
            total_income = applicant_income + coapplicant_income
            max_emi_allowed = total_income * 0.35
            
            # Calculate EMI-Based Loan Capacity
            loan_by_emi = max_emi_allowed * float(loan_amount_term) * 0.7
            
            # Final Decision Logic
            if ml_prediction < (loan_by_emi * 0.4):
                final_loan = loan_by_emi * 0.7
                decision_reason = "ML too low → adjusted using EMI capacity"
            elif ml_prediction > rule_based_loan:
                final_loan = min(rule_based_loan, loan_by_emi)
                decision_reason = "ML too high → capped using eligibility"
            else:
                final_loan = min(ml_prediction, loan_by_emi)
                decision_reason = "Balanced ML + EMI logic applied"
            
            # Respect user request
            requested_loan_raw = requested_loan_amount * 1000
            final_loan = min(final_loan, requested_loan_raw * 1.2)
            
            # Hard Safety Rules
            if total_income < 8000:
                final_loan = min(final_loan, 100000)
            
            # Minimum Safety  
            if final_loan < 50000:
                final_loan = 50000
                
            # Realistic Rounding (Nearest thousand)
            final_loan = round(final_loan, -3)
            
            # Recalculate EMI for accurate display
            calculated_emi = calculate_emi(final_loan, float(loan_amount_term))
            income_utilization = (calculated_emi / total_income) * 100 if total_income > 0 else 0
                
            # Label
            if final_loan < 500000:
                label = "Small Loan"
            elif final_loan < 5000000:
                label = "Medium Loan"
            else:
                label = "High Value Loan"
            
            # Calculate range for ML Prediction
            RMSE = 13.29 * 1000
            confidence_interval = 1.5 * RMSE
            lower_bound = max(0, predicted_amount - confidence_interval)
            upper_bound = predicted_amount + confidence_interval
            
            st.success("Analysis Complete! Review your estimated loan details below.")
            
            # Display Result
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Format gracefully
                st.metric(label="Final Loan Amount (Approved)", value=f"₹ {final_loan:,.2f}")
                
            col4, col5 = st.columns(2)
            with col4:
                st.metric(label="ML Prediction", value=f"₹ {predicted_amount:,.2f}")
            with col5:
                st.metric(label="Eligibility Cap", value=f"₹ {rule_based_loan:,.2f}")
                
            col6, col7, col8 = st.columns(3)
            with col6:
                st.metric(label="Estimated EMI", value=f"₹ {calculated_emi:,.2f}/mo")
            with col7:
                st.metric(label="Max Allowed EMI", value=f"₹ {max_emi_allowed:,.2f}/mo")
            with col8:
                st.metric(label="Loan based on EMI", value=f"₹ {loan_by_emi:,.2f}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Decision Reason:** {decision_reason}")
            st.info(f"**Loan Category:** {label}")
            st.info(f"**ML Confidence Range (± 1.5 RMSE):** ₹ {lower_bound:,.2f} — ₹ {upper_bound:,.2f}")
                
            # Allow user to download the report
            report_text = f"""Smart Loan Amount Estimator - Prediction Report
===============================================

👤 Personal Details:
--------------------
Gender:         {gender}
Married:        {married}
Dependents:     {dependents}
Education:      {education}
Self Employed:  {self_employed}

💼 Financial Details:
---------------------
Applicant Income:   {applicant_income}
Coapplicant Income: {coapplicant_income}

🏠 Loan Details:
----------------
Loan Term (Months): {loan_amount_term}
Credit History:     {credit_history}
Property Area:      {property_area}

🏆 Prediction Result:
---------------------
Final Loan Approved: ₹ {final_loan:,.2f}
ML Prediction:       ₹ {predicted_amount:,.2f}
Eligibility Cap:     ₹ {rule_based_loan:,.2f}
Estimated EMI:       ₹ {calculated_emi:,.2f}/month
Max Allowed EMI:     ₹ {max_emi_allowed:,.2f}/month
Income Utilization:  {income_utilization:.1f}%
Decision Reason:     {decision_reason}
Loan Category:       {label}
ML Confidence Range: ₹ {lower_bound:,.2f} to ₹ {upper_bound:,.2f}

-----------------------------------------------
Auto-generated by Smart Loan Amount Estimator
"""
            st.download_button(
                label="📥 Download Prediction Report (.txt)",
                data=report_text,
                file_name="loan_prediction_report.txt",
                mime="text/plain",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"An error occurred during model inference: {str(e)}")

st.divider()

# --- Visual Insights ---
with st.expander("📊 View Model Insights"):
    col_img, col_metrics = st.columns([2, 1])
    
    with col_metrics:
        st.markdown("### Key Metrics")
        st.markdown("""
        - **Algorithm:** CatBoost Regressor
        - **Model Accuracy (R²):** `0.937`
        - **RMSE:** `13.29`
        """)
        
        if best_params:
            st.markdown("### Best Hyperparameters")
            st.json(best_params)
            
    with col_img:
        st.markdown("### Feature Importance")
        try:
            # Assuming catboost models have .feature_importances_
            importance_values = model.feature_importances_
            try:
                # get_feature_names_out() in newer sklearn
                out_features = preprocessor.get_feature_names_out()
            except:
                out_features = [f"Feature {i}" for i in range(len(importance_values))]
                
            importance_df = pd.DataFrame({
                'Feature': out_features,
                'Importance': importance_values
            }).sort_values(by='Importance', ascending=True)
            
            st.bar_chart(importance_df.set_index('Feature'), horizontal=True)
            
        except Exception as e:
            # Fallback static importance if pipeline is tricky to extract from
            st.info("Dynamic feature importance unavailable from the preprocessor pipeline. Showing static feature insights.")
            static_importance = pd.DataFrame({
                'Feature': ['Credit_History', 'ApplicantIncome', 'CoapplicantIncome', 'Loan_Amount_Term', 'Property_Area'],
                'Importance': [0.45, 0.25, 0.15, 0.10, 0.05]
            }).sort_values(by='Importance', ascending=True)
            st.bar_chart(static_importance.set_index('Feature'), horizontal=True)
