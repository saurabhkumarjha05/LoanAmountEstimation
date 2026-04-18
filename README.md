# Smart Loan Amount Estimator 🏦

A modern, production-grade Streamlit Web Application that uses a pre-trained Advanced Machine Learning model (CatBoost) to predict the approved loan amount based on user profile data.

## 📁 Project Structure

```
├── models/
│   ├── best_loan_model.pkl    # Pre-trained CatBoostRegressor model
│   ├── preprocessor.pkl       # Sklearn ColumnTransformer
│   ├── feature_names.pkl      # List of feature names
│   └── best_params.json       # Hyperparameters
├── app.py                     # Main Streamlit Application
├── requirements.txt           # Python dependencies
└── README.md                  # Instructions
```

## 🚀 How to Run Locally

1. **Clone or Download the Repository** and navigate to the project directory:
   ```bash
   cd path/to/PBL4thSem
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure Model Files are Present**: 
   Verify that your `models` folder is located in the same directory as `app.py` and contains the `.pkl` and `.json` files.

5. **Run the Streamlit application**:
   ```bash
   streamlit run app.py
   ```

6. **View the App in your Browser**:
   Streamlit will host the app at `http://localhost:8501`.
