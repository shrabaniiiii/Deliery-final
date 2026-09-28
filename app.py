
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score

st.set_page_config(page_title="Delivery Delay Prediction Project Showcase", layout="wide")

st.title("📦 Delivery Delay Prediction Project")
st.markdown("This application showcases the entire process of building a delivery delay prediction model, from data exploration to an interactive prediction interface.")

# --- 1. Data Loading ---
st.header("1. Data Loading")
@st.cache_data # Cache data loading for performance
def load_data():
    df = pd.read_csv('/content/delivery_data.csv')
    return df

df = load_data()
st.subheader("Raw Data (First 5 Rows)")
st.dataframe(df.head())

# --- 2. Data Exploration ---
st.header("2. Data Exploration")

st.subheader("DataFrame Information")
import io
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("Descriptive Statistics")
st.dataframe(df.describe(include='all'))

st.subheader("Missing Values")
st.dataframe(df.isnull().sum())

st.subheader("Duplicate Rows")
st.write(f"Number of duplicate rows: {df.duplicated().sum()}")

# --- 3. Data Preprocessing ---
st.header("3. Data Preprocessing")
st.markdown("Categorical variables are converted to numerical using one-hot encoding.")

@st.cache_data
def preprocess_data(data):
    categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
    df_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop(columns=['ID', 'Reached.on.Time_Y.N'])
    y = df_encoded['Reached.on.Time_Y.N']
    return X, y, df_encoded

X, y, df_encoded = preprocess_data(df)

st.subheader("Encoded Data (First 5 Rows)")
st.dataframe(df_encoded.head())

st.subheader("Features Used for Training")
st.write(X.columns.tolist())

# --- 4. Model Training & Evaluation (using pre-trained model) ---
st.header("4. Model Performance (Logistic Regression)")

# Load the trained model
@st.cache_resource # Cache the model loading
def load_model():
    return joblib.load('logistic_regression_model.sav')

log_reg_model = load_model()

# Split data for evaluation (consistent with training split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred_log_reg = log_reg_model.predict(X_test)
y_pred_proba_log_reg = log_reg_model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
precision_log_reg = precision_score(y_test, y_pred_log_reg)
recall_log_reg = recall_score(y_test, y_pred_log_reg)
f1_log_reg = f1_score(y_test, y_pred_log_reg)
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_pred_proba_log_reg)
auc_log_reg = roc_auc_score(y_test, y_pred_proba_log_reg)

st.subheader("Performance Metrics")
metric_cols = st.columns(4)
metric_cols[0].metric("Accuracy", f"{accuracy_log_reg:.4f}")
metric_cols[1].metric("Precision", f"{precision_log_reg:.4f}")
metric_cols[2].metric("Recall", f"{recall_log_reg:.4f}")
metric_cols[3].metric("F1-Score", f"{f1_log_reg:.4f}")

st.subheader("Confusion Matrix")
fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
sns.heatmap(conf_matrix_log_reg, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted No Delay', 'Predicted Delay'],
            yticklabels=['Actual No Delay', 'Actual Delay'], ax=ax_cm)
ax_cm.set_title('Logistic Regression Confusion Matrix')
ax_cm.set_xlabel('Predicted Label')
ax_cm.set_ylabel('True Label')
st.pyplot(fig_cm)

st.subheader("ROC Curve")
fig_roc, ax_roc = plt.subplots(figsize=(10, 8))
ax_roc.plot(fpr_log_reg, tpr_log_reg, color='blue', label=f'Logistic Regression (AUC = {auc_log_reg:.2f})')
ax_roc.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curve for Logistic Regression Model')
ax_roc.legend()
ax_roc.grid(True)
st.pyplot(fig_roc)

# --- 5. Interactive Prediction Tool ---
st.header("5. Interactive Prediction Tool")
st.markdown("Enter details below to get a real-time prediction for delivery delay.")

def predict_delivery_delay(new_data_df, model, original_df_columns):
    """
    Predicts delivery delay for new data using the trained Logistic Regression model.
    """
    categorical_cols = ['Warehouse_block', 'Mode_of_Shipment', 'Product_importance', 'Gender']
    new_data_encoded = pd.get_dummies(new_data_df, columns=categorical_cols, drop_first=True)

    missing_cols = set(original_df_columns) - set(new_data_encoded.columns)
    for c in missing_cols:
        new_data_encoded[c] = 0

    extra_cols = set(new_data_encoded.columns) - set(original_df_columns)
    new_data_encoded = new_data_encoded.drop(columns=list(extra_cols))

    new_data_processed = new_data_encoded[original_df_columns]

    predictions = model.predict(new_data_processed)
    return pd.Series(predictions, index=new_data_df.index)

# Input widgets for features
col1_input, col2_input = st.columns(2)

with col1_input:
    warehouse_block = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'F'], key='warehouse_block_input')
    mode_of_shipment = st.selectbox("Mode of Shipment", ['Flight', 'Road', 'Ship'], key='mode_of_shipment_input')
    customer_care_calls = st.number_input("Customer Care Calls", min_value=1, max_value=7, value=3, key='customer_care_calls_input')
    customer_rating = st.slider("Customer Rating (1-5)", min_value=1, max_value=5, value=3, key='customer_rating_input')

with col2_input:
    cost_of_the_product = st.number_input("Cost of the Product ($", min_value=96, max_value=310, value=150, key='cost_of_product_input')
    prior_purchases = st.number_input("Prior Purchases", min_value=1, max_value=10, value=2, key='prior_purchases_input')
    product_importance = st.selectbox("Product Importance", ['low', 'medium', 'high'], key='product_importance_input')
    gender = st.radio("Gender", ['F', 'M'], key='gender_input')
    discount_offered = st.number_input("Discount Offered (%)", min_value=0, max_value=65, value=10, key='discount_offered_input')
    weight_in_gms = st.number_input("Weight in Grams", min_value=1000, max_value=7800, value=4000, key='weight_in_gms_input')


# Create a DataFrame from user inputs
input_data = pd.DataFrame({
    'Warehouse_block': [warehouse_block],
    'Mode_of_Shipment': [mode_of_shipment],
    'Customer_care_calls': [customer_care_calls],
    'Customer_rating': [customer_rating],
    'Cost_of_the_Product': [cost_of_the_product],
    'Prior_purchases': [prior_purchases],
    'Product_importance': [product_importance],
    'Gender': [gender],
    'Discount_offered': [discount_offered],
    'Weight_in_gms': [weight_in_gms]
})

# Define the exact columns used during training (X.columns.tolist() from the notebook)
original_df_columns = [
    'Customer_care_calls', 'Customer_rating', 'Cost_of_the_Product',
    'Prior_purchases', 'Discount_offered', 'Weight_in_gms',
    'Warehouse_block_B', 'Warehouse_block_C', 'Warehouse_block_D', 'Warehouse_block_F',
    'Mode_of_Shipment_Road', 'Mode_of_Shipment_Ship',
    'Product_importance_low', 'Product_importance_medium', 'Gender_M'
]

if st.button("Predict Delivery Status", key='predict_button'):
    prediction = predict_delivery_delay(input_data, log_reg_model, original_df_columns)
    if prediction.iloc[0] == 1:
        st.error("### ⚠️ Prediction: Delivery is LIKELY to be DELAYED!")
    else:
        st.success("### ✅ Prediction: Delivery is LIKELY to be ON TIME!")

    st.markdown("--- ")
    st.subheader("Input Data for Prediction")
    st.dataframe(input_data)
