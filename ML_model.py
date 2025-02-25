import numpy as np
import pandas as pd
import pickle

def load_ML_model():
    """
    Loads the trained XGBoost model along with necessary preprocessing components.

    Returns:
    - XGboost_model (XGBClassifier): The trained XGBoost model for churn prediction.
    - Scaler_loaded (StandardScaler): A scaler for normalizing numerical features.
    - columns_mappings (dict): A dictionary mapping categorical features to numerical values.
    - scale_cols (list): A list of numerical columns that require scaling.
    """
    
    with open("saved/best_xgb.pkl", "rb") as f:
        XGboost_model = pickle.load(f)

    with open("saved/scaler.pkl", "rb") as f:
        Scaler_loaded = pickle.load(f)
    
    with open("saved/columns_mappings.pkl", "rb") as f:
        columns_mappings = pickle.load(f)
    
    with open("saved/scale_cols.pkl", "rb") as f:
        scale_cols = pickle.load(f)
    
    return XGboost_model,Scaler_loaded,columns_mappings,scale_cols


def get_model_prediction(llm_result,ml_model,Scaler,columns_mappings,
                         scale_cols):
    """
    Prepares the input data, applies necessary transformations, 
    and predicts customer churn using the trained ML model.

    Args:
    - llm_result (dict): A dictionary containing structured customer data extracted by the LLM.
    - ml_model (XGBClassifier): The trained XGBoost model for prediction.
    - Scaler (StandardScaler): The preloaded scaler for numerical feature normalization.
    - columns_mappings (dict): A mapping dictionary for categorical features.
    - scale_cols (list): List of numerical columns to be scaled.

    Returns:
    - prediction (numpy.ndarray): The model's predicted churn label (0 = No Churn, 1 = Churn).
    - pred_prob (numpy.ndarray): The probability distribution of the prediction.
    """
    
    input_data_df = pd.DataFrame([llm_result])
    input_data_df.columns = input_data_df.columns.str.lower()
    for col, mapping in columns_mappings.items():
        input_data_df[col] = input_data_df[col].map(mapping)
    input_data_df[scale_cols] = Scaler.transform(input_data_df[scale_cols])
    prediction = ml_model.predict(input_data_df)
    pred_prob = ml_model.predict_proba(input_data_df)
    
    return prediction,pred_prob