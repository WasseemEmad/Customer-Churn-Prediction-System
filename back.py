from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
import json
import numpy as np
from Models.ML_model import load_ML_model, get_model_prediction
from Models.LLM_model import generate_customer_info, LLM_model, MODEL_ID, clean_llm_data,generate_analysis

app = FastAPI()

class CustomerDescription(BaseModel):
    """
    Defines the expected input format for customer descriptions.

    Attributes:
    - customer_description (str): A natural language description of the customer.
    """
    customer_description: str

ml_model, Scaler_loaded, columns_mappings, scale_cols = load_ML_model()
llm = LLM_model(MODEL_ID, temperature=0.1)


@app.post("/chat")
async def chatbot_response(data: CustomerDescription):
    """
    Handles user queries, extracts structured customer information using an LLM, 
    predicts churn probability, and generates a chatbot-style response.

    Args:
    - data (CustomerDescription): A Pydantic model containing the user-provided customer description.

    Returns:
    - dict: A chatbot response with churn prediction and analysis.
    """
    try:
        extracted_data = generate_customer_info(llm,data.customer_description)
        extracted_data = clean_llm_data(extracted_data)

        if isinstance(extracted_data, str):  
            try:
                extracted_data = json.loads(extracted_data)
            except json.JSONDecodeError:
                return {"bot_response": "Sorry, I couldn't understand that. Can you rephrase?"}

        if "error" in extracted_data:
            return {"bot_response": "I had trouble extracting details. Can you provide more info?"}

        # Get ML prediction
        prediction, probability = get_model_prediction(
            extracted_data, ml_model, Scaler_loaded, columns_mappings, scale_cols
        )

        prediction = prediction.tolist()[0] 
        probability = probability.tolist()[0]  
        analysis = generate_analysis(llm, extracted_data, probability[1])


        if prediction == 0:
            bot_message = f"Based on the details, I think the customer will **stay**. 🟢\n"
            bot_message += f"The probability of the customer to leave is: **{probability[1]*100:.2f}%.**  \n"
            bot_message += analysis
        else:
            bot_message = f"It looks like the customer might **leave**. ⚠️\n"
            bot_message += f"The probability of the customer to leave is: **{probability[1]*100:.2f}%**  \n"
            bot_message += analysis
            
        return {"bot_response": bot_message}

    except Exception as e:
        return {"bot_response": "Oops! Something went wrong. Try again."}