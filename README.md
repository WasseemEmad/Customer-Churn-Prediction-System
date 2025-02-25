#Customer Churn Prediction System

## Project Overview

The Customer Churn Prediction System is an AI-powered chatbot designed to help telecom companies predict customer churn. By leveraging Machine Learning (ML) and a Large Language Model (LLM), this system allows marketing teams to interact with the churn prediction model using natural language.

It provides:
✅ Customer churn predictions based on extracted customer details.
✅ Probabilities of churn risk to help businesses take preventive actions.
✅ AI-powered analysis of the prediction to provide insights into customer behavior.

The backend is built using FastAPI, while the frontend is developed with Streamlit for an interactive chatbot experience.

## How It Works

### User Input:

The user enters a natural language description of a customer (e.g., "A long-time customer with a family plan, recently started complaining about pricing.").

### Data Extraction with LLM:

An LLM extracts structured information from the text, converting it into a format suitable for the ML model.
Churn Prediction with ML Model:

The structured data is passed to an XGBoost-based churn prediction model, which predicts whether the customer will churn or stay.

### Probability & Analysis:

The model also returns the probability of churn, and the LLM generates an explanation of the prediction.

### Chatbot Response:

The chatbot presents the prediction, probability, and insights in a human-friendly way.

## Features
✅ Conversational AI Interface – Uses natural language processing to extract customer data.
✅ ML-Powered Churn Prediction – Predicts customer retention likelihood with high accuracy.
✅ LLM-Based Explanation – Provides an AI-generated interpretation of the prediction.
✅ FastAPI Backend – Ensures a scalable and efficient API service.
✅ Streamlit Frontend – Offers a user-friendly chatbot interface for interaction.

## Tech Stack

**Backend:** FastAPI

**Frontend:** Streamlit

**Machine Learning:** XGBoost

**Large Language Model (LLM):** Open-source LLM for feature extraction and analysis

**Data Processing:** Pandas, NumPy, Scikit-learn


Eaxple for the result:
![output 1](https://github.com/user-attachments/assets/16d854c0-eff4-4bea-ba63-e78efeb3efdf)

