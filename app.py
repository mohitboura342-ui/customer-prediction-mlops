from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# FastAPI app initialize karna
app = FastAPI(title="Customer Prediction API", description="ML Model for predicting customer output")

# Model load karna
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.pkl')

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# API Input ka structure define karna
# URL me spaces allow nahi hote, isliye Python variables me underscore '_' use kiya gaya hai
class CustomerData(BaseModel):
    Age: int
    Gender: str
    Marital_Status: str  
    Occupation: str
    Monthly_Income: str
    Educational_Qualifications: str
    Family_size: int
    Customer_Type: str


# 1. Home Endpoint (Ye sirf check karne ke liye hai)
@app.get("/")
def home():
    return {"message": "Customer Prediction API is running successfully!"}


# 2. Prediction Endpoint (Ye aapka ML model chalayega)
@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    # Input data ko dictionary me convert karna
    input_dict = data.model_dump()
    
    # Hamare training data me columns me spaces the, unhe wapas space format me lana
    input_dict['Marital Status'] = input_dict.pop('Marital_Status')
    input_dict['Monthly Income'] = input_dict.pop('Monthly_Income')
    input_dict['Educational Qualifications'] = input_dict.pop('Educational_Qualifications')
    input_dict['Customer Type'] = input_dict.pop('Customer_Type')
    
    # YAHAN ADD KIYA GAYA HAI: Family_size ko Family size me convert karna
    input_dict['Family size'] = input_dict.pop('Family_size')
    
    # Model hamesha DataFrame expect karta hai
    df_new = pd.DataFrame([input_dict])
    
    # Model se prediction aur probability nikalna
    prediction = model.predict(df_new)
    probability = model.predict_proba(df_new)[0][1]
    
    result = 'Yes' if prediction[0] == 1 else 'No'
    
    # API ka final response
    return {
        "prediction": result,
        "probability_of_yes": round(float(probability), 2)
    }