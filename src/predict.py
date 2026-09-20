import os
import pandas as pd
import joblib

def predict_new_customer(customer_data):
    # Dynamically getting model path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, 'models', 'model.pkl')
    
    if not os.path.exists(model_path):
        return "Error: Model file not found. Please run src/train.py first."
        
    # Load the trained pipeline
    model = joblib.load(model_path)
    
    # Convert incoming dictionary (new customer data) to DataFrame
    df_new = pd.DataFrame([customer_data])
    
    # Predict output and probability
    prediction = model.predict(df_new)
    probability = model.predict_proba(df_new)[0][1] # Probability of predicting '1' (Yes)
    
    result = 'Yes' if prediction[0] == 1 else 'No'
    return {"Prediction": result, "Probability of Yes": round(probability, 2)}

if __name__ == "__main__":
    # Example: Ek naye customer ka data
    new_customer = {
        "Age": 24,
        "Gender": "Male",
        "Marital Status": "Single",
        "Occupation": "Student",
        "Monthly Income": "No Income",
        "Educational Qualifications": "Post Graduate",
        "Family size": 3,
        "Customer Type": "New"
    }
    
    print("Incoming New Customer Data:")
    print(new_customer)
    print("-" * 30)
    
    # Function ko call karke result print karna
    output = predict_new_customer(new_customer)
    print(f"Model Result: {output}")