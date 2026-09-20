from fastapi.testclient import TestClient
from app import app

# FastAPI test client setup karna
client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Customer Prediction API is running successfully!"}

def test_predict():
    sample_data = {
        "Age": 24,
        "Gender": "Male",
        "Marital_Status": "Single",
        "Occupation": "Student",
        "Monthly_Income": "No Income",
        "Educational_Qualifications": "Post Graduate",
        "Family_size": 3, 
        "Customer_Type": "New"
    }
    
    response = client.post("/predict", json=sample_data)
    
    # Error pakadne ke liye print statement
    print("\n--- FASTAPI KA RESPONSE ---")
    print(response.json())
    print("---------------------------\n")
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "probability_of_yes" in response.json()