import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

#load the labels and the model/pipeline
labels = joblib.load('../models/target_labels.joblib')
model = joblib.load('../models/ml_pipeline.joblib')

app = FastAPI(title="Telco Customer Churn Prediction API", description="An API for predicting customer churn based on input features.", version="1.0.0")

class Input(BaseModel):
    gender: str
    seniorcitizen: int
    partner: str        
    dependents: str
    tenure: int
    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str
    contract: str
    paperlessbilling: str
    paymentmethod: str
    monthlycharges: float
    totalcharges: float

# creating a get endpoint for the API
@app.get("/health")
def health_check():
    return {"status": "API is running successfully."}

@app.post("/predict")
def predict_churn(ipayload: Input):
    # Convert the input payload to a DataFrame
    input_data = pd.DataFrame([ipayload.model_dump()])

    # Make predictions using the loaded model
    prediction = model.predict(input_data)[0]

    # Convert the prediction back to the original label
    prediction_label = labels.inverse_transform([prediction])[0]

    return {"prediction": prediction_label} 
    
