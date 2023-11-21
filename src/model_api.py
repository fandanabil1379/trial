import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Create an instance of FastAPI
app = FastAPI(
    title="Diabetes Prediction", 
    version="v1.0.0"
)

# Define a Pydantic model that represents the data structure
class Diabetes(BaseModel):
    Pregnancies: int
    Glucose: int
    BloodPressure: int
    SkinThickness: int
    Insulin: int
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int   

# Define a Python class to create a list to reformat the data
class Item(BaseModel):
    data: List[Diabetes]
        
# Loading the saved model
model = pickle.load(open('../model/final_model.sav', 'rb'))

# Create a POST endpoint to make prediction
@app.post('/prediction')
async def diabetes_prediction(parameters: Item):
    # Get inputs
    req = parameters.dict()['data']

    # Convert input into Pandas DataFrame
    data = pd.DataFrame(req)

    # Make the predictions
    res = model.predict(data).tolist()
    
    return {"Request": req, "Response": res}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)