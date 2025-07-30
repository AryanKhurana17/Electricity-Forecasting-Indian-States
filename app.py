from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List
import sys

MODEL_DIR = "forecasting_results"

app = FastAPI(
    title="Electricity Consumption Forecaster",
    description="""
This API provides daily electricity consumption predictions for Indian states and regions.

Features:
- Predict consumption for any state/region for a target date using trained LSTM models.
- Provides metadata about available dataset coverage.
- Built for integration with frontend dashboards or other clients.
""",
    version="1.0.1",
    contact={
        "name": "Aryan Khurana",
        "email": "aryan@example.com"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "Prediction",
            "description": "Endpoints for making electricity consumption predictions."
        },
        {
            "name": "Data",
            "description": "Endpoints for dataset metadata information."
        }
    ]
)

# Enable CORS for all origins (adjust in prod as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your prediction pipeline, with error handling
pipeline = None
pipeline_ready = False

try:
    from electricityforecasting.pipeline.prediction_pipeline import PredictionPipeline
    pipeline = PredictionPipeline(model_dir=MODEL_DIR)
    pipeline_ready = True
    print("Prediction pipeline initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize prediction pipeline: {e}")
    pipeline_ready = False


class PredictRequest(BaseModel):
    state_name: str = Field(..., example="Maharashtra", description="State or region to forecast")
    target_date: str = Field(
        ...,
        example="2025-01-01",
        description="Target date for prediction in YYYY-MM-DD format"
    )


class PredictResponse(BaseModel):
    state_name: str
    target_date: str
    predicted_consumption: float
    days_ahead: int
    last_data_date: str


@app.post(
    "/predict/",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Predict electricity consumption for a given state and date",
    description="""
Forecast daily electricity consumption using trained LSTM models.

- `state_name`: Indian state or region name (e.g., Maharashtra)
- `target_date`: Date string in format YYYY-MM-DD for which prediction is made

Responds with predicted consumption in MWh, metadata on days ahead and last data date.
"""
)
async def predict(req: PredictRequest = Body(...)):
    if not pipeline_ready:
        raise HTTPException(
            status_code=503,
            detail="Prediction service not ready. Please run the training pipeline first."
        )
    # Validate date format
    try:
        target_date = datetime.strptime(req.target_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid target_date format. Use YYYY-MM-DD.")

    try:
        # Calculate days ahead
        days_ahead = pipeline.get_days_ahead(req.target_date)
        # Make prediction
        prediction = pipeline.predict_for_date(req.state_name, req.target_date)
        return PredictResponse(
            state_name=req.state_name,
            target_date=req.target_date,
            predicted_consumption=prediction,
            days_ahead=days_ahead,
            last_data_date=pipeline.last_data_date.strftime('%Y-%m-%d')
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model for state '{req.state_name}' not found."
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get(
    "/data-info/",
    tags=["Data"],
    summary="Get dataset metadata",
    description="Returns the last data date, total records, and available states in the dataset."
)
async def get_data_info():
    if not pipeline_ready:
        # Fallback hard-coded response
        return {
            "last_data_date": "2024-09-28",
            "total_records": 4285,
            "available_states": [
                "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
                "Chandigarh", "Chhattisgarh", "DD", "Delhi", "DNH", "DVC",
                "Essar steel", "Goa", "Gujarat", "Haryana", "HP", "J&K",
                "Jharkhand", "Karnataka", "Kerala", "Maharashtra", "Manipur",
                "Meghalaya", "Mizoram", "MP", "Nagaland", "Odisha", "Punjab",
                "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "UP",
                "Uttarakhand", "West Bengal"
            ]
        }

    try:
        return pipeline.get_data_info()
    except Exception as e:
        return {
            "last_data_date": "2024-09-28",
            "total_records": 4285,
            "available_states": ["Maharashtra", "Gujarat", "Tamil Nadu"]
        }


@app.get("/", tags=["Root"])
async def root():
    status = "ready" if pipeline_ready else "not ready - run training pipeline first"
    return {
        "message": f"Electricity Consumption Forecaster API - {status}",
        "endpoints": {
            "predict": "/predict/",
            "data_info": "/data-info/",
            "documentation": "/docs"
        }
    }
