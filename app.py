from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from electricityforecasting.pipeline.prediction_pipeline import StatePredictor
from electricityforecasting.entity import PredictionInput, PredictionOutput
from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.constants.constants import *

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictor and load data
predictor = None
transformed_df = None

class PredictionRequest(BaseModel):
    state_name: str
    start_date: str
    days_ahead: Optional[int] = 7
    historical_data: Optional[List[float]] = None

class PredictionResponse(BaseModel):
    status: str
    state: str
    predictions: Dict[str, float]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

def load_transformed_data():
    """Load the transformed DataFrame"""
    global transformed_df
    try:
        df_path = VALIDATED_DATA_DIR / "clean_data.parquet"
        if df_path.exists():
            transformed_df = pd.read_parquet(df_path)
            # Apply log1p transformation
            import numpy as np
            transformed_df = np.log1p(transformed_df)
            logging.info(f"Loaded transformed DataFrame with shape: {transformed_df.shape}")
    except Exception as e:
        logging.error(f"Failed to load transformed data: {e}")

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global predictor
    try:
        predictor = StatePredictor()
        load_transformed_data()
        logging.info("FastAPI application started successfully")
    except Exception as e:
        logging.error(f"Failed to initialize application: {e}")

@app.get("/")
async def root():
    return {
        "message": "Electricity Consumption Forecasting API",
        "version": API_VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    return {
        "status": "healthy",
        "available_states": predictor.get_available_states(),
        "total_models": len(predictor.get_available_states()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/states")
async def get_available_states():
    if predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    return predictor.get_available_states()

@app.post("/predict", response_model=PredictionResponse)
async def predict_consumption(request: PredictionRequest):
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Predictor not initialized")
        
        # Create prediction input entity
        prediction_input = PredictionInput(
            state_name=request.state_name,
            start_date=request.start_date,
            days_ahead=request.days_ahead,
            historical_data=request.historical_data,
            use_latest_data=transformed_df is not None
        )
        
        # Make prediction
        prediction_output = predictor.predict_future(prediction_input, transformed_df)
        
        # Convert to API response
        return PredictionResponse(
            status=prediction_output.status,
            state=prediction_output.state_name,
            predictions=prediction_output.predictions,
            metadata=prediction_output.metadata,
            error_message=prediction_output.error_message
        )
        
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
