import os, sys, json, joblib, yaml
from typing import List, Any
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException

load_dotenv()  

def create_dirs(paths: List[Path]) -> None:
    try:
        for p in paths:
            p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise ElectricityForecastingException(e, sys)

def save_json(path: Path, data: dict) -> None:
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        raise ElectricityForecastingException(e, sys)

def save_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        raise ElectricityForecastingException(e, sys)

def read_yaml(path: Path) -> Any:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ElectricityForecastingException(e, sys)
