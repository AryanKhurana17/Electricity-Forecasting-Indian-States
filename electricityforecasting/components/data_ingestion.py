import sys
import certifi
import pandas as pd
import pymongo

from electricityforecasting.logger.logger import logging
from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.entity import DataIngestionConfig, DataIngestionArtifact
from electricityforecasting.utils.common import create_dirs, save_parquet

ca = certifi.where()

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        try:
            self.config = config
            create_dirs([self.config.root_dir])
            
            if not self.config.mongo_url:
                raise ValueError(f"MongoDB URL not found in configuration")
                
        except Exception as e:
            raise ElectricityForecastingException(e, sys)

    def fetch_from_mongo(self) -> pd.DataFrame:
        """Fetch data from MongoDB"""
        try:
            client = pymongo.MongoClient(self.config.mongo_url, tlsCAFile=ca)
            collection = client[self.config.database_name][self.config.collection_name]
            cursor = collection.find({}, {"_id": 0})
            df = pd.DataFrame(list(cursor))
            client.close()
            
            logging.info(f"Fetched {len(df)} records from MongoDB")
            return df
            
        except Exception as e:
            raise ElectricityForecastingException(e, sys)

    def run(self) -> DataIngestionArtifact:
        """Run data ingestion pipeline"""
        try:
            df = self.fetch_from_mongo()
            save_parquet(df, self.config.raw_data_file)
            
            # Create artifact
            artifact = DataIngestionArtifact(
                raw_data_file=self.config.raw_data_file,
                ingestion_status=True,
                total_records=len(df),
                columns=df.columns.tolist(),
                data_summary={
                    "shape": df.shape,
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "dtypes": df.dtypes.to_dict()
                }
            )
            
            logging.info(f"Data ingestion completed successfully. Artifact: {artifact}")
            return artifact
            
        except Exception as e:
            raise ElectricityForecastingException(e,sys)
