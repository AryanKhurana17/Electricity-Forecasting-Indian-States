# ETL PIPELINE FOR ELECTRICITY FORECASTING DATA
import os
import sys
import certifi
import pandas as pd
import pymongo
from dotenv import load_dotenv

from electricityforecasting.exception.exception import ElectricityForecastingException
from electricityforecasting.logger.logger import logging

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where()

class ElectricityDataExtract():
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise ElectricityForecastingException(e,sys)
        
    def csv_to_json_converter(self, file_path):
        try:
            logging.info(f"Converting CSV to JSON: {file_path}")
            
            # Read CSV and handle unnamed column
            data = pd.read_csv(file_path, index_col=0) 
            
            data.reset_index(drop=True, inplace=True)
            records = data.to_dict(orient='records')
            logging.info(f"Converted {len(records)} records")
            return records
        except Exception as e:
            raise ElectricityForecastingException(e,sys)
    
    def insert_data_to_mongodb(self, records, database, collection):
        try:
            logging.info(f"Inserting data to MongoDB: {database}.{collection}")
            
            mongo_client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.records = records
            self.database = mongo_client[database]
            self.collection = self.database[collection]
            
            self.collection.insert_many(self.records)
            mongo_client.close()
            
            logging.info(f"Successfully inserted {len(records)} records")
            return len(records)
        
        except Exception as e:
            raise ElectricityForecastingException(e,sys)

# if __name__ == '__main__':
#     FILE_PATH = "/Users/aryankhurana/Electricity-Forecasting-Indian-States/Indias_Electricity_Consumption_Dataset.csv"
#     DATABASE = "electricity_forecasting"
#     COLLECTION = "consumption_data"
    
#     extractor = ElectricityDataExtract()
#     records = extractor.csv_to_json_converter(FILE_PATH)
#     no_of_records = extractor.insert_data_to_mongodb(records, DATABASE, COLLECTION)
#     print(f"Inserted {no_of_records} records successfully")
