from src.logger import logging
from src.exception import CustomException
import sys,os
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

## Initialize the data ingestion config
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    row_data_path = os.path.join('artifacts','row.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion methos start')

        try:
            df = pd.read_csv(os.path.join('notebooks/data','gemstone.csv'))
            logging.info('Data read through pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.row_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.row_data_path,index=False)

            logging.info("Train test split start")

            train_set,test_set = train_test_split(df,test_size=0.33,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info('Error occure in data ingestion {e}'.format(e))
            
