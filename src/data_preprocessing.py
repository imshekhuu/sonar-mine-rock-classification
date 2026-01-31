import pandas as pd
from src.logger import logger
from pathlib import Path


class DataPreprocess:
    def preprocess(self, path):
        logger.info("Data preprocessing is started")

        df = pd.read_csv(path, header=None)

        x = df.drop(columns=60, axis = 1)
        y = df[60]

        logger.info("data preprocessing is completed")
        return x, y
