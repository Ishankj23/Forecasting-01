import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformationConfig:
    pass

class DataTransformation:
    def __init__(self):
        self.data_transformer_config=DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv('artifacts/train.csv')
            test_df = pd.read_csv('artifacts/test.csv')


            logging.info("load train and test data")

            for df in [train_df, test_df]:
                df['Transaction Date'] = pd.to_datetime(df['Transaction Date'], format='%m/%d/%Y %H:%M', errors='coerce')
                df['Transaction Date'] = df['Transaction Date'].dt.strftime('%m/%d/%Y')


            for df in [train_df, test_df]:
                df['Amount'] = df['Amount'].str.replace(',', '').astype(float)


            for df in [train_df, test_df]:
                if 'Parking' in df.columns:
                    df["Parking"] = df["Parking"].fillna(0)


            for df in [train_df, test_df]:
                if 'Transaction Size (sq.m)' in df.columns:
                    df["Transaction Size (sq.m)"] = df["Transaction Size (sq.m)"].fillna(df["Transaction Size (sq.m)"].mean())


            # Apply transformations for time series data
            train_df = self.prepare_time_series_data(train_df)
            test_df = self.prepare_time_series_data(test_df)

            # Save transformed data
            self.save_transformed_data(train_df, test_df)

            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_time_series_data(self, df):
        """
        Prepares a DataFrame for time series forecasting.
        Args:
        df (DataFrame): The input DataFrame with 'Transaction Date' and 'Amount' columns.
        Returns:
        DataFrame: Transformed DataFrame with 'Transaction Date' as index and 'Amount' processed.
        """
        try:
            # Ensure 'Transaction Date' is in datetime format
            df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

            # Sort by date to ensure correct order for time series analysis
            df = df.sort_values(by='Transaction Date')

            # Set the 'Transaction Date' as the index for time series analysis
            df.set_index('Transaction Date', inplace=True)

            # Interpolate missing values in the 'Amount' column
            df['Amount'] = df['Amount'].interpolate(method='linear')

            logging.info("Prepared data for time series forecasting")
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def save_transformed_data(self, train_df, test_df):
        try:
            train_file_path = os.path.join('artifacts', 'transformed_train.csv')
            test_file_path = os.path.join('artifacts', 'transformed_test.csv')

            train_df.to_csv(train_file_path, index=False)
            test_df.to_csv(test_file_path, index=False)

            logging.info(f"Transformed data saved to: {train_file_path} and {test_file_path}")
        except Exception as e:
            raise CustomException(e, sys)