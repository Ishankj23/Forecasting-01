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
            train_df = self.prepare_time_series_data(train_df, 'Transaction Date', 'Amount')
            test_df = self.prepare_time_series_data(test_df, 'Transaction Date', 'Amount')

            # Save transformed data
            self.save_transformed_data(train_df, test_df)

            return train_df, test_df
        except Exception as e:
            raise CustomException(e, sys)

    def prepare_time_series_data(self, df, date_column, value_column):
        """
        Prepare time series data for forecasting.
        Args:
            df (DataFrame): The input DataFrame.
            date_column (str): The column name for date.
            value_column (str): The column name for the value to forecast.
        Returns:
            DataFrame: Transformed DataFrame suitable for time series forecasting.
        """
        try:
            # Ensure 'Transaction Date' is in datetime format
            df[date_column] = pd.to_datetime(df[date_column])

            # Sort by date to ensure correct order for time series analysis
            df = df.sort_values(by=date_column)

            # Set the 'Transaction Date' as the index for time series analysis
            df.set_index(date_column, inplace=True)

            # Interpolate missing values in the 'Amount' column
            df[value_column] = df[value_column].interpolate(method='linear')

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