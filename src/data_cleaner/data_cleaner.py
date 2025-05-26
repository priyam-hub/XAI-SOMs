# DEPENDENCIES:

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from librosa import ex
import numpy as np
import pandas as pd
from logger.logger import LoggerSetup

# LOGGER SETUP
cleaner_logger = LoggerSetup(logger_name = "data_cleaner.py", log_filename_prefix = "data_cleaner").get_logger()

class DataCleaner:
    """
    A class for performing data cleaning operations on a DataFrame, including
    handling missing values, treating anomalies, removing outliers and duplicates,
    and converting data types.

    Attributes:
    
        `dataframe`           {pd.DataFrame}        : Original raw dataset passed during initialization.
        
        `cleaned_df`          {pd.DataFrame}        : Cleaned version of the dataset after processing.
    
    """

    def __init__(self, dataframe : pd.DataFrame) -> None:
        """
        Initialize the DataCleaner with a raw DataFrame.

        Arguments:

            dataframe         {pd.DataFrame}        : Raw input DataFrame to be cleaned.
        
        """

        if not isinstance(dataframe, pd.DataFrame):
            cleaner_logger.error("Input must be a pandas DataFrame.")
            
            raise TypeError("Input must be a pandas DataFrame.")
        
        if dataframe.empty:
            cleaner_logger.error("Input DataFrame is empty.")
            
            raise ValueError("Input DataFrame is empty.")
        
        if not all(col in dataframe.columns for col in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']):
            cleaner_logger.error("Input DataFrame must contain specific columns: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'].")
            
            raise ValueError("Input DataFrame must contain specific columns: ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'].")
        
        try:

            self.df         = dataframe.copy()
            self.cleaned_df = None

            cleaner_logger.info("DataCleaner initialized successfully with provided DataFrame.")

        except Exception as e:
            cleaner_logger.error(f"Error initializing DataCleaner: {repr(e)}")
            
            raise e

    def _replace_zeros_with_nan(self, columns : list) -> None:
        """
        Replace zeros with NaN for specified columns where zero is considered invalid.

        Arguments:

            `columns`           {list}         : List of column names in which zero values should be replaced.
        
        """

        try:
        
            for col in columns:
                zero_count = (self.df[col] == 0).sum()
                
                if zero_count > 0:
                    cleaner_logger.info(f"Replacing {zero_count} zeros in '{col}' with NaN.")
                    self.df[col].replace(0, np.nan, inplace = True)

            cleaner_logger.info("Zeros replaced with NaN in specified columns.")
        
        except Exception as e:
            cleaner_logger.error(f"Error replacing zeros with NaN: {repr(e)}")
        
            raise e

    def _impute_missing_values(self) -> None:
        """
        Fill missing (NaN) values using the median of each column.
        """
        
        try:
            
            if self.df.isna().sum().sum() == 0:
                cleaner_logger.info("No missing values found in the DataFrame.")
            
                return
            
            cleaner_logger.info("Imputing missing values with median...")
            
            missing_cols   = self.df.columns[self.df.isna().any()].tolist()
            
            for col in missing_cols:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                cleaner_logger.info(f"Filled missing values in '{col}' with median: {median_val}")

            cleaner_logger.info("Missing values imputed successfully.")

        except Exception as e:
            cleaner_logger.error(f"Error imputing missing values: {repr(e)}")
            
            raise e

    def _remove_outliers_iqr(self, columns : list) -> None:
        """
        Remove rows that contain outliers based on the IQR method for specified columns.

        Arguments:

            `columns`          {list}         : List of column names to apply IQR-based outlier detection.
        
        """
        
        try:

            initial_shape  = self.df.shape
            for col in columns:
                Q1         = self.df[col].quantile(0.25)
                Q3         = self.df[col].quantile(0.75)
                IQR        = Q3 - Q1
                lower      = Q1 - 1.5 * IQR
                upper      = Q3 + 1.5 * IQR
                self.df    = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
                
                cleaner_logger.info(f"Removed outliers in '{col}' using IQR. Remaining rows: {self.df.shape[0]}")
            
            final_shape    = self.df.shape
            
            cleaner_logger.info(f"Outlier removal reduced rows from {initial_shape[0]} to {final_shape[0]}.")

        except Exception as e:
            cleaner_logger.error(f"Error removing outliers using IQR: {repr(e)}")
            
            raise e

    def _remove_duplicates(self) -> None:
        """
        Drop duplicate rows from the DataFrame.
        """
        
        try:
            
            if self.df.duplicated().sum() == 0:
                cleaner_logger.info("No duplicate rows found in the DataFrame.")
                
                return
        
            before   = self.df.shape[0]
            self.df.drop_duplicates(inplace = True)
            after    = self.df.shape[0]
            
            cleaner_logger.info(f"Removed {before - after} duplicate rows.")

        except Exception as e:
            cleaner_logger.error(f"Error removing duplicates: {repr(e)}")
            
            raise e

    def _convert_data_types(self) -> None:
        """
        Ensure appropriate data types are set for each column.
        Converts columns to int or float as applicable.
        """

        try:

            if self.df.empty:
                cleaner_logger.info("DataFrame is empty, skipping data type conversion.")
                
                return
            
            cleaner_logger.info("Converting data types for specific columns...")
            
            self.df = self.df.astype({'Pregnancies'               : int,
                                      'Glucose'                   : float,
                                      'BloodPressure'             : float,
                                      'SkinThickness'             : float,
                                      'Insulin'                   : float,
                                      'BMI'                       : float,
                                      'DiabetesPedigreeFunction'  : float,
                                      'Age'                       : int,
                                      'Outcome'                   : int
                                      }
                                      )
            
            cleaner_logger.info("Data types converted where appropriate.")

        except Exception as e:
            cleaner_logger.error(f"Error converting data types: {repr(e)}")
            
            raise e

    def clean_Data(self) -> None:
        """
        Execute all data cleaning steps in sequence:
        - Replace invalid zeros
        - Impute missing values
        - Remove outliers
        - Remove duplicates
        - Convert data types

        Returns:
            
            `cleaned_df`         {pd.DataFrame}        : Cleaned DataFrame after processing.
        
        """

        try:

            if self.cleaned_df is not None:
                cleaner_logger.info("Data has already been cleaned. Returning existing cleaned DataFrame.")
                
                return
            
            cleaner_logger.info("Starting data cleaning process...")

            self._replace_zeros_with_nan(['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'])
            self._impute_missing_values()
            self._remove_outliers_iqr(['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'])
            self._remove_duplicates()
            self._convert_data_types()

            self.cleaned_df = self.df.copy()
            cleaner_logger.info("Data cleaning complete.")

            return self.cleaned_df
        
        except Exception as e:
            cleaner_logger.error(f"Error during data cleaning process: {repr(e)}")
            
            raise e

