# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from logger.logger import LoggerSetup
from src.utils.save_plot import PlotSaver

# LOGGER SETUP
feature_engineering_logger = LoggerSetup(logger_name = "feature_engineering.py", log_filename_prefix = "feature_engineering").get_logger()


class FeatureEngineer:
    """
    A class for performing feature engineering tasks including:
    - Creating new features based on domain knowledge
    - Plotting feature importance using a Random Forest model

    Attributes:
    
        dataframe          {pd.DataFrame}       : The cleaned input DataFrame.
    
    """

    def __init__(self, dataframe : pd.DataFrame, output_dir : str) -> None:
        """
        Initialize the FeatureEngineer with a cleaned DataFrame.

        Arguments:
        
            dataframe       {pd.DataFrame}       : Cleaned DataFrame to perform feature engineering on.
        
        """
        
        try:
        
            if not isinstance(dataframe, pd.DataFrame):
                feature_engineering_logger.error("Input must be a pandas DataFrame.")
        
                raise TypeError("Input must be a pandas DataFrame.")

            if dataframe.empty:
                feature_engineering_logger.error("Input DataFrame is empty.")
        
                raise ValueError("Input DataFrame is empty.")

            required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        
            if not all(col in dataframe.columns for col in required_columns):
                feature_engineering_logger.error(f"Input DataFrame must contain specific columns: {required_columns}.")
        
                raise ValueError(f"Input DataFrame must contain specific columns: {required_columns}.")
            
            self.df = dataframe.copy()
            self.plt_saver   = PlotSaver(output_dir = output_dir)

            feature_engineering_logger.info("FeatureEngineer initialized successfully with provided DataFrame.")

        except Exception as e:
            feature_engineering_logger.error(f"Error initializing FeatureEngineer: {repr(e)}")
            
            raise e

    def feature_extraction(self) -> pd.DataFrame:
        """
        Create new features from existing columns to enhance model performance.
        Features created:
            - AgeGroup                  : Categorize 'Age' into bins
            - BMI_Category              : Categorical version of BMI based on thresholds
            - Glucose_BMI_Interaction   : Product of Glucose and BMI

        Returns:
            
            pd.DataFrame: DataFrame with new features added.
        
        """

        try:

            if self.df.empty:
                feature_engineering_logger.error("DataFrame is empty. Cannot perform feature extraction.")
            
                raise ValueError("DataFrame is empty. Cannot perform feature extraction.")

            if not all(col in self.df.columns for col in ['Age', 'BMI', 'Glucose']):
                feature_engineering_logger.error("DataFrame must contain 'Age', 'BMI', and 'Glucose' columns for feature extraction.")
            
                raise ValueError("DataFrame must contain 'Age', 'BMI', and 'Glucose' columns for feature extraction.")
            
            feature_engineering_logger.info("Starting feature extraction...")


            self.df['AgeGroup']                = pd.cut(self.df['Age'],
                                                        bins    = [0, 30, 40, 50, 60, 100],
                                                        labels  = ['<30', '30-40', '40-50', '50-60', '60+']
                                                        )

            self.df['BMI_Category']            = pd.cut(self.df['BMI'],
                                                        bins    = [0, 18.5, 24.9, 29.9, 100],
                                                        labels  = ['Underweight', 'Normal', 'Overweight', 'Obese']
                                                        )

            self.df['Glucose_BMI_Interaction'] = self.df['Glucose'] * self.df['BMI']

            feature_engineering_logger.info("Feature extraction complete. New features added: AgeGroup, BMI_Category, Glucose_BMI_Interaction")
            
            return self.df
        
        except Exception as e:
            feature_engineering_logger.error(f"Error during feature extraction: {repr(e)}")
            
            raise e

    def feature_importance(self, target_col : str = 'Outcome') -> None:
        """
        Compute and plot feature importances using a Random Forest Classifier.
        Only numerical features are considered.

        Args:
        
            target_col (str): Target variable column. Default is 'Outcome'.
        
        """

        try:

            if target_col not in self.df.columns:
                feature_engineering_logger.error(f"Target column '{target_col}' not found in DataFrame.")
                
                raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

            if self.df.empty:
                feature_engineering_logger.error("DataFrame is empty. Cannot compute feature importance.")
                
                raise ValueError("DataFrame is empty. Cannot compute feature importance.")

            feature_engineering_logger.info("Calculating feature importance...")

            df_model        = self.df.select_dtypes(include = [np.number]).copy()

            if target_col not in df_model.columns:
                feature_engineering_logger.error(f"Target column '{target_col}' not found in numeric columns.")
                
                raise ValueError(f"Target column '{target_col}' not found in numeric columns.")
            
            X               = df_model.drop(columns = [target_col])
            y               = df_model[target_col]

            model           = RandomForestClassifier(n_estimators = 100, random_state = 42)
            model.fit(X, y)

            importances     = model.feature_importances_
            feature_names   = X.columns
            feat_imp_df     = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
            feat_imp_df     = feat_imp_df.sort_values(by = 'Importance', ascending = False)

            plt.figure(figsize = (10, 6))
            sns.barplot(x = 'Importance', y = 'Feature', data = feat_imp_df, palette = 'viridis')
            plt.title("Feature Importance")
            plt.tight_layout()

            self.plt_saver.save_plot(plot = plt, plot_name = "feature_importance")

            feature_engineering_logger.info(f"Feature importance plot saved.")

        except Exception as e:
            feature_engineering_logger.error(f"Error during feature importance calculation: {repr(e)}")
            
            raise e
