import os
from pathlib import Path

class Config:
    
    """
    Configuration class for storing credentials, file paths, and URLs.
    It also provides a method to ensure required directories exist.
    """

    DIABETES_EDA_RESULTS_PATH               = "./results/eda_results"
    DIABETES_RAW_DATASET_PATH               = "./data/diabetes.csv"
    DIABETES_CLEANED_DATASET_PATH         = "./data/diabetes_cleaned_data.csv"

    DIABETES_DATASET_NAME                   = "uciml/pima-indians-diabetes-database"
    DIABETES_DATASET_SAVE_PATH              = "./data/"


    @staticmethod
    def setup_directories():
        """
        Ensures that all required directories exist.
        If a directory does not exist, it creates it.
        """
        
        directories = []
        
        for directory in directories:
            
            if not os.path.exists(directory):
            
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            
            else:
                print(f"Directory already exists: {directory}")