# DEPENDENCIES

from config.config import Config
from logger.logger import LoggerSetup
from src.utils.data_loader import DataLoader
from src.utils.download_dataset import DownloadData
from src.SOM_Analyzer.SOM_Analyzer import SOMAnalyzer
from src.data_cleaner.data_cleaner import DataCleaner
from src.feature_engineering.feature_engineering import FeatureEngineer
from src.exploratory_data_analysis.exploratory_data_analyzer import XAI_SOM_EDA

import warnings
warnings.filterwarnings(action = "ignore")

# LOGGER SETUP
main_logger = LoggerSetup(logger_name = "main.py", log_filename_prefix = "main").get_logger()

def main():

    try:

        downloader            = DownloadData(dataset_name   = Config.DIABETES_DATASET_NAME,
                                             download_path  = Config.DIABETES_DATASET_SAVE_PATH,
                                             )

        downloader.download_dataset()

        main_logger.info("Dataset downloaded successfully.")

        dataLoader            = DataLoader()   
        diabetes_raw_df       = dataLoader.data_loader(file_path = Config.DIABETES_RAW_DATASET_PATH)
        main_logger.info("Data loaded successfully")

        data_analyzer         = XAI_SOM_EDA(dataframe  = diabetes_raw_df, 
                                            output_dir = Config.DIABETES_EDA_RESULTS_PATH
                                            )
        data_analyzer.run_all_eda()
        main_logger.info("EDA completed successfully.")

        cleaner               = DataCleaner(dataframe = diabetes_raw_df)
        diabetes_cleaned_df   = cleaner.clean_Data()
        main_logger.info("Data cleaning completed successfully.")
        
        dataLoader.data_saver(dataframe = diabetes_cleaned_df, 
                              file_path = Config.DIABETES_CLEANED_DATASET_PATH
                              )
        main_logger.info("Cleaned data saved successfully.")

        feature_engineer      = FeatureEngineer(dataframe  = diabetes_cleaned_df,
                                                output_dir = Config.RESULTS_PATH
                                                )
        
        featured_df           = feature_engineer.feature_extraction()
        feature_engineer.feature_importance()

        dataLoader.data_saver(dataframe = featured_df,
                              file_path = Config.DIABETES_EXTRACTED_FEATURES_PATH
                              )

        main_logger.info("Feature engineering completed successfully.")


        som_analyzer = SOMAnalyzer(dataframe    = featured_df, 
                                   output_dir   = Config.RESULTS_PATH,
                                   target_col   = 'Outcome'
                                   )
        
        som_analyzer.train_som(x_size         = 10, 
                               y_size         = 10, 
                               sigma          = 1.0, 
                               learning_rate  = 0.5, 
                               iterations     = 1000
                               )
        
        som_analyzer.identify_clusters()
        som_analyzer.label_clusters()
        som_analyzer.visualize_som_grid()

        main_logger.info("SOM analysis completed successfully.")
        main_logger.info("Pipeline executed successfully.")
            
    except Exception as e:
        
        print(f"Error Occurred In PipeLine: {repr(e)}")
        return


if __name__ == "__main__":
    main()