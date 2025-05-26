# DEPENDENCIES

from config.config import Config
from logger.logger import LoggerSetup
from src.utils.data_loader import DataLoader
from src.utils.download_dataset import DownloadData

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

        # dataLoader            = DataLoader()   
        # diabetes_raw_df       = dataLoader.data_loader(file_path = Config.DIABETES_RAW_DATASET_PATH)
        # main_logger.info("Data loaded successfully:")

    
    except Exception as e:
        
        print(f"Error Occurred In PipeLine: {repr(e)}")
        return


if __name__ == "__main__":
    main()