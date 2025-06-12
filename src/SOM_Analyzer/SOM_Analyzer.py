# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from math import e
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from logger.logger import LoggerSetup
from src.utils.save_plot import PlotSaver

# LOGGER SETUP
som_logger = LoggerSetup(logger_name = "SOM_Analyzer.py", log_filename_prefix = "SOM_Analyzer").get_logger()

class SOMAnalyzer:
    """
    A class for training and analyzing Self-Organizing Maps (SOMs) on cleaned and feature-engineered data,
    especially designed for medical datasets like the PIMA diabetes dataset.
    
    Attributes:
    
        `dataframe`            (pd.DataFrame)        : The input dataframe with cleaned and engineered features.
        
        `target_col`                (str)            : The name of the target variable column.
        
        `feature_cols`           (List[str])         : The list of selected feature columns to train the SOM.
        
        `som`                     (MiniSom)          : The trained MiniSom object.
    """
    
    def __init__(self, output_dir : str, dataframe : pd.DataFrame, target_col : str = "Outcome",  ) -> None:
        """
        Initialize the SOMAnalyzer class.
        
        Arguments:

            `dataframe`         {pd.DataFrame}          : Cleaned and feature-engineered dataframe.

            `target_col`             {str}              : Name of the target variable. Default is "Outcome".

            `output_dir`             {str}              : Directory to save the SOM grid visualization.
        
        """
        try:
            
            if not isinstance(dataframe, pd.DataFrame):
                som_logger.error("Input must be a pandas DataFrame.")
            
                raise TypeError("Input must be a pandas DataFrame.")
            
            if dataframe.empty:
                som_logger.error("Input DataFrame is empty.")
            
                raise ValueError("Input DataFrame is empty.")
            
            required_columns = ['Glucose', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']
            
            if not all(col in dataframe.columns for col in required_columns):
                som_logger.error(f"Input DataFrame must contain specific columns: {required_columns}.")
            
                raise ValueError(f"Input DataFrame must contain specific columns: {required_columns}.")
            
            self.dataframe         = dataframe.copy()
            self.target_col        = target_col
            self.feature_cols      = ['Glucose_BMI_Interaction', 'Glucose', 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']
            self.scaler            = MinMaxScaler()
            self.som               = None
            self.x_scaled          = None
            self.output_dir        = output_dir
            self.plt_saver         = PlotSaver(output_dir = self.output_dir)

        except Exception as e:
            som_logger.error(f"Error initializing SOMAnalyzer: {repr(e)}")
            
            raise e

    def train_som(self, x_size : int = 10, y_size : int = 10, sigma : float = 1.0, learning_rate : float = 0.5, iterations : int = 1000) -> None:
        """
        Train the Self-Organizing Map (SOM) on the selected feature columns.
        
        Arguments:
            
            x_size                    {int}          : Width of the SOM grid.
            
            y_size                    {int}          : Height of the SOM grid.
            
            sigma                    {float}         : Spread of the neighborhood function.
            
            learning_rate            {float}         : Learning rate for training.
            
            iterations                {int}          : Number of training iterations.
        """

        try:

            if not isinstance(x_size, int) or not isinstance(y_size, int):
                som_logger.error("x_size and y_size must be integers.")
                
                raise TypeError("x_size and y_size must be integers.")
            
            if x_size <= 0 or y_size <= 0:
                som_logger.error("x_size and y_size must be positive integers.")
                
                raise ValueError("x_size and y_size must be positive integers.")
            
            if not isinstance(sigma, (int, float)) or sigma <= 0:
                som_logger.error("sigma must be a positive number.")
                
                raise ValueError("sigma must be a positive number.")
            
            if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
                som_logger.error("learning_rate must be a positive number.")
                
                raise ValueError("learning_rate must be a positive number.")
            
            if not isinstance(iterations, int) or iterations <= 0:
                som_logger.error("iterations must be a positive integer.")
                
                raise ValueError("iterations must be a positive integer.")
        
            X                   = self.dataframe[self.feature_cols].values
            self.x_scaled       = self.scaler.fit_transform(X)
            self.som            = MiniSom(x_size, y_size, self.x_scaled.shape[1], sigma = sigma, learning_rate = learning_rate)
            self.som.random_weights_init(self.x_scaled)
            
            som_logger.info("Training SOM...")
            
            for i in tqdm(range(iterations), desc = "SOM Training", ncols = 100):
                rand_idx        = np.random.randint(0, len(self.x_scaled))
                
                self.som.update(self.x_scaled[rand_idx], self.som.winner(self.x_scaled[rand_idx]), i, iterations)

            som_logger.info("SOM training completed.")

        except Exception as e:
            som_logger.error(f"Error during SOM training: {repr(e)}")
            
            raise e

    def identify_clusters(self) -> pd.Series:
        """
        Identify SOM clusters using Best Matching Units (BMUs).
        
        Returns:
            
            pd.Series               : Cluster labels for each data point.
        
        """
        
        try:

            if self.som is None:
                som_logger.error("SOM has not been trained yet. Call train_som() first.")
                
                raise ValueError("SOM has not been trained yet. Call train_som() first.")
            
            if self.x_scaled is None:
                som_logger.error("Data has not been scaled yet. Call train_som() first.")
                
                raise ValueError("Data has not been scaled yet. Call train_som() first.")
            
            clusters                  = [self.som.winner(x) for x in self.x_scaled]
            cluster_labels            = pd.Series(clusters).astype(str)
            self.dataframe['SOM_Cluster']    = cluster_labels
            som_logger.info("Clusters identified using SOM.")
            
            return self.dataframe['SOM_Cluster']
        
        except Exception as e:
            som_logger.error(f"Error identifying clusters: {repr(e)}")
            
            raise e

    def label_clusters(self) -> pd.Series:
        """
        Label clusters using heuristics based on average glucose and BMI.
        
        Returns:
            
            pd.Series               : Named labels for each SOM cluster.
        
        """
        try: 

            if 'SOM_Cluster' not in self.dataframe.columns:
                som_logger.error("SOM clusters have not been identified yet. Call identify_clusters() first.")
                
                raise ValueError("SOM clusters have not been identified yet. Call identify_clusters() first.")
            
            if self.dataframe.empty:
                som_logger.error("DataFrame is empty. Cannot label clusters.")
                
                raise ValueError("DataFrame is empty. Cannot label clusters.")
            
            cluster_names  = {}
            grouped        = self.dataframe.groupby('SOM_Cluster')[['Glucose', 'BMI']].mean()
            
            for cluster, row in grouped.iterrows():
            
                if row['Glucose'] > 140 and row['BMI'] > 30:
                    label  = "High Risk (Diabetic)"
            
                elif row['Glucose'] > 110:
                    label  = "Pre-diabetic"
            
                else:
                    label  = "Healthy"
            
                cluster_names[cluster] = label
            
            self.dataframe['SOM_Concept'] = self.dataframe['SOM_Cluster'].map(cluster_names)
            som_logger.info("Clusters labeled with concepts.")
            
            return self.dataframe['SOM_Concept']
        
        except Exception as e:
            som_logger.error(f"Error labeling clusters: {repr(e)}")
            
            raise e

    def visualize_som_grid(self):
        """
        Visualize the SOM grid and save the plot.
        
        Arguments:
            
            save_path (str): File path to save the visualization.
        
        """
        
        try:

            if self.som is None:
                som_logger.error("SOM has not been trained yet. Call train_som() first.")
                
                raise ValueError("SOM has not been trained yet. Call train_som() first.")
            
            if self.x_scaled is None:
                som_logger.error("Data has not been scaled yet. Call train_som() first.")
                
                raise ValueError("Data has not been scaled yet. Call train_som() first.")
            
            plt.figure(figsize = (10, 8))
            
            wmap        = {}
            
            for x, t in zip(self.x_scaled, self.dataframe[self.target_col]):
                w       = self.som.winner(x)
                wmap[w] = wmap.get(w, []) + [t]
            
            for x in range(self.som.get_weights().shape[0]):
            
                for y in range(self.som.get_weights().shape[1]):
            
                    plt.text(x + .5, 
                             y + .5, 
                             str(int(np.mean(wmap.get((x, y), [0])))),
                             ha     = 'center', 
                             va     = 'center',
                             bbox   = dict(facecolor  = 'white', 
                                           alpha      = 0.5, 
                                           lw         = 0
                                           ))
            
            plt.pcolor(self.som.distance_map().T, cmap = 'bone_r')
            plt.colorbar(label = 'Distance')
            plt.title("SOM Grid Mapping (Outcome mean per cell)")
            plt.tight_layout()
            
            self.plt_saver.save_plot(plot = plt, plot_name = "som_grid_mapping")
            som_logger.info(f"SOM grid mapping saved")

        except Exception as e:
            som_logger.error(f"Error visualizing SOM grid: {repr(e)}")
            
            raise e
