# DEPENDENCIES

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
from math import pi
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

from logger.logger import LoggerSetup
from src.utils.save_plot import PlotSaver

# LOGGER SETUP
eda_logger = LoggerSetup(logger_name = "exploratory_data_analyzer.py", log_filename_prefix = "exploratory_data_analyzer").get_logger()

class XAI_SOM_EDA:
    
    """
    A class for comprehensive Exploratory Data Analysis (EDA) of Medium Article Recommendation Dataset
    
    Attributes:

        `df`                     {pd.DataFrame}         : DataFrame containing text and emotion data.

        `output_dir`                  {str}             : Optional directory to save generated plots.
    
    """

    def __init__(self, dataframe : pd.DataFrame, output_dir : str = None) -> None:
        """
        Initialize the XAI-SOM_EDA class.
        
        Arguments:

            `df`                   {pd.DataFrame}         : Input DataFrame with text and emotion data.
            
            `output_dir`                {str}             : Optional path to save plots.
        
        Raises:

            ValueError                                    : If required columns are not found in the DataFrame.
        
        Returns:

            None
        
        """
        
        try:

            if not isinstance(dataframe, pd.DataFrame):
                eda_logger.error("Input data is not a pandas DataFrame")
                
                raise 

            else:
                eda_logger.info(f"DataFrame Loaded with shape: {dataframe.shape}")

            self.df               = dataframe
            self.output_dir       = output_dir
            self.plt_saver        = PlotSaver(output_dir = output_dir)

            if output_dir:
                Path(output_dir).mkdir(parents = True, exist_ok = True)

                eda_logger.info(f"Output directory created: {output_dir}")

            eda_logger.info(f"XAI-SOM_EDA Class initialized")

        except Exception as e:
            eda_logger.error(f"Error initializing EmotionEDA: {repr(e)}")
            
            raise

    def plot_feature_histograms(self) -> None:
        """
        Create and save a single plot with histograms for all numerical features.
        """
        
        try:
            numeric_cols   = self.df.select_dtypes(include = [np.number]).columns.tolist()
            n_cols         = 3
            n_rows         = int(np.ceil(len(numeric_cols) / n_cols))

            fig, axes      = plt.subplots(n_rows, n_cols, 
                                          figsize    = (18, 5 * n_rows), 
                                          facecolor  = 'white'
                                          )
            
            axes           = axes.flatten()

            sns.set_style("whitegrid")

            for i, col in enumerate(numeric_cols):
                sns.histplot(self.df[col], 
                             bins       = 30, 
                             kde        = True, 
                             ax         = axes[i], 
                             color      = sns.color_palette("coolwarm", len(numeric_cols))[i % len(numeric_cols)],
                             edgecolor  = 'black'
                             )
                
                axes[i].set_title(f"{col}", fontsize = 14, fontweight = 'bold')
                axes[i].set_xlabel("Value", fontsize = 12)
                axes[i].set_ylabel("Frequency", fontsize = 12)
                axes[i].tick_params(axis = 'both', labelsize = 10)
                sns.despine(ax = axes[i])

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle("Histograms of Numerical Features", fontsize = 18, fontweight = 'bold')
            plt.tight_layout(rect = [0, 0, 1, 0.98])
            self.plt_saver.save_plot(plot = fig, plot_name = "feature_histograms")

            eda_logger.info("Histogram plot saved successfully.")

        except Exception as e:
            eda_logger.error(f"Error in plot_feature_histograms: {repr(e)}")

            raise

    def plot_feature_boxplots(self) -> None:
        """
        Create and save a single plot with beautifully styled boxplots for all numerical features.
        """
        
        try:
        
            numeric_cols   = self.df.select_dtypes(include = [np.number]).columns.tolist()
            n_cols         = 3
            n_rows         = int(np.ceil(len(numeric_cols) / n_cols))

            fig, axes      = plt.subplots(n_rows, n_cols, figsize = (18, 5 * n_rows), facecolor = 'white')
            axes           = axes.flatten()

            sns.set_style("whitegrid")

            for i, col in enumerate(numeric_cols):
                sns.boxplot(y      = self.df[col],
                            ax     = axes[i],
                            color  = sns.color_palette("Set2")[i % len(sns.color_palette("Set2"))],
                            width  = 0.4,fliersize=4
                            )
                
                axes[i].set_title(f"{col}", fontsize = 14, fontweight = 'bold')
                axes[i].set_ylabel("Value", fontsize = 12)
                axes[i].tick_params(axis = 'y', labelsize = 10)
                sns.despine(ax = axes[i], top = True, right = True)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.suptitle("Boxplots of Numerical Features", fontsize = 18, fontweight = 'bold')
            plt.tight_layout(rect = [0, 0, 1, 0.98])
            
            self.plt_saver.save_plot(plot = fig, plot_name = "feature_boxplots")

            eda_logger.info("Beautiful boxplot figure saved successfully.")

        except Exception as e:
            eda_logger.error(f"Error in plot_feature_boxplots: {repr(e)}")
            
            raise

    def plot_target_balance(self, target_col: str = "Outcome") -> None:
        """
        Plot and save a beautifully styled class balance of the target variable.
        
        Arguments:
        
            `target_col`         {str}        : Column name of the target class (default is 'Outcome')
        
        """
        
        try:
        
            class_counts = self.df[target_col].value_counts().sort_index()
            class_labels = class_counts.index.tolist()
            class_values = class_counts.values
            total        = class_values.sum()

            fig, ax      = plt.subplots(figsize = (8, 5))
            sns.set_style("whitegrid")

            bars         = sns.barplot(x        = class_labels,
                                       y        = class_values,
                                       palette  = "Set2",
                                       ax       = ax
                                       )

            ax.set_title(f"Class Balance of '{target_col}'", fontsize = 16, fontweight = 'bold')
            ax.set_xlabel("Class", fontsize = 13)
            ax.set_ylabel("Count", fontsize = 13)
            ax.tick_params(axis = 'both', labelsize = 11)

            for i, bar in enumerate(bars.patches):
                count   = int(bar.get_height())
                percent = (count / total) * 100
                ax.text(bar.get_x() + bar.get_width() / 2,
                        count + 2,
                        f"{count} ({percent:.1f}%)",
                        ha          = 'center',
                        fontsize    = 11,
                        fontweight  = 'semibold'
                        )

            sns.despine(top = True, right = True)
            plt.tight_layout()
            
            self.plt_saver.save_plot(plot = fig, plot_name = "target_class_balance")
            
            eda_logger.info("Beautiful target class balance plot saved.")

        except Exception as e:
            eda_logger.error(f"Error in plot_target_balance: {repr(e)}")
            
            raise

    def plot_correlation_heatmap(self) -> None:
        """
        Plot and save a heatmap showing the correlations between all numerical features.
        """
        
        try:
            
            plt.figure(figsize=(10, 8))
            corr = self.df.corr()
            
            sns.heatmap(corr, 
                        annot        = True, 
                        fmt          = ".2f", 
                        cmap         = "coolwarm", 
                        square       = True, 
                        linewidths   = 0.5
                        )
            
            plt.title("Correlation Heatmap of Features", fontsize = 14, fontweight = 'bold')
            plt.tight_layout()
            
            self.plt_saver.save_plot(plot = plt.gcf(), plot_name = "correlation_heatmap")
            
            eda_logger.info("Correlation heatmap saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_correlation_heatmap: {repr(e)}")
        
            raise

    def plot_pairplot_by_outcome(self) -> None:
        """
        Create a pairplot of selected features colored by the Outcome class.
        """
        
        try:
        
            selected = ['Glucose', 'BMI', 'Age', 'Insulin', 'Outcome']
        
            sns.pairplot(self.df[selected], 
                         hue        = 'Outcome', 
                         palette    = 'Set1', 
                         diag_kind  = 'kde'
                         )
        
            plt.suptitle("Pairplot of Selected Features by Outcome", fontsize = 16, y = 1.02)
            plt.tight_layout()
        
            self.plt_saver.save_plot(plot = plt.gcf(), plot_name = "pairplot_by_outcome")
        
            eda_logger.info("Pairplot by outcome saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_pairplot_by_outcome: {repr(e)}")
        
            raise
    
    def plot_violin_by_outcome(self) -> None:
        """
        Plot violin plots of numerical features split by Outcome.
        """
        
        try:
        
            features  = ['Glucose', 'BMI', 'Age', 'BloodPressure']
            n_cols    = 2
            n_rows    = int(np.ceil(len(features) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
            axes      = axes.flatten()

            for i, col in enumerate(features):
                sns.violinplot(x        = "Outcome", 
                               y        = col, 
                               data     = self.df, 
                               ax       = axes[i], 
                               palette  = "Pastel1"
                               )
            
                axes[i].set_title(f"{col} Distribution by Outcome")

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            
            self.plt_saver.save_plot(plot = fig, plot_name = "violin_by_outcome")
            
            eda_logger.info("Violin plots by outcome saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_violin_by_outcome: {repr(e)}")
        
            raise

    def plot_age_distribution_by_class(self) -> None:
        """
        Plot age distribution split by diabetes outcome.
        """
        
        try:
        
            plt.figure(figsize = (8, 5))
        
            sns.histplot(data      = self.df, 
                         x         = "Age", 
                         hue       = "Outcome", 
                         bins      = 20, 
                         palette   = "Set2", 
                         kde       = True, 
                         element   = "step", 
                         stat      = "density"
                         )
        
            plt.title("Age Distribution by Outcome", fontsize = 14, fontweight = 'bold')
            plt.xlabel("Age")
            plt.ylabel("Density")
            plt.tight_layout()
        
            self.plt_saver.save_plot(plot = plt.gcf(), plot_name = "age_distribution_by_outcome")
        
            eda_logger.info("Age distribution by outcome plot saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_age_distribution_by_class: {repr(e)}")
        
            raise

    def plot_bmi_vs_glucose(self) -> None:
        """
        Create a scatter plot of BMI vs Glucose colored by Outcome.
        """
        
        try:
        
            plt.figure(figsize = (8, 6))
        
            sns.scatterplot(data      = self.df,
                            x         = "Glucose",
                            y         = "BMI",
                            hue       = "Outcome",
                            palette   = "Dark2",
                            style     = "Outcome",
                            s         = 70,
                            alpha     = 0.7
                            )
        
            plt.title("BMI vs Glucose by Outcome", fontsize = 14, fontweight = 'bold')
            plt.xlabel("Glucose Level")
            plt.ylabel("BMI")
            plt.tight_layout()
        
            self.plt_saver.save_plot(plot = plt.gcf(), plot_name = "bmi_vs_glucose_by_outcome")
        
            eda_logger.info("Scatterplot of BMI vs Glucose saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_bmi_vs_glucose: {repr(e)}")
        
            raise

    def plot_radar_by_outcome(self) -> None:
        """
        Create a radar chart comparing mean values of features for each class in Outcome.
        """
        try:
            
            features     = self.df.columns.difference(["Outcome"])
            grouped      = self.df.groupby("Outcome")[features].mean()

            categories   = features.tolist()
            N            = len(categories)
            angles       = [n / float(N) * 2 * pi for n in range(N)]
            angles      += angles[:1]     

            fig, ax      = plt.subplots(figsize    = (8, 8), 
                                        subplot_kw = dict(polar = True)
                                        )
            
            colors       = ['#1f77b4', '#ff7f0e']

            for i, (label, row) in enumerate(grouped.iterrows()):
                values   = row.tolist()
                values  += values[:1]
                
                ax.plot(angles, 
                        values, 
                        label     = f'Outcome {label}', 
                        color     = colors[i], 
                        linewidth = 2
                        )
                
                ax.fill(angles, 
                        values, 
                        alpha  = 0.25, 
                        color  = colors[i]
                        )

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize = 10)
            ax.set_title("Radar Chart of Mean Feature Values by Outcome", fontsize = 14, weight = 'bold')
            ax.legend(loc = 'upper right', bbox_to_anchor = (1.1, 1.1))

            plt.tight_layout()
            
            self.plt_saver.save_plot(plot = fig, plot_name = "radar_by_outcome")
            
            eda_logger.info("Radar chart by outcome saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_radar_by_outcome: {repr(e)}")
        
            raise

    def plot_missing_zero_heatmap(self) -> None:
        """
        Show heatmap of features with missing (zero) values.
        """
        
        try:
        
            zero_df   = self.df.copy()
            zero_df   = zero_df.replace(0, np.nan)
            mask      = zero_df.isnull()

            plt.figure(figsize = (10, 6))
        
            sns.heatmap(mask, 
                        cbar  = False, 
                        cmap  = 'viridis'
                        )
        
            plt.title("Heatmap of Missing or Zero Values", fontsize = 14, weight = 'bold')
            plt.xlabel("Features")
            plt.ylabel("Observations")
            plt.tight_layout()
        
            self.plt_saver.save_plot(plot = plt.gcf(), plot_name = "missing_zero_heatmap")
            eda_logger.info("Missing/zero heatmap saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_missing_zero_heatmap: {repr(e)}")
        
            raise

    def plot_age_group_vs_outcome(self) -> None:
        """
        Show stacked bar plot of age groups vs diabetes outcome.
        """
        
        try:
        
            df_copy              = self.df.copy()
            df_copy["AgeGroup"]  = pd.cut(df_copy["Age"], bins = [20, 30, 40, 50, 60, 70, 80], right = False)
            age_outcome          = df_copy.groupby(["AgeGroup", "Outcome"]).size().unstack().fillna(0)

            age_outcome.plot(kind      = 'bar', 
                             stacked   = True, 
                             color     = ["#90CAF9", "#EF9A9A"], 
                             figsize   = (10, 6)
                             )
        
            plt.title("Diabetes Outcome by Age Group", fontsize = 14, weight = 'bold')
            plt.xlabel("Age Group")
            plt.ylabel("Number of Patients")
            plt.legend(title = "Outcome", labels = ["No Diabetes", "Diabetes"])
            plt.tight_layout()
        
            self.plt_saver.save_plot(plot = plt.gcf(), plot_name ="age_group_vs_outcome")
            eda_logger.info("Age group vs outcome stacked bar plot saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_age_group_vs_outcome: {repr(e)}")
        
            raise

    def plot_parallel_coordinates(self) -> None:
        """
        Create a parallel coordinates plot to visualize multivariate patterns.
        """
        
        try:
            

            df_copy      = self.df.copy()
            df_sample    = df_copy.sample(200, random_state = 42)  

            plt.figure(figsize = (12, 6))
            
            parallel_coordinates(df_sample, 
                                 class_column   = "Outcome", 
                                 color          = ("#1f77b4", "#ff7f0e"), 
                                 alpha          = 0.5
                                 )
            
            plt.title("Parallel Coordinates Plot by Outcome", fontsize = 14, weight = 'bold')
            plt.ylabel("Feature Value")
            plt.xticks(rotation = 30)
            plt.tight_layout()
            
            self.plt_saver.save_plot(plot = plt.gcf(), plot_name = "parallel_coordinates")
            eda_logger.info("Parallel coordinates plot saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_parallel_coordinates: {repr(e)}")
        
            raise

    def plot_binned_features_by_outcome(self) -> None:
        """
        Plot countplots of binned glucose and BMI levels split by outcome.
        """
        
        try:
        
            df_copy               = self.df.copy()
            df_copy["BMIBin"]     = pd.cut(df_copy["BMI"], 
                                           bins    = [0, 18.5, 25, 30, 40, 60], 
                                           labels  = ["Underweight", "Normal", "Overweight", "Obese", "Severely Obese"]
                                           )
            
            df_copy["GlucoseBin"] = pd.cut(df_copy["Glucose"], 
                                           bins    = [0, 80, 120, 160, 200], 
                                           labels  = ["Low", "Normal", "High", "Very High"]
                                           )

            fig, axes = plt.subplots(1, 2, figsize = (14, 5))
        
            sns.countplot(data     = df_copy, 
                          x        = "GlucoseBin", 
                          hue      = "Outcome", 
                          palette  = "Set2", 
                          ax       = axes[0]
                          )
        
            axes[0].set_title("Outcome by Glucose Level")
        
            sns.countplot(data     = df_copy, 
                          x        = "BMIBin", 
                          hue      = "Outcome", 
                          palette  = "Set1", 
                          ax       = axes[1]
                          )
        
            axes[1].set_title("Outcome by BMI Level")

            for ax in axes:
                ax.set_xlabel("")
                ax.set_ylabel("Count")

            plt.tight_layout()
            
            self.plt_saver.save_plot(plot = fig, plot_name = "binned_feature_outcome")
            eda_logger.info("Binned feature countplots saved.")
        
        except Exception as e:
            eda_logger.error(f"Error in plot_binned_features_by_outcome: {repr(e)}")
        
            raise


    def run_all_eda(self):
        """
        Function to run all EDA functions in sequence.

        """

        try:

            self.plot_feature_histograms()
            self.plot_feature_boxplots()
            self.plot_target_balance()
            self.plot_correlation_heatmap()
            self.plot_pairplot_by_outcome()
            self.plot_violin_by_outcome()
            self.plot_age_distribution_by_class()
            self.plot_bmi_vs_glucose()
            self.plot_radar_by_outcome()
            self.plot_missing_zero_heatmap()
            self.plot_age_group_vs_outcome()
            self.plot_parallel_coordinates()
            self.plot_binned_features_by_outcome()

            eda_logger.info("All EDA plots generated successfully")

        except Exception as e:
            eda_logger.error(f"Error running all EDA functions: {repr(e)}")
            
            raise
