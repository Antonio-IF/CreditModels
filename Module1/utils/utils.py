"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Module 1 Modelo de puntuacion crediticia                                                   -- #
# -- script: utils.py - Python script with the main functionality                                        -- #
# -- authors: diegotita4 - Antonio-IF - anasofiabrizuela                                                 -- #
# -- license: GNU GENERAL PUBLIC LICENSE - Version 3, 29 June 2007                                       -- #
# -- repository: https://github.com/Antonio-IF/CreditModels                                              -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

# --------------------------------------------------

# LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# ------------------------------

# 
sns.set(style="whitegrid")

# --------------------------------------------------

class DataComparison:
    def __init__(self, datasets, date_column=None, columns_to_analyze=None):
        """
        Initializes the DataComparison class.
        
        Parameters:
        - datasets (dict): A dictionary where keys are dataset names and values are file paths.
        - date_column (str): Optional date column to set as index.
        - columns_to_analyze (list): Columns to include in the analysis.
        """
        self.datasets = {}
        for name, path in datasets.items():
            try:
                data = pd.read_csv(path)
                if date_column and date_column in data.columns:
                    data[date_column] = pd.to_datetime(data[date_column])
                    data.set_index(date_column, inplace=True)
                if columns_to_analyze:
                    data = data[columns_to_analyze]
                self.datasets[name] = data
            except Exception as e:
                print(f"Error loading {name}: {e}")
        
    def summary_statistics(self):
        """
        Prints summary statistics for each dataset.
        """
        for name, data in self.datasets.items():
            print(f"\nSummary Statistics for {name}:")
            print(data.describe())
            print("\nMissing Values:")
            print(data.isnull().sum())
            print("\nOutliers (Z-Score > 3):")
            print(np.where(np.abs(stats.zscore(data.select_dtypes(include=[np.number]))) > 3))

    def plot_histograms(self):
        """
        Plots histograms for numeric columns in each dataset.
        """
        for name, data in self.datasets.items():
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns].hist(bins=30, figsize=(14, 10), edgecolor='black')
            plt.suptitle(f'Histograms of Numeric Columns for {name}', fontsize=16)
            plt.tight_layout()
            plt.show()

    def plot_boxplots(self):
        """
        Plots boxplots for numeric columns in each dataset.
        """
        for name, data in self.datasets.items():
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            data[numeric_columns].plot(kind='box', subplots=True, layout=(2, 3), figsize=(14, 10), patch_artist=True)
            plt.suptitle(f'Boxplots of Numeric Columns for {name}', fontsize=16)
            plt.tight_layout()
            plt.show()

    def plot_correlation_matrices(self):
        """
        Plots correlation matrices for each dataset.
        """
        for name, data in self.datasets.items():
            plt.figure(figsize=(12, 8))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title(f'Correlation Matrix of Numeric Columns for {name}')
            plt.show()

    def compare_correlations(self):
        """
        Compares the correlation matrices between datasets.
        """
        correlations = {name: data.corr() for name, data in self.datasets.items()}
        dataset_names = list(correlations.keys())

        # Compare correlations between all pairs of datasets
        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                corr_diff = correlations[name1] - correlations[name2]
                plt.figure(figsize=(12, 8))
                sns.heatmap(corr_diff, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title(f'Correlation Difference: {name1} vs {name2}')
                plt.show()

    def perform_comparison(self):
        """
        Runs the full comparison analysis.
        """
        self.summary_statistics()
        self.plot_histograms()
        self.plot_boxplots()
        self.plot_correlation_matrices()
        self.compare_correlations()

# -------------------------------------------------- 
class EDA:
    def __init__(self, dataset_path, date_column=None, columns_to_analyze=None):
        self.dataset_path = dataset_path
        self.date_column = date_column
        self.columns_to_analyze = columns_to_analyze
        self.data = pd.read_csv(self.dataset_path)

        if self.date_column and self.date_column in self.data.columns:
            self.data[self.date_column] = pd.to_datetime(self.data[self.date_column])
            self.data.set_index(self.date_column, inplace=True)

        if self.columns_to_analyze:
            self.data = self.data[self.columns_to_analyze]

    # ------------------------------

    def data_summary(self):
        print("First few rows of the data:")
        print(self.data.head(10))

        print("\nInformation:")
        print(self.data.info())

        print("\nSummary statistics:")
        print(self.data.describe())

        print("\nMissing values:")
        print(self.data.isnull().sum())

        print("\nOutliers")
        print(np.where(np.abs(stats.zscore(self.data.select_dtypes(include=[np.number]))) > 3))

    # ------------------------------

    def data_statistics(self):
        numeric_data = self.data.select_dtypes(include=[np.number])

        print("\nMean of each column:")
        print(numeric_data.mean())

        print("\nMedian of each column:")
        print(numeric_data.median())

        print("\nMode of each column:")
        print(numeric_data.mode().iloc[0])

        print("\nVariance of each column:")
        print(numeric_data.var())

        print("\nStandard deviation of each column:")
        print(numeric_data.std())

    # ------------------------------

    def plot_histograms(self):
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns].hist(bins=30, figsize=(14, 10), edgecolor='black')
        plt.suptitle('Histograms of Numeric Columns', fontsize=16)
        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_boxplots(self):
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_columns].plot(kind='box', subplots=True, layout=(2, 3), figsize=(14, 10), patch_artist=True)
        plt.suptitle('Boxplots of Numeric Columns', fontsize=16)
        plt.tight_layout()
        plt.show()

    # ------------------------------

    def plot_correlation_matrix(self):
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Numeric Columns')
        plt.show()

    # ------------------------------

    def perform_eda(self):
        self.data_summary()
        self.data_statistics()
        self.plot_histograms()
        self.plot_boxplots()
        self.plot_correlation_matrix()

# --------------------------------------------------