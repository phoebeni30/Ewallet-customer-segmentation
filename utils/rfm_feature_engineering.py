import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.rfm_config import ALL_FEATURES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PowerTransformer

class RFMFeatureEngineering:
    """
    This is the class for data preprocessing and extracting RFM features at customer-level for model.

    Output:
    rfm_processed_data (csv)
    """

    def __init__(self, file_name):
        self.df_rfm = None
        self.customer_features = None
        self.customer_features_transformed = None
        self.df_file_name = file_name

        self._load_file()
        
    def _load_file(self):
        output_item_path = Path(__file__).resolve().parent.parent/'output'
        output_item_name = [path.name for path in output_item_path.iterdir()]

        for name, path in zip(output_item_name, list(output_item_path.iterdir())):
            if self.df_file_name in name:
                df_file_path = path
                print(path)

        self.df_rfm = pd.read_csv(df_file_path, encoding='utf-8', delimiter=';')


        print('Successfully loading df_rfm!')
        print(f'Dataframe df_rfm size: {self.df_rfm.shape}')

    
    def _create_customer_features(self):
        features = ['userID'] + list(ALL_FEATURES.keys())
        self.customer_features = self.df_rfm[features]
        print(f'Dataframe customer_features size: {self.customer_features.shape}')
        print(self.customer_features.head())

    def _power_transformation_scaled(self):
        if self.customer_features is None:
            print('Fail to detect feature file. Extract features first! ')
        
        else:
            features_withoutID = self.customer_features.set_index('userID')
        
            print(f'Processing yeojohnson transformation...')
            transform_model = PowerTransformer(method='yeo-johnson', standardize=True)
            self.customer_features_transformed = transform_model.fit_transform(features_withoutID)
            self.customer_features_transformed = pd.DataFrame(self.customer_features_transformed,columns=features_withoutID.columns)
            self.customer_features_transformed = pd.concat([self.customer_features['userID'],self.customer_features_transformed],axis=1)

            print('Successfully executing yeojohnson transformation and scale data!')
            print(f'Dataframe customer_features_transformed size: {self.customer_features_transformed.shape}')
            print(self.customer_features_transformed.head())
            
            return self.customer_features_transformed

    
    def _plot_feature_histogram(self, transformed=False):

        # Check transformed or transformed_data existed
        if transformed and self.customer_features_transformed is not None:
            data = self.customer_features_transformed
            title = 'Feature Histograms after Yeojohnson Transformation'

        else:
            # Check feature data existed
            if self.customer_features is not None:
                data = self.customer_features
                title = 'Feature Histograms before Yeojohnson Transformation'
            
            else: 
                print(f'Fail to display features. Extract features first!')
            
        # Plot feature histograms
        if 'userID' in data.columns:
            data =  data.drop(columns='userID')

        num_features = len(list(data.columns))
        print(f'Number of feature to plot: {num_features}')

        if num_features%4 == 0:
            ncols = 4
            nrows = num_features % ncols
            print(f'ncols, nrows = {ncols},{nrows}')

        else:
            ncols = 3
            nrows = num_features // ncols + 1
            print(f'ncols, nrows = {ncols},{nrows}')


        fig, axes = plt.subplots(nrows=nrows, ncols=ncols,figsize=(15, 4 * nrows))
        axes_flatten = axes.flatten()

        with sns.plotting_context(context='notebook'):

            for i, feature in enumerate(list(data.columns)):        
                sns.histplot(
                        x = data[feature],
                        ax=axes_flatten[i],
                        bins=30,
                        kde=True
                )
        
        # Remove blank subplots
        for i in range(num_features, len(axes_flatten)):
            fig.delaxes(axes_flatten[i])

        # Set title for each selected data
        fig.suptitle(title, fontsize = 16, y=1.02)

        # Show charts
        plt.tight_layout()
        plt.show()


    def _export_feature_files(self, output_folder,customer_feature_name, customer_feature_transform_name):
        if self.customer_features is None:
            print('Fail to detect feature file. Extract features first! ')
        
        elif self.customer_features_transformed is None:
            print('Fail to detect feature file. Extract features first! ')

        else:
            # Export raw_features_data
            self.customer_features.to_csv(
                f'{output_folder}/{customer_feature_name}.csv',
                sep=',',
                encoding='utf-8-sig',
                index=False
            )
            print('Successfully exported Raw Features Data!')

            # Export transformed_features_data
            self.customer_features_transformed.to_csv(
                f'{output_folder}/{customer_feature_transform_name}.csv',
                sep=',',
                encoding='utf-8-sig',
                index=False
            )
            print('Successfully exported Transformed Features Data!')     


if __name__ == "__main__":
    rfm_features = RFMFeatureEngineering('rfm.csv')
    try:
        customer_features = rfm_features._create_customer_features()
        customer_features_transformed = rfm_features._power_transformation_scaled()
        print(customer_features_transformed.head())

        rfm_features._export_feature_files(output_folder=f"{Path(__file__).resolve().parent.parent/'output'}", customer_feature_name='customer_features',customer_feature_transform_name='customer_feature_transformed')
        rfm_features._plot_feature_histogram(transformed=True)
    except Exception as e:
        print(f'{e}')
# %%
