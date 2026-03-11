import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.custom_features_config2 import ALL_FEATURES


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import PowerTransformer

class CustomFeatureEngineering:
    """
    This is the class for data preprocessing and extracting RFM features at customer-level for model.

    Output:
    rfm_processed_data (csv)
    """

    def __init__(self, file_name):
        self.df_file_name = file_name
        self.df = None
        self.customer_features = None
        self.customer_features_transformed = None
        self.features = None

        self._load_file()
        
    def _load_file(self):
        output_item_path = Path(__file__).resolve().parent.parent/'output'
        output_item_name = [path.name for path in output_item_path.iterdir()]

        for name, path in zip(output_item_name, list(output_item_path.iterdir())):
            if self.df_file_name in name:
                df_file_path = path
                print(path)

        self.df = pd.read_csv(df_file_path, encoding='utf-8-sig', delimiter=',')
        print(f'Successfully loading {self.df_file_name}!')
        print(f'Dataframe {self.df_file_name} size: {self.df.shape}')
        print(self.df.head())

        self.df['TransactionDate'] = pd.to_datetime(self.df['TransactionDate'], format='ISO8601').dt.date
        print('Successfully format date [TransactionDate]')
    
    def _create_customer_features(self):
        """
        Create customer-level aggregated features

        Returns:
            pd.DataFrame: Customer features per customer dataframe
        """
        self.features = ['userID'] + list(ALL_FEATURES.keys())
        print(f'Succesfully loading features: \n {self.features}')
        num_customers = self.df['userID'].nunique()
        self.customer_features = pd.DataFrame(
            data = np.zeros((num_customers,len(self.features)), dtype=float),
            columns=self.features
        )

        self.customer_features['userID'] = self.customer_features[
            'userID'
        ].astype('object')

        print('Extracting features by customer-level...')

        for i, (userid,value) in enumerate(self.df.groupby('userID')):
            """
            Returns single value for each targeted box.
            """
            # === Basic metrics === 
            # userID
            self.customer_features.iat[i,0] = userid

            # Count_TransID
            self.customer_features.iat[i,1] = value['transID'].nunique()

            # Sum_SalesAmount
            self.customer_features.iat[i,2] = value['SalesAmount'].sum()

            # Mean_SalesAmount
            self.customer_features.iat[i,3] = value['SalesAmount'].mean()

            # Pct_TransIDWithVoucherApplied
            self.customer_features.iat[i,4] = value[value['VoucherStatus']=='Yes']['transID'].nunique() / self.customer_features.iat[i,1]

            # Mean_DaysBetweenTransaction
            unique_transaction = value[['transID', 'TransactionDate']].drop_duplicates(subset='transID')
            unique_transaction['TransactionDate'] = pd.to_datetime(unique_transaction['TransactionDate'])
            unique_transaction = unique_transaction.sort_values(by='TransactionDate')
            mean_diff_days = unique_transaction['TransactionDate'].diff().dt.total_seconds().mean() / 86400
            self.customer_features.iat[i, 5] = 0 if pd.isna(mean_diff_days) else mean_diff_days
            
            # === Other metrics === 
            # === MERCHANT == 
            # Count_DistinctMerchant
            self.customer_features.iat[i,6] = value['merchantID'].nunique()

            # === CHANNEL ===
            # Pct_TransIDByDelivery
            self.customer_features.iat[i,7] = value[value['Channel'] == 'Delivery']['transID'].nunique() / self.customer_features.iat[i,1]

            # Pct_TransIDByApp
            self.customer_features.iat[i,8] = value[value['OrderFrom'] == 'APP']['transID'].nunique() / self.customer_features.iat[i,1]


            if (i+1) % 10000 == 0:
                print(f'Processing {i+1}/{num_customers} users...')

        print('Successfully feature extracting!')
        
        # Handling outliers
        upper_limit = self.customer_features['Sum_SalesAmount'].quantile(0.99)
        self.customer_features['Sum_SalesAmount'] = self.customer_features['Sum_SalesAmount'].clip(upper=upper_limit)
        print('Successfully processing outliers!')

        print(self.customer_features.head())
        return self.customer_features

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
            title = 'Feature Histograms after Yeo' \
            '' \
            'johnson Transformation'

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
    features = CustomFeatureEngineering(file_name='fact_merge.csv')
    try:
        customer_features = features._create_customer_features()
        customer_features_transformed = features._power_transformation_scaled()
        features._export_feature_files(output_folder=f"{Path(__file__).resolve().parent.parent/'output'}",
                                       customer_feature_name='customer_features_custom2',
                                       customer_feature_transform_name='customer_feature_transformed_custom2')
        features._plot_feature_histogram(transformed=True)
    except Exception as e:
        print(f'{e}')