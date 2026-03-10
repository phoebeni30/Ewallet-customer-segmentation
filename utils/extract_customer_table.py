# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from pathlib import Path
from functools import reduce


class CustomerFeaturePipeline:
    """
    A class for extracting customer table as demand.
    Outputs: customer_table.csv
            'userID',
            'FirstAppID',
            'FirstAppIDDate',
            'SecondAppID',
            'SecondAppIDDate',
            'LastestStore',
            'LastestStoreDate',
            'NumberUniqueMerchant',
            'NumberUniqueChannel',
            'SalesInLast30Days',
            'MerchantWithHighestAppliedVoucher',
            'ProvinceWithHighestSales'
    """
    def __init__(self):
        self.df_fact = None
        self.df_dim_merchant = None
        self.df_dim_store = None
        self.max_date = None
        self.list_df_apps = {}
        self.df_merge = None
        self.df_final = None
        self.num_app_requested = 0

        self._load_dfs()
        self._get_df_merge()
    
    def _load_dfs(self):
        current_dir = Path(__file__).resolve().parent
        data_path = current_dir.parent / "data"
        data_item_dirs = list(data_path.iterdir())
        print(f'Data dirs: {data_item_dirs}')

        self.df_dim_merchant = pd.read_csv(data_item_dirs[0], delimiter=';')
        self.df_dim_store = pd.read_csv(data_item_dirs[1], delimiter=';')
        self.df_fact = pd.read_csv(data_item_dirs[2], delimiter=',')
        print('Successfully read file!')
        
        self.df_fact['TransactionDate'] = pd.to_datetime(self.df_fact['TransactionDate'], format='ISO8601').dt.date
        print('Successfully format date [TransactionDate]')

        self.max_date = self.df_fact['TransactionDate'].max()


    def _get_df_merge(self):
        df_mer_renamed = self.df_dim_merchant.rename(columns={'appid':'appID'})

        self.df_merge = pd.merge(self.df_fact, df_mer_renamed, on='appID', how='left')
        self.df_merge = pd.merge(self.df_merge, self.df_dim_store, on='storeID', how='left')
        print(f'Dataframe size: {self.df_merge.shape}')

        if self.df_merge.shape[0] > self.df_fact.shape[0]:
            print('Warning: Dataframe expanded! Check for duplicates.')

        return self.df_merge

    def _get_earliest_app(self, num_app_requested):
        self.num_app_requested = int(num_app_requested)
        df = self.df_fact.groupby(['userID', 'appID'])['TransactionDate'].min().reset_index()
        df = df.sort_values(by=['userID', 'TransactionDate'], ascending=[True, True])
        df['row_number'] = df.groupby('userID').cumcount() + 1
        
        for i in range(self.num_app_requested):
            dfi = df[df['row_number'] == (i+1)][['userID', 'appID', 'TransactionDate']]
            dfi.columns = ['userID', f'App{i+1}_ID', f'App{i+1}_Date']

            self.list_df_apps[f'df{i+1}'] = dfi
        
        print(f'There are {len(self.list_df_apps)} in total.')
            
        return self.list_df_apps

    def _get_latest_store(self):
        df = self.df_fact.groupby(['userID', 'storeID'])['TransactionDate'].max().reset_index()
        df = df.sort_values(by=['userID', 'TransactionDate'], ascending=[True, False])
        df['row_number'] = df.groupby('userID').cumcount() + 1
        df_lastest_store = df[df['row_number'] == 1][['userID', 'storeID', 'TransactionDate']]
        df_lastest_store.columns = ['userID','LastestStore', 'LastestStoreDate']

        return df_lastest_store

    def _get_unique_merchants(self):
        
        return self.df_merge.groupby('userID')['merchantID'].nunique().reset_index()

    def _get_unique_channels(self):
        
        return self.df_fact.groupby('userID')['Channel'].nunique().reset_index()

    def _get_sales_last_30_days(self):
        is_within_30_days = pd.to_timedelta(self.max_date - self.df_fact['TransactionDate']) <= pd.Timedelta(days=30)
        df_30 = self.df_fact[is_within_30_days]
        return df_30.groupby('userID')['SalesAmount'].sum().reset_index()

    def _get_highest_voucher_merchant(self, voucher_status='Yes'):
        df_voucher = self.df_merge[self.df_fact['VoucherStatus'] == voucher_status]
        df_grouped = df_voucher.groupby(['userID', 'merchantName'])['transID'].count().reset_index()
        df_grouped = df_grouped.sort_values(by=['userID', 'transID'], ascending=[True, False])
        df_grouped['row_number'] = df_grouped.groupby('userID').cumcount() + 1
        
        return df_grouped[df_grouped['row_number'] == 1][['userID', 'merchantName']]

    def _get_highest_sales_province_last_45_days(self):
        is_within_45_days = pd.to_timedelta(self.max_date - self.df_fact['TransactionDate']) <= pd.Timedelta(days=45)
        df_45 = self.df_merge[is_within_45_days]
        
        df_grouped = df_45.groupby(['userID', 'Province'])['SalesAmount'].sum().reset_index()
        df_grouped = df_grouped.sort_values(by=['userID', 'SalesAmount'], ascending=[True, False])
        df_grouped['row_number'] = df_grouped.groupby('userID').cumcount() + 1
        
        return df_grouped[df_grouped['row_number'] == 1][['userID', 'Province']]

    def run(self, num_app_requested):
        """Pipeline processing"""
        print("Feature processing...")
        
        # Call function
        self.list_df_apps = self._get_earliest_app(num_app_requested)
        df_latest_store = self._get_latest_store()
        df_unique_merchant = self._get_unique_merchants()
        df_unique_channel = self._get_unique_channels()
        df_sales_30 = self._get_sales_last_30_days()
        df_voucher_merchant = self._get_highest_voucher_merchant()
        df_province_45 = self._get_highest_sales_province_last_45_days()

        # Merge: Dataframe + Names
        dfs = list(self.list_df_apps.values())
        dfs.extend([
            df_latest_store,
            df_unique_merchant,
            df_unique_channel,
            df_sales_30,
            df_voucher_merchant,
            df_province_45
        ])

        print("Processing file merging...")
        dynamic_columns = ['userID']
        for i in range(self.num_app_requested):
            dynamic_columns.extend([f'App{i+1}_ID', f'App{i+1}_Date'])    
        
        dynamic_columns.extend(
        [
            'LastestStore', 'LastestStoreDate',
            'NumberUniqueMerchant',
            'NumberUniqueChannel',
            'SalesInLast30Days',
            'MerchantWithHighestAppliedVoucher',
            'ProvinceWithHighestSales'
        ]
        )
        
        # Merge
        self.df_final = reduce(lambda left, right: pd.merge(left, right, on='userID', how='outer'), dfs)
        self.df_final.columns = dynamic_columns
        # Làm sạch data: Fill 0 cho những khách hàng không có giao dịch trong 30 ngày
        self.df_final['SalesInLast30Days'] = self.df_final['SalesInLast30Days'].fillna(0)
        
        print(f"Done! Output dataframe has {len(self.df_final)} rows.")
        print(self.df_final.head(5))

        return self.df_final
    
    def export_merge_csv(self, output_folder):
        file_path = Path(f'{output_folder}/fact_merge.csv')

        if not file_path.exists():
            self.df_merge.to_csv(f'{output_folder}/fact_merge.csv',sep=',', encoding='utf-8-sig', index=False)
            print(f'File fact_merge.csv successfully exported!')
        else:
            print('File already existed!')

    def export_final_csv(self, output_folder):
        file_path = Path(f'{output_folder}/customer_table_final.csv')
        
        if not file_path.exists():
            self.df_final.to_csv(f'{output_folder}/customer_table_final.csv',sep=',',encoding='utf-8-sig', index=False)
            print(f'File customer_table_final.csv successfully exported!')
        else:
            print('File already existed!')

if __name__ == "__main__":
    pipeline = CustomerFeaturePipeline()
    try:
        pipeline.run(num_app_requested=3)
        pipeline.export_merge_csv(output_folder=f"{Path(__file__).resolve().parent.parent/'output'}")
        pipeline.export_final_csv(output_folder=f"{Path(__file__).resolve().parent.parent/'output'}")

    except Exception as e:
        print(f'{e}')

