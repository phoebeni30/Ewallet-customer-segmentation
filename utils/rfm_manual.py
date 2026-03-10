import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class RFMManual:
    def __init__(self, file_name):
        self.df_merge_file_name = file_name
        self.df_merge = None
        self.df_rfm = None
        self.df_rfm_segments = None
        self.df_segment_counts = None

        self._load_file()
        self._get_rfm_segments()


    def _load_file(self):
        output_item_path = Path(__file__).resolve().parent.parent/'output'
        output_item_name = [path.name for path in output_item_path.iterdir()]

        for name, path in zip(output_item_name, list(output_item_path.iterdir())):
            if self.df_merge_file_name in name:
                df_merge_file_path = path
                print(path)

        self.df_merge = pd.read_csv(df_merge_file_path, encoding='utf-8-sig', delimiter=',')
        print('Successfully loading df_merge!')
        print(f'Dataframe size: {self.df_merge.shape}')

        self.df_merge['TransactionDate'] = pd.to_datetime(self.df_merge['TransactionDate'], format='ISO8601').dt.date
        print('Successfully format date [TransactionDate]')

    def _get_recency(self):
        # Recency
        reference_date = self.df_merge['TransactionDate'].max() + pd.Timedelta(days=1)
        df_recency = self.df_merge.groupby('userID')['TransactionDate'].max().to_frame(name='TransactionDate')
        df_recency['Recency'] = reference_date - df_recency['TransactionDate']
        df_recency['Recency'] = pd.to_timedelta(df_recency['Recency']).dt.days

        print(df_recency['Recency'].head())

        return df_recency['Recency']
    
    def _get_frequency(self):
        # Frequency
        df_frequency = self.df_merge.groupby('userID')['transID'].count().to_frame(name='Frequency')
        print(df_frequency.head())

        return df_frequency
    
    def _get_monetary(self):
        # Monetary
        df_monetary = self.df_merge.groupby('userID')['SalesAmount'].sum().to_frame(name='Monetary')
        print(df_monetary.head())
        
        return df_monetary

    def _get_rfm_segments(self):
        rfm_segments = {
        'Champions': ['555', '554', '544', '545', '454', '455', '445'],
        'Loyal Customers': ['543', '444', '435', '355', '354', '345', '344', '335'],
        'Potential Loyalists': ['553', '551', '552', '541', '542', '533', '532', '531', '452', '451', '442', '441', '431', '453', '433', '432', '423', '353', '352', '351', '342', '341', '333', '323'],
        'Recent Customers': ['512', '511', '422', '421', '412', '411', '311'],
        'Promising': ['525', '524', '523', '522', '521', '515', '514', '513', '425', '424', '413', '414', '415', '315', '314', '313'],
        'Customers Needing Attention': ['535', '534', '443', '434', '343', '334', '325', '324'],
        'About to Sleep': ['331', '321', '312', '221', '213'],
        'At Risk': ['255', '254', '245', '244', '253', '252', '243', '242', '235', '234', '225', '224', '153', '152', '145', '143', '142', '135', '134', '133', '125', '124'],
        'Can’t Lose Them': ['155', '154', '144', '214', '215', '115', '114', '113'],
        'Hibernating': ['332', '322', '231', '241', '251', '233', '232', '223', '222', '132', '123', '122', '212', '211'],
        'Lost': ['111', '112', '121', '131', '141', '151']
}
        self.df_rfm_segments = pd.DataFrame(list(rfm_segments.items()), columns=['Segment','RFM_Scores'])
        self.df_rfm_segments = self.df_rfm_segments.explode('RFM_Scores')

    def _get_concate_df(self):
        recency = self._get_recency()
        frequency = self._get_frequency()
        monetary = self._get_monetary()

        self.df_rfm = pd.concat([recency, frequency, monetary], axis=1).reset_index()
        print(f'Dataframe df_rfm shape before scoring: {self.df_rfm.shape}')
        print(self.df_rfm.head())

        print('START SCORING R-F-M...')
        print('='*20)

        # 1. Recency ranking (0-5)
        # - Define bins: 0-7 days, 14-30 days, 30-60 days, 60-120 days, >120 days
        rbins  = [0, 7, 30, 60, 120, np.inf]
        rlabels = [i for i in range(5,0,-1)]
        self.df_rfm['rRecency'] = pd.cut(self.df_rfm['Recency'], labels=rlabels, bins=rbins)
        rcounts =  self.df_rfm['rRecency'].value_counts().sort_index()
        print(f'Recency ranking counts: {rcounts}')
        print(rbins)

        # 2. Frequency ranking (0-5)
        # - Define bins:
        # 1 order: New / Occasional Customers
        # 2 orders: Returning Customers
        # 3-4 orders: Regular / Loyal Customers
        # 5-10 orders: High-Loyalty Customers
        # Over 10 orders (to infinity): VIP Customers
        fbins  = [0, 1, 2, 4, 10, np.inf]
        flabels = [i for i in range(1,6,1)]
        self.df_rfm['rFrequency'] = pd.cut(self.df_rfm['Frequency'], labels=flabels, bins=fbins)
        fcounts =  self.df_rfm['rFrequency'].value_counts().sort_index()
        print(f'Frequency ranking counts: {fcounts}')

        # 3. Monetary ranking (0-5)
        avg_salesamount_by_customer = self.df_merge.groupby('userID')['SalesAmount'].mean().reset_index()
        mbins = [0] + avg_salesamount_by_customer['SalesAmount'].quantile([0.2, 0.4, 0.6, 0.8]).values.tolist() + [np.inf]
        mlabels = [i for i in range(1,6,1)]
        self.df_rfm['rMonetary'] = pd.cut(self.df_rfm['Monetary'], labels=mlabels, bins=mbins)

        mcounts =  self.df_rfm['rMonetary'].value_counts().sort_index()
        print(f'Monetary ranking counts: {mcounts}')

        # Concate and format RFM Scores
        self.df_rfm['rfm_score_groups'] = self.df_rfm['rRecency'].astype('str') + self.df_rfm['rFrequency'].astype('str') + self.df_rfm['rMonetary'].astype('str')
        print(f'Dataframe df_rfm shape after scoring: {self.df_rfm.shape}')
        
        print('START CONNECTING RFM SEGMENTS...')
        print('='*20)
        self.df_rfm = pd.merge(self.df_rfm, self.df_rfm_segments, how='left',left_on='rfm_score_groups', right_on='RFM_Scores')
        print(f'Dataframe df_rfm shape after connecting RFM segment table: {self.df_rfm.shape}')

        return self.df_rfm

    def _get_segment_breakdown(self):
        self.df_segment_counts = self.df_rfm['Segment'].value_counts().reset_index()
        print(self.df_segment_counts.head())

        return self.df_segment_counts
    
    def _export_df_rfm(self, output_folder, filename):
        file_path = Path(f'{output_folder}/{filename}.csv')
        
        if not file_path.exists():
            self.df_rfm.to_csv(f'{output_folder}/{filename}.csv',sep=';',encoding='utf-8-sig')
            print(f'File rfm.csv successfully exported!')
        else:
            print('File already existed!')

    def _export_df_segment_breakdown(self, output_folder, filename):
        file_path = Path(f'{output_folder}/{filename}.csv')
        
        if not file_path.exists():
            self.df_segment_counts.to_csv(f'{output_folder}/{filename}.csv',sep=';',encoding='utf-8-sig')
            print(f'File segment_breakdown.csv successfully exported!')
        else:
            print('File already existed!')



if __name__ == "__main__":
    rfm_manual = RFMManual('fact_merge')
    try:
        rfm_manual._get_concate_df()
        rfm_manual._get_segment_breakdown()
        rfm_manual._export_df_rfm(output_folder=f"{Path(__file__).resolve().parent.parent/'output'}",filename='rfm')
        rfm_manual._export_df_segment_breakdown(output_folder=f"{Path(__file__).resolve().parent.parent/'output'}",filename='segment_breakdown')

    except Exception as e:
        print(f'{e}')