#%%
import sys
import os


from utils.cluster_model import ClusterAnalysis
from utils.rfm_config import ALL_FEATURES
import pandas as pd
import numpy as np


# Load file
raw_features_path= os.path.join(project_root, './output/customer_features.csv')
transformed_features_path= os.path.join(project_root, './output/customer_feature_transformed.csv')

cluster_model = ClusterAnalysis(raw_features_path, transformed_features_path)
cluster_model._load_features(ALL_FEATURES=ALL_FEATURES)
df, df_origin = cluster_model._load_file(raw_features_path, transformed_features_path)

print(df)
print(df_origin)

# Opimal cluster
optimal_clusters = cluster_model._find_optimal_cluster(use_minibatch=True)
cluster_model._plot_optimal_cluster()


# PCA Analysis
pca = cluster_model._pca_analysis()
cluster_model._plot_pca_analysis()


# Apply Kmeans & Visualization 2D-3D-Cohort_Table
cluster_results = cluster_model._apply_kmeans(k_values=[2,3,4])
cluster_model._plot_cluster_2d(k_values=[2,3,4])
cluster_model._plot_cluster_3d(k_values=[2,3,4])

# %%
cluster_model._display_cohort_table(k_values = [2,3,4])

# %%
df = df.reset_index()

from utils.rfm_manual import RFMManual
rfm_manual = RFMManual()
df_rfm = rfm_manual._get_concate_df()
#%%
df_merge = pd.merge(df, df_rfm, on='userID',how='left')
df_merge
# %%
df_rfm
# %%
