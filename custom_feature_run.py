#%%
import sys
import os 
project_root = r"D:\03 Data science\00 AIO2025\01 Projects\Case9_ZDS_interview_test"
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from utils.cluster_model import ClusterAnalysis
from utils.custom_features_config import ALL_FEATURES
import pandas as pd


# Load file
raw_features_path= os.path.join(project_root, 'output','customer_features_custom.csv')
transformed_features_path= os.path.join(project_root,'output','customer_feature_transformed_custom.csv')

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




#%%
# Apply Kmeans & Visualization 2D-3D-Cohort_Table
cluster_results = cluster_model._apply_kmeans(k_values=[2,3,4])
cluster_model._feature_loading()
cluster_model._plot_cluster_2d(k_values=[2,3,4])
cluster_model._plot_cluster_3d(k_values=[2,3,4])




# %%
cluster_model._display_cohort_table(k_values = [2,3,4])

# %%
df = df.drop(columns=['level_0','index'])

#%%
df_origin = df_origin.reset_index()
#%%
df_fact = pd.read_csv(os.path.join(project_root,'output','fact_merge.csv'),delimiter=',',encoding='utf-8-sig')
df_merge = pd.merge(df_origin, df_fact, on='userID',how='left')
cluster1 = df_merge[df_merge['clusters_by_k_4'].isin([0,1])]

print(cluster1['Channel'].value_counts())
print(cluster1['Mean_TransIDCountPerChannel'].value_counts())
#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Giả sử df_fact_merge là biến chứa DataFrame của bạn
# Lấy danh sách các cột cần kiểm tra từ ALL_FEATURES
features_to_check = [
    'Count_TransID', 'Sum_SalesAmount', 'Mean_SalesAmount', 
    'Pct_TransIDWithVoucherApplied', 'Mean_DaysBetweenTransaction',
    'Mean_TransIDCountPerMerchant', 'Mean_SalesAmountSumPerMerchant',
    'Mean_TransIDCountPerChannel', 'Mean_SalesAmountSumPerChannel',
    'Mean_TransIDCountPerSource', 'Mean_SalesAmountSumPerSource'
]

# Tạo một DataFrame phụ chỉ chứa các tính năng cần phân tích
df_subset = df_origin[features_to_check]

# Tính toán ma trận tương quan (mặc định là Pearson)
corr_matrix = df_subset.corr()

# Thiết lập kích thước khung hình
plt.figure(figsize=(14, 10))

# Vẽ Heatmap
sns.heatmap(corr_matrix, 
            annot=True,         # Hiển thị giá trị số trên từng ô
            fmt=".2f",          # Làm tròn 2 chữ số thập phân
            cmap="coolwarm",    # Dùng dải màu từ lạnh (xanh) sang nóng (đỏ)
            vmin=-1, vmax=1,    # Cố định thang đo từ -1 đến 1
            linewidths=0.5,     # Tạo đường viền mảnh giữa các ô cho dễ nhìn
            square=True)        # Ép các ô thành hình vuông

# Căn chỉnh tiêu đề và nhãn
plt.title("Biểu đồ tương quan (Correlation Heatmap) - fact_merge", fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Hiển thị biểu đồ
plt.show()
# %%
