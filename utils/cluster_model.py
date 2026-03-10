import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from IPython.display import display

class ClusterAnalysis:
    """
    Docstring for RFMClusterAnalysis: This is a class for handling cluster model, 
    visualizing clusters and explanation.
    """

    def __init__(self,raw_features_path=None, transformed_features_path=None):
        self.raw_features_path = raw_features_path
        self.transformed_features_path = transformed_features_path
        self.raw_customer_features = None
        self.transformed_customer_features = None
        
        self.optimal_clusters = {}
        self.cluster_results = {}
        
        self.pca = None
        self.df = None
        self.df_origin = None
        self.df_pca = None
        self.all_features = None
    
    def _load_features(self, ALL_FEATURES):
        self.all_features = list(ALL_FEATURES.keys())

    def _load_file(self, raw_features_path=None, transformed_features_path=None):
        self.raw_customer_features = pd.read_csv(raw_features_path,sep=',',encoding='utf-8-sig')
        self.transformed_customer_features = pd.read_csv(transformed_features_path,sep=',',encoding='utf-8-sig')

        self.df_origin = self.raw_customer_features.set_index('userID')
        self.df = self.transformed_customer_features.set_index('userID')
        
        self.df_origin = self.df_origin[self.all_features]
        self.df = self.df[self.all_features]

        print(f'Successfully loading file: raw_customer_features: {self.df_origin.shape}')
        print(f'Successfully loading file: transformed_customer_features: {self.df.shape}')

        return self.df, self.df_origin
    

    def _find_optimal_cluster(self, use_minibatch=True):
        print(f'Transformed feature data shape: {self.df.shape}')

        inertias = []
        silhouette_scores = []
        k_range = range(2, 11)

        sample_size = min(20000, self.df.shape[0]) 

        for k in k_range:
            if use_minibatch and self.df.shape[0] > 100000:
                kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=3, batch_size=2048)
            else:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                
            labels = kmeans.fit_predict(self.df)
            inertias.append(kmeans.inertia_)

            # Pass the sample_size parameter here (THIS IS THE KEY OPTIMIZATION STEP)
            score = silhouette_score(self.df, labels, sample_size=sample_size, random_state=42)
            silhouette_scores.append(score)
            
            print(f"Completed K={k} | Inertia: {kmeans.inertia_:.0f} | Silhouette Score: {score:.4f}")

        self.optimal_clusters = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'best_k_silhouette': list(k_range)[np.argmax(silhouette_scores)]
        }

        print(f"The optimal number of clusters based on Silhouette Score is: {self.optimal_clusters['best_k_silhouette']}")
        return self.optimal_clusters

    def _plot_optimal_cluster(self):
        nrows, ncols = 1,2
        method_name = ['inertias','silhouette_scores']

        fig,axes = plt.subplots(nrows, ncols, figsize=(12,6))

        for i, method in enumerate(method_name):
            # Elbow Inertia
            axes[i].plot(
                self.optimal_clusters['k_range'],
                self.optimal_clusters[method],
                markersize=5,
                marker='o',
                color='purple',
                linestyle='-'
            )
            axes[i].set_title(method)
            axes[i].set_xlabel('k_range')
            axes[i].set_ylabel(method)
            axes[i].grid(True)

        plt.tight_layout()
        plt.show()

        print(f'Optimize k by Silhouette Score: k = {list(self.optimal_clusters['k_range'])[np.argmax(self.optimal_clusters['silhouette_scores'])]}') 

    def _pca_analysis(self):
        print(f'Transformed feature data shape: {self.df.shape}')

        # PCA Model
        self.pca = PCA(n_components=(len(self.df.columns)-1))
        pca_features = self.pca.fit_transform(self.df)

        pca_columns = [f'PC{i+1}' for i in range(pca_features.shape[1])]
        
        self.df_pca = pd.DataFrame(pca_features,
                            columns=pca_columns,
                            index=self.df.index)

        return self.df_pca

    def _plot_pca_analysis(self):
        # Plot PCA
        plt.figure(figsize=(12,6))
        plt.bar(
            range(1, len(self.pca.explained_variance_ratio_)+1),
            self.pca.explained_variance_ratio_,
            label = 'Single Variance'
        )

        plt.step(
            range(1, len(self.pca.explained_variance_ratio_)+1),
            np.cumsum(self.pca.explained_variance_ratio_),
            label = 'Cumulative variance',
            color = 'red',
            linewidth = 2,
            where='mid'
        )

        plt.axhline(y=0.8, color='green',linestyle='--',label='80% variance')
        plt.axhline(y=0.9, color='orange',linestyle='--',label='90% variance')

        plt.xlabel('Principle Components')
        plt.ylabel('Ratio of explained variation ratio')
        plt.title('PCA Analysis - Explained Variance')

        plt.legend()
        plt.tight_layout()
        plt.show()

        for i in range(min(5,len(self.pca.explained_variance_ratio_))):
            cumsum = np.sum(self.pca.explained_variance_ratio_[:i+1])
            print(f'PC1-PC{i+1}: {cumsum:.2%}')        


    def _apply_kmeans(self, k_values = [2,3,4]):
        # Apply kmeans
        self.cluster_results = {}
        k_values = k_values

        for k in k_values:
            kmeans = KMeans(n_clusters=k,random_state=42,n_init=10)
            clusters = kmeans.fit_predict(self.df)

            # Add cluster to df and df_pca
            cluster_col = f'clusters_by_k_{k}'
            self.df[cluster_col] = clusters
            self.df_origin[cluster_col] = clusters
            self.df_pca[cluster_col] = clusters

            # Dict for plot
            self.cluster_results[k] = {
                'cluster':clusters,
                'size_of_each_cluster':pd.Series(clusters).value_counts().sort_index(),
                'means':self.df_origin.groupby(cluster_col).mean()
            }

            print(f'Cluster sizes (k={k})')
            print(self.cluster_results[k]["size_of_each_cluster"])
        
        return self.cluster_results


    def _plot_cluster_2d(self, k_values = [2,3,4]):

        fig, axes = plt.subplots(1,len(k_values),figsize=(18,7))

        for i in range(len(k_values)):
            scatter = axes[i].scatter(
                self.df_pca['PC1'],
                self.df_pca['PC2'],
                c = self.df_pca[f'clusters_by_k_{k_values[i]}'],
                cmap='viridis',
                s=20
            )

            axes[i].set_title(f'Kmeans Cluster k={k_values[i]}')
            axes[i].set_xlabel('PC1')
            axes[i].set_ylabel('PC2')
        
            plt.colorbar(
                scatter,
                ax=axes[i],
                label='Cluster'
            )
            
        plt.tight_layout()
        plt.show()        


    def _plot_cluster_3d(self, k_values = [2,3,4]):
        fig = plt.figure(figsize=(18,7))

        for i, k in enumerate(k_values):
            ax = fig.add_subplot(1,len(k_values),i+1,projection='3d')

            if len(self.all_features) <=3:
                scatter = ax.scatter(
                    self.df_pca['PC1'],
                    self.df_pca['PC2'],
                    c = self.df_pca[f'clusters_by_k_{k}'],
                    cmap='viridis',
                    s=50
                )

                ax.set_title(f'Kmeans Cluster k = {k}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')

                ax.view_init(elev=20, azim=135)

                plt.colorbar(
                    scatter,
                    ax=ax
                )
            
            else:
                scatter = ax.scatter(
                    self.df_pca['PC1'],
                    self.df_pca['PC2'],
                    self.df_pca['PC3'],
                    c = self.df_pca[f'clusters_by_k_{k}'],
                    cmap='viridis',
                    s=50
                )

                ax.set_title(f'Kmeans Cluster k = {k}')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')

                ax.view_init(elev=20, azim=135)

                plt.colorbar(
                    scatter,
                    ax=ax
                )
            
        plt.tight_layout()
        plt.show()        

    def _feature_loading(self):
        loadings = pd.DataFrame(
            self.pca.components_.T, 
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)], 
            index=self.all_features  
        )

        loadings_top5 = loadings[['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]

        list_pcs = [f'PC{i+1}' for i in range(5)]
        cluster_pca_profile = self.df_pca.groupby('clusters_by_k_4')[list_pcs].mean()
        print(cluster_pca_profile)

        plt.figure(figsize=(12, 8))
        sns.heatmap(loadings_top5, 
                    annot=True,      
                    cmap='coolwarm', 
                    fmt=".2f", 
                    center=0,      
                    linewidths=0.5)

        plt.title('Feature Loadings' , fontsize=15)
        plt.xlabel('Principal Components', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.show()

    def _display_cohort_table(self, k_values = [2,3,4]):
        for k in k_values:
            print(f'\n=== CLUSTER ANALYSIS K={k} ===')
            cluster_means = self.cluster_results[k]['means']
            cluster_sizes = self.cluster_results[k]['size_of_each_cluster']

            for i, values in enumerate(cluster_sizes):
                print(f'Cluster {i}: {values} users ({(values/cluster_sizes.sum()):.2%})')

            display(cluster_means.round(2).style.background_gradient(cmap='viridis',axis=0))        




