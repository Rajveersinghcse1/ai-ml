"""
Ultra-Advanced Clustering Module
Comprehensive customer segmentation using state-of-the-art clustering algorithms
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import pickle

# Core ML libraries
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import (
    KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering,
    OPTICS, MeanShift, Birch, AffinityPropagation
)
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score
)
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, TruncatedSVD
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

# Scipy
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy import stats

# Advanced clustering (with fallbacks)
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def prepare_features_for_clustering(rfm_df, features=['Recency', 'Frequency', 'Monetary']):
    """
    Prepare and scale features for clustering
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe
    features : list
        List of feature names to use for clustering
    
    Returns:
    --------
    tuple
        (scaled_features, scaler)
    """
    # Extract features
    X = rfm_df[features].values
    
    # Handle any potential infinite or NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def find_optimal_clusters(X, max_clusters=10, method='both'):
    """
    Find optimal number of clusters using Elbow method and Silhouette score
    
    Parameters:
    -----------
    X : np.array
        Scaled feature matrix
    max_clusters : int
        Maximum number of clusters to test
    method : str
        'elbow', 'silhouette', or 'both'
    
    Returns:
    --------
    dict
        Dictionary with inertias and silhouette scores
    """
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        
        if method in ['elbow', 'both']:
            inertias.append(kmeans.inertia_)
        
        if method in ['silhouette', 'both']:
            score = silhouette_score(X, kmeans.labels_)
            silhouette_scores.append(score)
    
    return {
        'K_range': list(K_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }

def plot_elbow_curve(results):
    """
    Plot elbow curve for determining optimal clusters
    
    Parameters:
    -----------
    results : dict
        Results from find_optimal_clusters
    """
    plt.figure(figsize=(12, 5))
    
    # Elbow plot
    plt.subplot(1, 2, 1)
    plt.plot(results['K_range'], results['inertias'], 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method')
    plt.grid(True)
    
    # Silhouette plot
    if results['silhouette_scores']:
        plt.subplot(1, 2, 2)
        plt.plot(results['K_range'], results['silhouette_scores'], 'ro-')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def perform_kmeans_clustering(X, n_clusters=4, random_state=42):
    """
    Perform K-Means clustering
    
    Parameters:
    -----------
    X : np.array
        Scaled feature matrix
    n_clusters : int
        Number of clusters
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    tuple
        (model, labels, metrics)
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Calculate metrics
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin,
        'inertia': kmeans.inertia_
    }
    
    return kmeans, labels, metrics

def perform_hierarchical_clustering(X, n_clusters=4, linkage_method='ward'):
    """
    Perform Hierarchical clustering
    
    Parameters:
    -----------
    X : np.array
        Scaled feature matrix
    n_clusters : int
        Number of clusters
    linkage_method : str
        Linkage method ('ward', 'complete', 'average', 'single')
    
    Returns:
    --------
    tuple
        (model, labels, metrics)
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = hierarchical.fit_predict(X)
    
    # Calculate metrics
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    metrics = {
        'silhouette_score': silhouette,
        'davies_bouldin_score': davies_bouldin
    }
    
    return hierarchical, labels, metrics

def plot_dendrogram(X, method='ward', max_samples=1000):
    """
    Plot dendrogram for hierarchical clustering
    
    Parameters:
    -----------
    X : np.array
        Scaled feature matrix
    method : str
        Linkage method
    max_samples : int
        Maximum number of samples to plot (for performance)
    """
    # Sample data if too large
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Calculate linkage
    Z = linkage(X_sample, method=method)
    
    # Plot
    plt.figure(figsize=(15, 7))
    dendrogram(Z)
    plt.title(f'Dendrogram (Linkage method: {method})')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()

def perform_dbscan_clustering(X, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering
    
    Parameters:
    -----------
    X : np.array
        Scaled feature matrix
    eps : float
        Maximum distance between two samples
    min_samples : int
        Minimum number of samples in a neighborhood
    
    Returns:
    --------
    tuple
        (model, labels, metrics)
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    
    # Calculate metrics (excluding noise points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_ratio': n_noise / len(labels)
    }
    
    # Calculate silhouette score if we have valid clusters
    if n_clusters > 1 and n_noise < len(labels):
        mask = labels != -1
        if sum(mask) > 0:
            metrics['silhouette_score'] = silhouette_score(X[mask], labels[mask])
    
    return dbscan, labels, metrics

def analyze_clusters(rfm_df, labels, cluster_col='Cluster'):
    """
    Analyze cluster characteristics
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe
    labels : np.array
        Cluster labels
    cluster_col : str
        Name for cluster column
    
    Returns:
    --------
    pd.DataFrame
        Cluster analysis summary
    """
    # Add cluster labels to dataframe
    rfm_with_clusters = rfm_df.copy()
    rfm_with_clusters[cluster_col] = labels
    
    # Calculate cluster statistics
    cluster_summary = rfm_with_clusters.groupby(cluster_col).agg({
        'Recency': ['mean', 'median', 'std'],
        'Frequency': ['mean', 'median', 'std'],
        'Monetary': ['mean', 'median', 'std', 'sum'],
        'CustomerID': 'count'
    }).round(2)
    
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    cluster_summary = cluster_summary.rename(columns={'CustomerID_count': 'Customer_Count'})
    
    # Calculate percentage
    cluster_summary['Percentage'] = (cluster_summary['Customer_Count'] / len(rfm_with_clusters) * 100).round(2)
    
    return cluster_summary

def assign_cluster_names(cluster_summary):
    """
    Assign meaningful names to clusters based on their characteristics
    
    Parameters:
    -----------
    cluster_summary : pd.DataFrame
        Cluster summary statistics
    
    Returns:
    --------
    dict
        Mapping of cluster numbers to names
    """
    cluster_names = {}
    
    for idx in cluster_summary.index:
        r_mean = cluster_summary.loc[idx, 'Recency_mean']
        f_mean = cluster_summary.loc[idx, 'Frequency_mean']
        m_mean = cluster_summary.loc[idx, 'Monetary_mean']
        
        # Determine cluster characteristics
        if r_mean < cluster_summary['Recency_mean'].median() and \
           f_mean > cluster_summary['Frequency_mean'].median() and \
           m_mean > cluster_summary['Monetary_mean'].median():
            cluster_names[idx] = 'Champions'
        elif f_mean > cluster_summary['Frequency_mean'].median():
            cluster_names[idx] = 'Loyal Customers'
        elif r_mean < cluster_summary['Recency_mean'].median():
            cluster_names[idx] = 'Recent Customers'
        elif r_mean > cluster_summary['Recency_mean'].quantile(0.75):
            cluster_names[idx] = 'At Risk / Lost'
        else:
            cluster_names[idx] = 'Potential Customers'
    
    return cluster_names


class UltraAdvancedClustering:
    """
    Ultra-advanced clustering engine with ensemble methods and automatic optimization
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the advanced clustering engine
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.scaler = None
        self.dimensionality_reducer = None
        self.clustering_results = {}
        self.ensemble_results = {}
        self.optimal_clusters = None
        
    def prepare_data(self, X: np.ndarray, scaling_method: str = 'standard',
                    handle_outliers: bool = True) -> np.ndarray:
        """
        Advanced data preparation with outlier handling and scaling
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        scaling_method : str
            Scaling method: 'standard', 'minmax', 'robust'
        handle_outliers : bool
            Whether to handle outliers
        """
        
        # Handle missing values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Outlier detection and handling
        if handle_outliers:
            iso_forest = IsolationForest(contamination=0.1, random_state=self.random_state)
            outliers = iso_forest.fit_predict(X_clean)
            
            # Cap outliers at 95th percentile
            for i in range(X_clean.shape[1]):
                upper_limit = np.percentile(X_clean[:, i], 95)
                lower_limit = np.percentile(X_clean[:, i], 5)
                X_clean[:, i] = np.clip(X_clean[:, i], lower_limit, upper_limit)
        
        # Feature scaling
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        
        X_scaled = self.scaler.fit_transform(X_clean)
        
        return X_scaled
    
    def reduce_dimensionality(self, X: np.ndarray, method: str = 'pca',
                            n_components: Optional[int] = None,
                            explained_variance_threshold: float = 0.95) -> np.ndarray:
        """
        Advanced dimensionality reduction
        
        Parameters:
        -----------
        X : np.ndarray
            Scaled feature matrix
        method : str
            Reduction method: 'pca', 'tsne', 'umap', 'ica', 'factor_analysis'
        n_components : Optional[int]
            Number of components (auto-determined if None)
        """
        
        if method == 'pca':
            if n_components is None:
                # Determine components based on explained variance
                pca_full = PCA()
                pca_full.fit(X)
                cumsum = np.cumsum(pca_full.explained_variance_ratio_)
                n_components = np.argmax(cumsum >= explained_variance_threshold) + 1
                n_components = min(n_components, X.shape[1])
            
            self.dimensionality_reducer = PCA(n_components=n_components, random_state=self.random_state)
            
        elif method == 'tsne':
            n_components = min(n_components or 2, 3)  # t-SNE typically uses 2-3 components
            self.dimensionality_reducer = TSNE(
                n_components=n_components, 
                random_state=self.random_state,
                perplexity=min(30, X.shape[0] - 1)
            )
            
        elif method == 'umap' and UMAP_AVAILABLE:
            n_components = n_components or min(10, X.shape[1])
            self.dimensionality_reducer = umap.UMAP(
                n_components=n_components,
                random_state=self.random_state,
                n_neighbors=min(15, X.shape[0] - 1)
            )
            
        elif method == 'ica':
            n_components = n_components or min(X.shape[1], X.shape[0] - 1)
            self.dimensionality_reducer = FastICA(
                n_components=n_components,
                random_state=self.random_state
            )
            
        elif method == 'factor_analysis':
            n_components = n_components or min(X.shape[1] // 2, 10)
            self.dimensionality_reducer = FactorAnalysis(
                n_components=n_components,
                random_state=self.random_state
            )
        else:
            print(f"Method {method} not available or not supported. Using PCA.")
            return self.reduce_dimensionality(X, 'pca', n_components, explained_variance_threshold)
        
        X_reduced = self.dimensionality_reducer.fit_transform(X)
        
        print(f"Dimensionality reduced from {X.shape[1]} to {X_reduced.shape[1]} using {method}")
        
        return X_reduced
    
    def find_optimal_clusters_advanced(self, X: np.ndarray, max_clusters: int = 15,
                                     methods: List[str] = None) -> Dict[str, Dict]:
        """
        Advanced optimal cluster detection using multiple metrics and methods
        """
        
        if methods is None:
            methods = ['kmeans', 'gaussian_mixture', 'spectral']
            if HDBSCAN_AVAILABLE:
                methods.append('hdbscan')
        
        results = {}
        
        for method in methods:
            print(f"Finding optimal clusters using {method}...")
            
            if method == 'kmeans':
                results[method] = self._optimize_kmeans(X, max_clusters)
            elif method == 'gaussian_mixture':
                results[method] = self._optimize_gaussian_mixture(X, max_clusters)
            elif method == 'spectral':
                results[method] = self._optimize_spectral_clustering(X, max_clusters)
            elif method == 'hdbscan' and HDBSCAN_AVAILABLE:
                results[method] = self._optimize_hdbscan(X)
        
        # Determine overall optimal number of clusters
        self.optimal_clusters = self._consensus_optimal_clusters(results)
        
        return results
    
    def perform_ensemble_clustering(self, X: np.ndarray, n_clusters: int = None,
                                  methods: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Perform ensemble clustering using multiple algorithms
        """
        
        if n_clusters is None:
            n_clusters = self.optimal_clusters or 4
        
        if methods is None:
            methods = [
                'kmeans', 'gaussian_mixture', 'spectral_clustering', 
                'agglomerative', 'dbscan'
            ]
            if HDBSCAN_AVAILABLE:
                methods.append('hdbscan')
        
        ensemble_labels = {}
        
        print(f"Performing ensemble clustering with {len(methods)} methods...")
        
        for method in methods:
            try:
                if method == 'kmeans':
                    labels = self._kmeans_clustering(X, n_clusters)
                elif method == 'gaussian_mixture':
                    labels = self._gaussian_mixture_clustering(X, n_clusters)
                elif method == 'spectral_clustering':
                    labels = self._spectral_clustering(X, n_clusters)
                elif method == 'agglomerative':
                    labels = self._agglomerative_clustering(X, n_clusters)
                elif method == 'dbscan':
                    labels = self._dbscan_clustering(X)
                elif method == 'hdbscan' and HDBSCAN_AVAILABLE:
                    labels = self._hdbscan_clustering(X)
                elif method == 'optics':
                    labels = self._optics_clustering(X)
                elif method == 'birch':
                    labels = self._birch_clustering(X, n_clusters)
                
                if labels is not None and len(set(labels)) > 1:
                    ensemble_labels[method] = labels
                    print(f"✓ {method}: {len(set(labels))} clusters")
                else:
                    print(f"✗ {method}: Failed or single cluster")
                    
            except Exception as e:
                print(f"✗ {method}: Error - {e}")
        
        self.ensemble_results = ensemble_labels
        return ensemble_labels
    
    def create_consensus_clustering(self, ensemble_labels: Dict[str, np.ndarray],
                                  method: str = 'voting') -> np.ndarray:
        """
        Create consensus clustering from ensemble results
        
        Parameters:
        -----------
        ensemble_labels : Dict[str, np.ndarray]
            Dictionary of clustering results from different methods
        method : str
            Consensus method: 'voting', 'weighted_voting', 'co_association'
        """
        
        if not ensemble_labels:
            raise ValueError("No ensemble results available")
        
        n_samples = len(list(ensemble_labels.values())[0])
        
        if method == 'voting':
            # Simple majority voting
            consensus_labels = self._majority_voting_consensus(ensemble_labels, n_samples)
            
        elif method == 'weighted_voting':
            # Weighted voting based on silhouette scores
            consensus_labels = self._weighted_voting_consensus(ensemble_labels, n_samples)
            
        elif method == 'co_association':
            # Co-association matrix based consensus
            consensus_labels = self._co_association_consensus(ensemble_labels, n_samples)
        
        return consensus_labels
    
    def evaluate_clustering_quality(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive clustering quality evaluation
        """
        
        metrics = {}
        
        # Internal validation metrics
        if len(set(labels)) > 1:
            try:
                metrics['silhouette_score'] = silhouette_score(X, labels)
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
                metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
            except:
                pass
        
        # Cluster statistics
        unique_labels = np.unique(labels)
        metrics['n_clusters'] = len(unique_labels)
        
        if -1 in unique_labels:  # Noise points (from DBSCAN/HDBSCAN)
            metrics['n_noise'] = np.sum(labels == -1)
            metrics['noise_ratio'] = metrics['n_noise'] / len(labels)
        
        # Cluster size statistics
        cluster_sizes = [np.sum(labels == label) for label in unique_labels if label != -1]
        if cluster_sizes:
            metrics['min_cluster_size'] = min(cluster_sizes)
            metrics['max_cluster_size'] = max(cluster_sizes)
            metrics['avg_cluster_size'] = np.mean(cluster_sizes)
            metrics['cluster_size_std'] = np.std(cluster_sizes)
            metrics['cluster_balance'] = 1 - (np.std(cluster_sizes) / np.mean(cluster_sizes))
        
        return metrics
    
    def analyze_cluster_stability(self, X: np.ndarray, n_clusters: int,
                                method: str = 'bootstrap', n_iterations: int = 100) -> Dict:
        """
        Analyze clustering stability using bootstrap or cross-validation
        """
        
        stability_scores = []
        
        for i in range(n_iterations):
            # Bootstrap sampling
            if method == 'bootstrap':
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_sample = X[indices]
            else:
                # Random subsampling
                sample_size = int(0.8 * len(X))
                indices = np.random.choice(len(X), size=sample_size, replace=False)
                X_sample = X[indices]
            
            # Perform clustering on sample
            kmeans1 = KMeans(n_clusters=n_clusters, random_state=i, n_init=10)
            labels1 = kmeans1.fit_predict(X_sample)
            
            # Perform clustering on different sample
            indices2 = np.random.choice(len(X), size=len(X_sample), replace=True)
            X_sample2 = X[indices2]
            
            kmeans2 = KMeans(n_clusters=n_clusters, random_state=i+1000, n_init=10)
            labels2 = kmeans2.fit_predict(X_sample2)
            
            # Calculate similarity between clusterings
            if len(set(labels1)) > 1 and len(set(labels2)) > 1:
                try:
                    stability = adjusted_rand_score(labels1, labels2)
                    stability_scores.append(stability)
                except:
                    pass
        
        stability_analysis = {
            'mean_stability': np.mean(stability_scores) if stability_scores else 0,
            'std_stability': np.std(stability_scores) if stability_scores else 0,
            'min_stability': np.min(stability_scores) if stability_scores else 0,
            'max_stability': np.max(stability_scores) if stability_scores else 0,
            'n_iterations': len(stability_scores)
        }
        
        return stability_analysis
    
    def detect_hierarchical_structure(self, X: np.ndarray, method: str = 'ward') -> Dict:
        """
        Detect hierarchical cluster structure
        """
        
        # Calculate linkage matrix
        Z = linkage(X, method=method)
        
        # Find natural cluster levels
        distances = Z[:, 2]
        distance_diffs = np.diff(distances)
        
        # Find significant jumps in distance
        threshold = np.mean(distance_diffs) + 2 * np.std(distance_diffs)
        significant_jumps = np.where(distance_diffs > threshold)[0]
        
        # Suggest cluster numbers based on hierarchical structure
        suggested_clusters = []
        for jump_idx in significant_jumps[-5:]:  # Last 5 significant jumps
            n_clusters = len(Z) - jump_idx + 1
            if 2 <= n_clusters <= 20:
                suggested_clusters.append(n_clusters)
        
        hierarchy_info = {
            'linkage_matrix': Z,
            'distances': distances,
            'distance_jumps': distance_diffs,
            'suggested_clusters': sorted(set(suggested_clusters), reverse=True)
        }
        
        return hierarchy_info
    
    def create_cluster_profiles(self, X: np.ndarray, labels: np.ndarray,
                              feature_names: List[str] = None) -> pd.DataFrame:
        """
        Create detailed cluster profiles with statistical analysis
        """
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        
        # Create dataframe
        df = pd.DataFrame(X, columns=feature_names)
        df['Cluster'] = labels
        
        # Calculate comprehensive statistics
        profiles = []
        
        for cluster_id in sorted(df['Cluster'].unique()):
            if cluster_id == -1:  # Skip noise points
                continue
                
            cluster_data = df[df['Cluster'] == cluster_id]
            
            profile = {'Cluster': cluster_id, 'Size': len(cluster_data)}
            
            for feature in feature_names:
                feature_data = cluster_data[feature]
                
                profile.update({
                    f'{feature}_mean': feature_data.mean(),
                    f'{feature}_median': feature_data.median(),
                    f'{feature}_std': feature_data.std(),
                    f'{feature}_min': feature_data.min(),
                    f'{feature}_max': feature_data.max(),
                    f'{feature}_q25': feature_data.quantile(0.25),
                    f'{feature}_q75': feature_data.quantile(0.75)
                })
            
            profiles.append(profile)
        
        profile_df = pd.DataFrame(profiles)
        
        return profile_df
    
    # Helper methods for specific clustering algorithms
    def _optimize_kmeans(self, X: np.ndarray, max_clusters: int) -> Dict:
        """Optimize K-means clustering"""
        
        results = {'k_values': [], 'inertias': [], 'silhouette_scores': [], 'ch_scores': []}
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            
            results['k_values'].append(k)
            results['inertias'].append(kmeans.inertia_)
            results['silhouette_scores'].append(silhouette_score(X, labels))
            results['ch_scores'].append(calinski_harabasz_score(X, labels))
        
        # Find optimal k using elbow method and silhouette score
        optimal_k_silhouette = results['k_values'][np.argmax(results['silhouette_scores'])]
        optimal_k_elbow = self._find_elbow_point(results['k_values'], results['inertias'])
        
        results['optimal_k_silhouette'] = optimal_k_silhouette
        results['optimal_k_elbow'] = optimal_k_elbow
        
        return results
    
    def _optimize_gaussian_mixture(self, X: np.ndarray, max_clusters: int) -> Dict:
        """Optimize Gaussian Mixture Model clustering"""
        
        results = {'k_values': [], 'bic_scores': [], 'aic_scores': [], 'silhouette_scores': []}
        
        for k in range(2, min(max_clusters + 1, X.shape[0])):
            try:
                gmm = GaussianMixture(n_components=k, random_state=self.random_state)
                labels = gmm.fit_predict(X)
                
                results['k_values'].append(k)
                results['bic_scores'].append(gmm.bic(X))
                results['aic_scores'].append(gmm.aic(X))
                results['silhouette_scores'].append(silhouette_score(X, labels))
            except:
                continue
        
        if results['bic_scores']:
            optimal_k_bic = results['k_values'][np.argmin(results['bic_scores'])]
            optimal_k_silhouette = results['k_values'][np.argmax(results['silhouette_scores'])]
            
            results['optimal_k_bic'] = optimal_k_bic
            results['optimal_k_silhouette'] = optimal_k_silhouette
        
        return results
    
    def _optimize_spectral_clustering(self, X: np.ndarray, max_clusters: int) -> Dict:
        """Optimize Spectral clustering"""
        
        results = {'k_values': [], 'silhouette_scores': []}
        
        for k in range(2, min(max_clusters + 1, X.shape[0])):
            try:
                spectral = SpectralClustering(
                    n_clusters=k, 
                    random_state=self.random_state,
                    affinity='rbf'
                )
                labels = spectral.fit_predict(X)
                
                results['k_values'].append(k)
                results['silhouette_scores'].append(silhouette_score(X, labels))
            except:
                continue
        
        if results['silhouette_scores']:
            optimal_k = results['k_values'][np.argmax(results['silhouette_scores'])]
            results['optimal_k'] = optimal_k
        
        return results
    
    def _optimize_hdbscan(self, X: np.ndarray) -> Dict:
        """Optimize HDBSCAN clustering"""
        
        if not HDBSCAN_AVAILABLE:
            return {}
        
        results = {'min_cluster_sizes': [], 'n_clusters': [], 'silhouette_scores': []}
        
        min_cluster_sizes = [max(2, int(len(X) * ratio)) for ratio in [0.01, 0.02, 0.05, 0.1, 0.15]]
        
        for min_cluster_size in min_cluster_sizes:
            try:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=max(1, min_cluster_size // 2)
                )
                labels = clusterer.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                if n_clusters > 1:
                    # Calculate silhouette score excluding noise
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        silhouette = silhouette_score(X[mask], labels[mask])
                        
                        results['min_cluster_sizes'].append(min_cluster_size)
                        results['n_clusters'].append(n_clusters)
                        results['silhouette_scores'].append(silhouette)
            except:
                continue
        
        if results['silhouette_scores']:
            best_idx = np.argmax(results['silhouette_scores'])
            results['optimal_min_cluster_size'] = results['min_cluster_sizes'][best_idx]
        
        return results
    
    # Individual clustering methods
    def _kmeans_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform K-means clustering"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
        return kmeans.fit_predict(X)
    
    def _gaussian_mixture_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform Gaussian Mixture clustering"""
        gmm = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
        return gmm.fit_predict(X)
    
    def _spectral_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform Spectral clustering"""
        spectral = SpectralClustering(
            n_clusters=n_clusters, 
            random_state=self.random_state,
            affinity='rbf'
        )
        return spectral.fit_predict(X)
    
    def _agglomerative_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform Agglomerative clustering"""
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        return agg.fit_predict(X)
    
    def _dbscan_clustering(self, X: np.ndarray) -> np.ndarray:
        """Perform DBSCAN clustering"""
        # Auto-determine eps using k-distance
        neighbors = NearestNeighbors(n_neighbors=4)
        neighbors_fit = neighbors.fit(X)
        distances, indices = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, 3], axis=0)
        
        # Use knee point as eps
        eps = distances[int(0.9 * len(distances))]
        
        dbscan = DBSCAN(eps=eps, min_samples=4)
        return dbscan.fit_predict(X)
    
    def _hdbscan_clustering(self, X: np.ndarray) -> np.ndarray:
        """Perform HDBSCAN clustering"""
        if not HDBSCAN_AVAILABLE:
            return None
        
        min_cluster_size = max(2, int(len(X) * 0.05))
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=max(1, min_cluster_size // 2)
        )
        return clusterer.fit_predict(X)
    
    def _optics_clustering(self, X: np.ndarray) -> np.ndarray:
        """Perform OPTICS clustering"""
        optics = OPTICS(min_samples=max(2, int(len(X) * 0.05)))
        return optics.fit_predict(X)
    
    def _birch_clustering(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Perform BIRCH clustering"""
        birch = Birch(n_clusters=n_clusters)
        return birch.fit_predict(X)
    
    # Consensus methods
    def _majority_voting_consensus(self, ensemble_labels: Dict[str, np.ndarray], 
                                 n_samples: int) -> np.ndarray:
        """Create consensus using majority voting"""
        
        # Create co-occurrence matrix
        co_matrix = np.zeros((n_samples, n_samples))
        
        for method, labels in ensemble_labels.items():
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] == labels[j] and labels[i] != -1:
                        co_matrix[i, j] += 1
                        co_matrix[j, i] += 1
        
        # Normalize
        co_matrix = co_matrix / len(ensemble_labels)
        
        # Apply threshold and create final clustering
        threshold = 0.5
        consensus_labels = np.zeros(n_samples, dtype=int)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_samples):
            if not visited[i]:
                # Find all points that should be in the same cluster
                cluster_points = np.where(co_matrix[i] >= threshold)[0]
                consensus_labels[cluster_points] = cluster_id
                visited[cluster_points] = True
                cluster_id += 1
        
        return consensus_labels
    
    def _weighted_voting_consensus(self, ensemble_labels: Dict[str, np.ndarray],
                                 n_samples: int) -> np.ndarray:
        """Create consensus using weighted voting based on quality"""
        
        # Calculate weights based on silhouette scores
        weights = {}
        X_dummy = np.random.randn(n_samples, 3)  # Dummy data for weight calculation
        
        for method, labels in ensemble_labels.items():
            try:
                if len(set(labels)) > 1:
                    weights[method] = max(0, silhouette_score(X_dummy, labels))
                else:
                    weights[method] = 0
            except:
                weights[method] = 0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {k: 1 / len(weights) for k in weights.keys()}
        
        # Weighted co-occurrence matrix
        co_matrix = np.zeros((n_samples, n_samples))
        
        for method, labels in ensemble_labels.items():
            weight = weights[method]
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] == labels[j] and labels[i] != -1:
                        co_matrix[i, j] += weight
                        co_matrix[j, i] += weight
        
        # Create consensus clustering
        return self._consensus_from_matrix(co_matrix, threshold=0.3)
    
    def _co_association_consensus(self, ensemble_labels: Dict[str, np.ndarray],
                                n_samples: int) -> np.ndarray:
        """Create consensus using co-association matrix"""
        
        co_matrix = np.zeros((n_samples, n_samples))
        
        # Build co-association matrix
        for method, labels in ensemble_labels.items():
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] == labels[j] and labels[i] != -1:
                        co_matrix[i, j] += 1
                        co_matrix[j, i] += 1
        
        # Normalize by number of methods
        co_matrix = co_matrix / len(ensemble_labels)
        
        # Apply hierarchical clustering on similarity matrix
        distance_matrix = 1 - co_matrix
        
        # Fill diagonal
        np.fill_diagonal(distance_matrix, 0)
        
        # Perform hierarchical clustering
        condensed_distances = squareform(distance_matrix, checks=False)
        Z = linkage(condensed_distances, method='average')
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters_from_dendrogram(Z)
        
        # Get final clustering
        consensus_labels = fcluster(Z, optimal_k, criterion='maxclust') - 1
        
        return consensus_labels
    
    def _consensus_from_matrix(self, co_matrix: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Create clustering from co-association matrix"""
        
        n_samples = co_matrix.shape[0]
        consensus_labels = np.full(n_samples, -1)
        cluster_id = 0
        
        for i in range(n_samples):
            if consensus_labels[i] == -1:  # Not assigned yet
                # Find all points strongly connected to this point
                connected = np.where(co_matrix[i] >= threshold)[0]
                
                if len(connected) > 0:
                    consensus_labels[connected] = cluster_id
                    cluster_id += 1
        
        return consensus_labels
    
    def _consensus_optimal_clusters(self, optimization_results: Dict[str, Dict]) -> int:
        """Determine consensus optimal number of clusters"""
        
        optimal_ks = []
        
        for method, results in optimization_results.items():
            if 'optimal_k_silhouette' in results:
                optimal_ks.append(results['optimal_k_silhouette'])
            elif 'optimal_k_bic' in results:
                optimal_ks.append(results['optimal_k_bic'])
            elif 'optimal_k' in results:
                optimal_ks.append(results['optimal_k'])
        
        if optimal_ks:
            # Return mode or median
            from collections import Counter
            counts = Counter(optimal_ks)
            if len(counts.most_common(1)) > 0:
                return counts.most_common(1)[0][0]
            else:
                return int(np.median(optimal_ks))
        
        return 4  # Default fallback
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in K-means inertia curve"""
        
        if len(k_values) < 3:
            return k_values[0] if k_values else 2
        
        # Calculate rate of change
        differences = np.diff(inertias)
        differences2 = np.diff(differences)
        
        # Find the point where the second derivative is maximum
        elbow_idx = np.argmax(differences2) + 1
        
        return k_values[min(elbow_idx, len(k_values) - 1)]
    
    def _find_optimal_clusters_from_dendrogram(self, Z: np.ndarray) -> int:
        """Find optimal clusters from dendrogram"""
        
        distances = Z[:, 2]
        
        # Look for the largest gap in distances
        if len(distances) > 1:
            gaps = np.diff(distances)
            optimal_idx = np.argmax(gaps)
            optimal_k = len(Z) - optimal_idx
            
            # Ensure reasonable number of clusters
            return max(2, min(optimal_k, 10))
        
        return 3  # Default
