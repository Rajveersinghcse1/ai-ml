"""
Ultra-Advanced RFM Analysis Module
Comprehensive RFM analysis with advanced variants, predictive scoring, and dynamic segmentation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Advanced analytics
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Survival analysis (if available)
try:
    from lifelines import BetaGeoFitter, GammaGammaFitter
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

# Time series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

def calculate_rfm(df, customer_id_col='CustomerID', date_col='InvoiceDate', amount_col='TotalAmount', reference_date=None):
    """
    Calculate RFM metrics for each customer
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
    customer_id_col : str
        Name of customer ID column
    date_col : str
        Name of date column
    amount_col : str
        Name of amount/revenue column
    reference_date : datetime
        Reference date for recency calculation (default: max date + 1 day)
    
    Returns:
    --------
    pd.DataFrame
        RFM dataframe with Recency, Frequency, and Monetary columns
    """
    # Set reference date
    if reference_date is None:
        reference_date = df[date_col].max() + pd.Timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = df.groupby(customer_id_col).agg({
        date_col: lambda x: (reference_date - x.max()).days,  # Recency
        customer_id_col: 'count',  # Frequency
        amount_col: 'sum'  # Monetary
    })
    
    # Rename columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Reset index
    rfm = rfm.reset_index()
    
    return rfm

def calculate_rfm_scores(rfm_df, recency_labels=None, frequency_labels=None, monetary_labels=None):
    """
    Calculate RFM scores using quantile-based segmentation
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe
    recency_labels : list
        Custom labels for recency quantiles (default: [1,2,3,4,5])
    frequency_labels : list
        Custom labels for frequency quantiles (default: [1,2,3,4,5])
    monetary_labels : list
        Custom labels for monetary quantiles (default: [1,2,3,4,5])
    
    Returns:
    --------
    pd.DataFrame
        RFM dataframe with score columns
    """
    rfm = rfm_df.copy()
    
    # Set default labels
    if recency_labels is None:
        recency_labels = [5, 4, 3, 2, 1]  # Reversed for recency (lower is better)
    if frequency_labels is None:
        frequency_labels = [1, 2, 3, 4, 5]
    if monetary_labels is None:
        monetary_labels = [1, 2, 3, 4, 5]
    
    # Calculate quintiles and assign scores
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=recency_labels, duplicates='drop')
    rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=5, labels=frequency_labels, duplicates='drop')
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=monetary_labels, duplicates='drop')
    
    # Convert to numeric
    rfm['R_Score'] = rfm['R_Score'].astype(int)
    rfm['F_Score'] = rfm['F_Score'].astype(int)
    rfm['M_Score'] = rfm['M_Score'].astype(int)
    
    # Calculate RFM Score (concatenated)
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    # Calculate RFM Score (summed)
    rfm['RFM_Score_Sum'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    
    return rfm

def segment_customers(rfm_df):
    """
    Segment customers based on RFM scores into meaningful groups
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with scores
    
    Returns:
    --------
    pd.DataFrame
        RFM dataframe with segment labels
    """
    rfm = rfm_df.copy()
    
    # Define segmentation rules
    def assign_segment(row):
        r, f, m = row['R_Score'], row['F_Score'], row['M_Score']
        
        # Champions: Bought recently, buy often, and spend the most
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        
        # Loyal Customers: Buy regularly, good monetary value
        elif r >= 3 and f >= 4 and m >= 3:
            return 'Loyal Customers'
        
        # Potential Loyalists: Recent customers with average frequency
        elif r >= 4 and f >= 2 and m >= 2:
            return 'Potential Loyalists'
        
        # Recent Customers: Bought recently but not frequently
        elif r >= 4 and f <= 2:
            return 'Recent Customers'
        
        # Promising: Recent shoppers, but haven't spent much
        elif r >= 3 and f <= 2 and m <= 2:
            return 'Promising'
        
        # Customers Needing Attention: Above average recency, frequency, and monetary
        elif r >= 3 and f >= 3 and m >= 3:
            return 'Customers Needing Attention'
        
        # About to Sleep: Below average recency, frequency, and monetary
        elif r <= 3 and f <= 3 and m <= 3:
            return 'About to Sleep'
        
        # At Risk: Spent big money, purchased often but long time ago
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        
        # Cannot Lose Them: Made big purchases and often, but long time ago
        elif r <= 2 and f >= 4 and m >= 4:
            return 'Cannot Lose Them'
        
        # Hibernating: Last purchase long ago, low spenders
        elif r <= 2 and f <= 2 and m <= 2:
            return 'Hibernating'
        
        # Lost: Lowest recency, frequency, and monetary scores
        elif r <= 1:
            return 'Lost'
        
        else:
            return 'Others'
    
    rfm['Segment'] = rfm.apply(assign_segment, axis=1)
    
    return rfm

def get_segment_summary(rfm_df):
    """
    Get summary statistics for each customer segment
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with segments
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics by segment
    """
    summary = rfm_df.groupby('Segment').agg({
        'Recency': ['mean', 'median'],
        'Frequency': ['mean', 'median'],
        'Monetary': ['mean', 'median', 'sum'],
        'CustomerID': 'count'
    }).round(2)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'CustomerID_count': 'Customer_Count'})
    summary = summary.sort_values('Customer_Count', ascending=False)
    
    return summary


class AdvancedRFMAnalyzer:
    """
    Ultra-advanced RFM analysis with multiple variants and predictive capabilities
    """
    
    def __init__(self, df: pd.DataFrame, customer_col: str = 'CustomerID',
                 date_col: str = 'InvoiceDate', amount_col: str = 'TotalAmount',
                 quantity_col: str = 'Quantity', product_col: str = 'StockCode'):
        """
        Initialize Advanced RFM Analyzer
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        """
        self.df = df.copy()
        self.customer_col = customer_col
        self.date_col = date_col
        self.amount_col = amount_col
        self.quantity_col = quantity_col
        self.product_col = product_col
        
        # Prepare data
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        self.reference_date = self.df[self.date_col].max() + pd.Timedelta(days=1)
        
        # RFM variants storage
        self.rfm_variants = {}
        self.segment_models = {}
        
    def calculate_traditional_rfm(self) -> pd.DataFrame:
        """
        Calculate traditional RFM metrics
        """
        
        rfm = self.df.groupby(self.customer_col).agg({
            self.date_col: lambda x: (self.reference_date - x.max()).days,
            self.customer_col: 'count',
            self.amount_col: 'sum'
        })
        
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        rfm = rfm.reset_index()
        
        self.rfm_variants['traditional'] = rfm
        return rfm
    
    def calculate_rfmt_analysis(self) -> pd.DataFrame:
        """
        Calculate RFMT (Recency, Frequency, Monetary, Time) analysis
        Adding Time dimension for customer tenure analysis
        """
        
        # Calculate traditional RFM first
        rfm = self.calculate_traditional_rfm()
        
        # Add Time dimension (customer tenure)
        customer_tenure = self.df.groupby(self.customer_col)[self.date_col].agg(['min', 'max'])
        customer_tenure['Tenure_Days'] = (customer_tenure['max'] - customer_tenure['min']).dt.days + 1
        customer_tenure['Days_Since_First_Purchase'] = (
            self.reference_date - customer_tenure['min']
        ).dt.days
        
        # Merge with RFM
        rfmt = rfm.merge(
            customer_tenure[['Tenure_Days', 'Days_Since_First_Purchase']].reset_index(),
            on=self.customer_col
        )
        
        # Calculate time-based metrics
        rfmt['Purchase_Rate'] = rfmt['Frequency'] / (rfmt['Tenure_Days'] + 1)
        rfmt['Time_Between_Purchases'] = rfmt['Tenure_Days'] / (rfmt['Frequency'] + 1)
        
        # RFMT scores
        rfmt = self._calculate_rfmt_scores(rfmt)
        
        self.rfm_variants['rfmt'] = rfmt
        return rfmt
    
    def calculate_rfmv_analysis(self) -> pd.DataFrame:
        """
        Calculate RFMV (Recency, Frequency, Monetary, Variety) analysis
        Adding Variety dimension for product diversity
        """
        
        # Calculate traditional RFM
        rfm = self.calculate_traditional_rfm()
        
        # Add Variety dimension (product diversity)
        product_variety = self.df.groupby(self.customer_col).agg({
            self.product_col: 'nunique',
            self.quantity_col: ['sum', 'std']
        })
        
        product_variety.columns = ['Product_Variety', 'Total_Quantity', 'Quantity_Std']
        product_variety['Quantity_Consistency'] = (
            1 - (product_variety['Quantity_Std'] / product_variety['Total_Quantity'].replace(0, 1))
        ).fillna(1)
        
        # Merge with RFM
        rfmv = rfm.merge(product_variety.reset_index(), on=self.customer_col)
        
        # Calculate variety-based metrics
        rfmv['Avg_Products_Per_Purchase'] = rfmv['Product_Variety'] / rfmv['Frequency']
        rfmv['Product_Exploration_Rate'] = rfmv['Product_Variety'] / rfmv['Total_Quantity']
        
        # RFMV scores
        rfmv = self._calculate_rfmv_scores(rfmv)
        
        self.rfm_variants['rfmv'] = rfmv
        return rfmv
    
    def calculate_weighted_rfm(self, recency_weight: float = 0.3, 
                             frequency_weight: float = 0.3, 
                             monetary_weight: float = 0.4) -> pd.DataFrame:
        """
        Calculate weighted RFM with custom weights for business priorities
        """
        
        rfm = self.calculate_traditional_rfm()
        
        # Normalize RFM values to 0-1 scale
        scaler = MinMaxScaler()
        rfm_normalized = rfm.copy()
        
        # Invert recency (lower recency should have higher score)
        rfm_normalized['Recency_Inverted'] = rfm['Recency'].max() - rfm['Recency']
        
        # Scale all metrics
        rfm_normalized[['Recency_Norm', 'Frequency_Norm', 'Monetary_Norm']] = scaler.fit_transform(
            rfm_normalized[['Recency_Inverted', 'Frequency', 'Monetary']]
        )
        
        # Calculate weighted score
        rfm_normalized['Weighted_RFM_Score'] = (
            recency_weight * rfm_normalized['Recency_Norm'] +
            frequency_weight * rfm_normalized['Frequency_Norm'] +
            monetary_weight * rfm_normalized['Monetary_Norm']
        )
        
        # Segment based on weighted score
        rfm_normalized['Weighted_Segment'] = pd.qcut(
            rfm_normalized['Weighted_RFM_Score'], 
            q=5, 
            labels=['Low Value', 'Bronze', 'Silver', 'Gold', 'Platinum']
        )
        
        self.rfm_variants['weighted'] = rfm_normalized
        return rfm_normalized
    
    def calculate_dynamic_rfm(self, time_periods: List[int] = [30, 90, 180, 365]) -> pd.DataFrame:
        """
        Calculate dynamic RFM across different time periods
        """
        
        dynamic_rfm_results = {}
        
        for period in time_periods:
            cutoff_date = self.reference_date - pd.Timedelta(days=period)
            period_data = self.df[self.df[self.date_col] >= cutoff_date]
            
            if len(period_data) > 0:
                period_rfm = period_data.groupby(self.customer_col).agg({
                    self.date_col: lambda x: (self.reference_date - x.max()).days,
                    self.customer_col: 'count',
                    self.amount_col: 'sum'
                })
                
                period_rfm.columns = [f'Recency_{period}d', f'Frequency_{period}d', f'Monetary_{period}d']
                dynamic_rfm_results[f'{period}d'] = period_rfm
        
        # Combine all periods
        if dynamic_rfm_results:
            dynamic_rfm = pd.concat(dynamic_rfm_results.values(), axis=1)
            dynamic_rfm = dynamic_rfm.fillna(0).reset_index()
            
            # Calculate trend metrics
            if len(time_periods) >= 2:
                dynamic_rfm = self._calculate_rfm_trends(dynamic_rfm, time_periods)
            
            self.rfm_variants['dynamic'] = dynamic_rfm
            return dynamic_rfm
        
        return pd.DataFrame()
    
    def calculate_predictive_rfm(self) -> pd.DataFrame:
        """
        Calculate predictive RFM using customer lifetime value modeling
        """
        
        if not LIFELINES_AVAILABLE:
            print("lifelines not available. Calculating simplified predictive RFM.")
            return self._calculate_simplified_predictive_rfm()
        
        try:
            # Prepare data for BG/NBD and Gamma-Gamma models
            customer_summary = self.df.groupby(self.customer_col).agg({
                self.date_col: ['min', 'max', 'count'],
                self.amount_col: 'sum'
            })
            
            customer_summary.columns = ['first_purchase', 'last_purchase', 'frequency', 'monetary']
            
            # Calculate required metrics
            customer_summary['T'] = (self.reference_date - customer_summary['first_purchase']).dt.days
            customer_summary['recency'] = (
                customer_summary['last_purchase'] - customer_summary['first_purchase']
            ).dt.days
            customer_summary['frequency'] = customer_summary['frequency'] - 1  # Repeat purchases
            customer_summary['avg_order_value'] = (
                customer_summary['monetary'] / (customer_summary['frequency'] + 1)
            )
            
            # Filter valid customers
            valid_customers = customer_summary[
                (customer_summary['frequency'] >= 0) & 
                (customer_summary['T'] > 0) &
                (customer_summary['monetary'] > 0)
            ].copy()
            
            if len(valid_customers) < 10:
                print("Insufficient data for predictive modeling.")
                return self._calculate_simplified_predictive_rfm()
            
            # Fit BG/NBD model
            bgf = BetaGeoFitter(penalizer_coef=0.001)
            bgf.fit(valid_customers['frequency'], valid_customers['recency'], valid_customers['T'])
            
            # Predict future purchases
            valid_customers['predicted_purchases_30d'] = bgf.predict(
                30, valid_customers['frequency'], valid_customers['recency'], valid_customers['T']
            )
            valid_customers['predicted_purchases_90d'] = bgf.predict(
                90, valid_customers['frequency'], valid_customers['recency'], valid_customers['T']
            )
            
            # Calculate probability of being alive
            valid_customers['probability_alive'] = bgf.conditional_probability_alive(
                valid_customers['frequency'], valid_customers['recency'], valid_customers['T']
            )
            
            # Fit Gamma-Gamma model for monetary prediction
            repeat_customers = valid_customers[valid_customers['frequency'] > 0]
            
            if len(repeat_customers) > 5:
                ggf = GammaGammaFitter(penalizer_coef=0.001)
                ggf.fit(repeat_customers['frequency'], repeat_customers['avg_order_value'])
                
                # Predict CLV
                valid_customers['predicted_clv_90d'] = 0
                valid_customers.loc[repeat_customers.index, 'predicted_clv_90d'] = ggf.customer_lifetime_value(
                    bgf,
                    repeat_customers['frequency'],
                    repeat_customers['recency'],
                    repeat_customers['T'],
                    repeat_customers['avg_order_value'],
                    time=90,
                    freq='D'
                )
            
            # Create predictive RFM segments
            valid_customers = self._create_predictive_segments(valid_customers)
            
            self.rfm_variants['predictive'] = valid_customers.reset_index()
            return valid_customers.reset_index()
            
        except Exception as e:
            print(f"Error in predictive RFM calculation: {e}")
            return self._calculate_simplified_predictive_rfm()
    
    def calculate_behavioral_rfm(self) -> pd.DataFrame:
        """
        Calculate behavioral RFM incorporating customer behavior patterns
        """
        
        # Start with traditional RFM
        rfm = self.calculate_traditional_rfm()
        
        # Add behavioral metrics
        behavioral_metrics = []
        
        for customer_id in rfm[self.customer_col].unique():
            customer_data = self.df[self.df[self.customer_col] == customer_id]
            
            # Calculate behavioral patterns
            purchase_intervals = customer_data[self.date_col].diff().dt.days.dropna()
            
            behavior = {
                self.customer_col: customer_id,
                'Purchase_Regularity': 1 / (purchase_intervals.std() + 1) if len(purchase_intervals) > 1 else 1,
                'Seasonal_Preference': self._calculate_seasonal_preference(customer_data),
                'Weekend_Preference': self._calculate_weekend_preference(customer_data),
                'Bulk_Purchase_Tendency': customer_data[self.quantity_col].std() / customer_data[self.quantity_col].mean()
                    if customer_data[self.quantity_col].mean() > 0 else 0,
                'Price_Sensitivity': self._calculate_price_sensitivity(customer_data),
                'Product_Loyalty': self._calculate_product_loyalty(customer_data)
            }
            
            behavioral_metrics.append(behavior)
        
        behavioral_df = pd.DataFrame(behavioral_metrics)
        
        # Merge with RFM
        behavioral_rfm = rfm.merge(behavioral_df, on=self.customer_col)
        
        # Calculate behavioral scores
        behavioral_rfm = self._calculate_behavioral_scores(behavioral_rfm)
        
        self.rfm_variants['behavioral'] = behavioral_rfm
        return behavioral_rfm
    
    def detect_rfm_anomalies(self, method: str = 'isolation_forest') -> pd.DataFrame:
        """
        Detect anomalous customers using RFM metrics
        """
        
        rfm = self.calculate_traditional_rfm()
        
        if method == 'isolation_forest':
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            rfm_features = rfm[['Recency', 'Frequency', 'Monetary']].values
            
            # Standardize features
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm_features)
            
            # Detect anomalies
            anomaly_labels = iso_forest.fit_predict(rfm_scaled)
            rfm['Anomaly_Score'] = iso_forest.decision_function(rfm_scaled)
            rfm['Is_Anomaly'] = anomaly_labels == -1
            
        elif method == 'statistical':
            # Use statistical methods (Z-score based)
            rfm_numeric = rfm[['Recency', 'Frequency', 'Monetary']]
            
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(rfm_numeric))
            
            # Flag as anomaly if any metric has |Z-score| > 3
            rfm['Is_Anomaly'] = (z_scores > 3).any(axis=1)
            rfm['Max_Z_Score'] = z_scores.max(axis=1)
        
        return rfm
    
    def optimize_segmentation(self, rfm_variant: str = 'traditional', 
                            max_clusters: int = 10) -> Dict:
        """
        Optimize customer segmentation using multiple clustering algorithms
        """
        
        if rfm_variant not in self.rfm_variants:
            print(f"RFM variant '{rfm_variant}' not calculated yet.")
            return {}
        
        rfm_data = self.rfm_variants[rfm_variant]
        
        # Prepare features for clustering
        if rfm_variant == 'traditional':
            feature_cols = ['Recency', 'Frequency', 'Monetary']
        elif rfm_variant == 'rfmt':
            feature_cols = ['Recency', 'Frequency', 'Monetary', 'Tenure_Days', 'Purchase_Rate']
        elif rfm_variant == 'rfmv':
            feature_cols = ['Recency', 'Frequency', 'Monetary', 'Product_Variety']
        else:
            feature_cols = [col for col in rfm_data.columns if col not in [self.customer_col]]
        
        # Filter numeric columns only
        numeric_cols = rfm_data[feature_cols].select_dtypes(include=[np.number]).columns
        X = rfm_data[numeric_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Test different clustering algorithms
        clustering_results = {}
        
        # K-Means clustering
        kmeans_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            
            if len(set(labels)) > 1:  # Ensure multiple clusters
                silhouette = silhouette_score(X_scaled, labels)
                calinski = calinski_harabasz_score(X_scaled, labels)
                kmeans_scores.append({
                    'n_clusters': k,
                    'silhouette_score': silhouette,
                    'calinski_harabasz_score': calinski,
                    'model': kmeans,
                    'labels': labels
                })
        
        if kmeans_scores:
            # Choose best K-means based on silhouette score
            best_kmeans = max(kmeans_scores, key=lambda x: x['silhouette_score'])
            clustering_results['kmeans'] = best_kmeans
        
        # Gaussian Mixture Model
        gmm_scores = []
        for k in range(2, min(max_clusters + 1, len(X_scaled))):
            try:
                gmm = GaussianMixture(n_components=k, random_state=42)
                labels = gmm.fit_predict(X_scaled)
                
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(X_scaled, labels)
                    bic = gmm.bic(X_scaled)
                    aic = gmm.aic(X_scaled)
                    
                    gmm_scores.append({
                        'n_clusters': k,
                        'silhouette_score': silhouette,
                        'bic': bic,
                        'aic': aic,
                        'model': gmm,
                        'labels': labels
                    })
            except:
                continue
        
        if gmm_scores:
            # Choose best GMM based on BIC (lower is better)
            best_gmm = min(gmm_scores, key=lambda x: x['bic'])
            clustering_results['gaussian_mixture'] = best_gmm
        
        # DBSCAN (density-based)
        try:
            # Test different epsilon values
            eps_values = [0.3, 0.5, 0.7, 1.0, 1.5]
            dbscan_scores = []
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=5)
                labels = dbscan.fit_predict(X_scaled)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters > 1:
                    # Exclude noise points for silhouette calculation
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        silhouette = silhouette_score(X_scaled[mask], labels[mask])
                        dbscan_scores.append({
                            'eps': eps,
                            'n_clusters': n_clusters,
                            'silhouette_score': silhouette,
                            'noise_ratio': np.sum(labels == -1) / len(labels),
                            'model': dbscan,
                            'labels': labels
                        })
            
            if dbscan_scores:
                # Choose DBSCAN with best silhouette score and reasonable noise ratio
                valid_dbscan = [s for s in dbscan_scores if s['noise_ratio'] < 0.3]
                if valid_dbscan:
                    best_dbscan = max(valid_dbscan, key=lambda x: x['silhouette_score'])
                    clustering_results['dbscan'] = best_dbscan
                
        except Exception as e:
            print(f"DBSCAN clustering failed: {e}")
        
        # Store results
        self.segment_models[rfm_variant] = clustering_results
        
        return clustering_results
    
    def create_segment_profiles(self, rfm_variant: str = 'traditional', 
                              clustering_method: str = 'kmeans') -> pd.DataFrame:
        """
        Create detailed profiles for each customer segment
        """
        
        if rfm_variant not in self.segment_models:
            print("No segmentation results available. Run optimize_segmentation first.")
            return pd.DataFrame()
        
        if clustering_method not in self.segment_models[rfm_variant]:
            print(f"Clustering method '{clustering_method}' not available.")
            return pd.DataFrame()
        
        # Get clustering results
        clustering_result = self.segment_models[rfm_variant][clustering_method]
        labels = clustering_result['labels']
        
        # Get RFM data
        rfm_data = self.rfm_variants[rfm_variant].copy()
        rfm_data['Segment'] = labels
        
        # Calculate segment profiles
        numeric_cols = rfm_data.select_dtypes(include=[np.number]).columns
        numeric_cols = numeric_cols.drop('Segment')
        
        segment_profiles = rfm_data.groupby('Segment')[numeric_cols].agg(['mean', 'median', 'std']).round(2)
        
        # Add segment sizes
        segment_sizes = rfm_data.groupby('Segment').size().to_frame('Size')
        segment_profiles = segment_profiles.join(segment_sizes)
        
        # Add percentage
        segment_profiles[('Size', 'Percentage')] = (
            segment_profiles[('Size', 'Size')] / len(rfm_data) * 100
        ).round(1)
        
        return segment_profiles
    
    def generate_rfm_insights(self) -> Dict[str, any]:
        """
        Generate comprehensive insights from RFM analysis
        """
        
        insights = {
            'summary_statistics': {},
            'segment_analysis': {},
            'recommendations': [],
            'key_findings': []
        }
        
        # Analyze each RFM variant
        for variant_name, rfm_data in self.rfm_variants.items():
            
            if len(rfm_data) == 0:
                continue
            
            # Summary statistics
            insights['summary_statistics'][variant_name] = {
                'total_customers': len(rfm_data),
                'avg_recency': rfm_data.get('Recency', pd.Series()).mean(),
                'avg_frequency': rfm_data.get('Frequency', pd.Series()).mean(),
                'avg_monetary': rfm_data.get('Monetary', pd.Series()).mean(),
                'total_revenue': rfm_data.get('Monetary', pd.Series()).sum()
            }
            
            # Segment analysis (if segments exist)
            if 'Segment' in rfm_data.columns:
                segment_stats = rfm_data.groupby('Segment').agg({
                    'Recency': 'mean',
                    'Frequency': 'mean', 
                    'Monetary': ['mean', 'sum'],
                    self.customer_col: 'count'
                })
                insights['segment_analysis'][variant_name] = segment_stats.to_dict()
        
        # Generate key findings
        if 'traditional' in self.rfm_variants:
            traditional_rfm = self.rfm_variants['traditional']
            
            # High-value customers
            high_value = traditional_rfm.nlargest(10, 'Monetary')
            insights['key_findings'].append(
                f"Top 10 customers contribute ${high_value['Monetary'].sum():,.2f} "
                f"({high_value['Monetary'].sum()/traditional_rfm['Monetary'].sum()*100:.1f}% of total revenue)"
            )
            
            # At-risk customers
            at_risk = traditional_rfm[
                (traditional_rfm['Recency'] > traditional_rfm['Recency'].quantile(0.8)) &
                (traditional_rfm['Monetary'] > traditional_rfm['Monetary'].quantile(0.6))
            ]
            if len(at_risk) > 0:
                insights['key_findings'].append(
                    f"{len(at_risk)} high-value customers are at risk of churning (haven't purchased recently)"
                )
            
            # Frequency insights
            single_purchase_customers = (traditional_rfm['Frequency'] == 1).sum()
            insights['key_findings'].append(
                f"{single_purchase_customers} customers ({single_purchase_customers/len(traditional_rfm)*100:.1f}%) "
                f"have made only one purchase"
            )
        
        # Generate recommendations
        insights['recommendations'] = self._generate_actionable_recommendations(insights)
        
        return insights
    
    # Helper methods
    def _calculate_rfmt_scores(self, rfmt_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFMT scores"""
        
        rfmt = rfmt_df.copy()
        
        # Traditional RFM scores (quintiles)
        rfmt['R_Score'] = pd.qcut(rfmt['Recency'], q=5, labels=[5,4,3,2,1], duplicates='drop')
        rfmt['F_Score'] = pd.qcut(rfmt['Frequency'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        rfmt['M_Score'] = pd.qcut(rfmt['Monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Time score (tenure and purchase rate)
        rfmt['T_Score'] = pd.qcut(rfmt['Purchase_Rate'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Convert to numeric
        for col in ['R_Score', 'F_Score', 'M_Score', 'T_Score']:
            rfmt[col] = rfmt[col].astype(int)
        
        # Combined RFMT score
        rfmt['RFMT_Score'] = (
            rfmt['R_Score'].astype(str) + 
            rfmt['F_Score'].astype(str) + 
            rfmt['M_Score'].astype(str) + 
            rfmt['T_Score'].astype(str)
        )
        
        return rfmt
    
    def _calculate_rfmv_scores(self, rfmv_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFMV scores"""
        
        rfmv = rfmv_df.copy()
        
        # Traditional RFM scores
        rfmv['R_Score'] = pd.qcut(rfmv['Recency'], q=5, labels=[5,4,3,2,1], duplicates='drop')
        rfmv['F_Score'] = pd.qcut(rfmv['Frequency'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        rfmv['M_Score'] = pd.qcut(rfmv['Monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Variety score
        rfmv['V_Score'] = pd.qcut(rfmv['Product_Variety'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # Convert to numeric
        for col in ['R_Score', 'F_Score', 'M_Score', 'V_Score']:
            rfmv[col] = rfmv[col].astype(int)
        
        return rfmv
    
    def _calculate_rfm_trends(self, dynamic_rfm: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """Calculate RFM trends across time periods"""
        
        # Calculate trends for each metric
        for metric in ['Frequency', 'Monetary']:
            metric_cols = [f'{metric}_{p}d' for p in periods if f'{metric}_{p}d' in dynamic_rfm.columns]
            
            if len(metric_cols) >= 2:
                # Calculate trend slope
                trends = []
                for idx, row in dynamic_rfm.iterrows():
                    values = [row[col] for col in metric_cols]
                    x = np.arange(len(values))
                    
                    if np.var(values) > 0:
                        slope = np.polyfit(x, values, 1)[0]
                    else:
                        slope = 0
                    
                    trends.append(slope)
                
                dynamic_rfm[f'{metric}_Trend'] = trends
        
        return dynamic_rfm
    
    def _calculate_simplified_predictive_rfm(self) -> pd.DataFrame:
        """Calculate simplified predictive RFM without lifelines"""
        
        rfm = self.calculate_traditional_rfm()
        
        # Simple trend analysis
        customer_trends = []
        
        for customer_id in rfm[self.customer_col].unique():
            customer_data = self.df[self.df[self.customer_col] == customer_id]
            
            # Calculate simple predictions based on historical patterns
            recent_period = customer_data[self.date_col].max() - pd.Timedelta(days=90)
            recent_data = customer_data[customer_data[self.date_col] >= recent_period]
            
            if len(recent_data) > 0 and len(customer_data) > 1:
                recent_frequency = len(recent_data)
                total_frequency = len(customer_data)
                
                # Simple momentum calculation
                momentum = recent_frequency / (90/365) if total_frequency > 0 else 0
                
                # Predict next 90 days activity
                predicted_frequency = min(momentum * (90/365), 10)  # Cap at 10
                predicted_monetary = recent_data[self.amount_col].mean() * predicted_frequency
                
            else:
                predicted_frequency = 0
                predicted_monetary = 0
            
            customer_trends.append({
                self.customer_col: customer_id,
                'predicted_purchases_90d': predicted_frequency,
                'predicted_clv_90d': predicted_monetary,
                'activity_momentum': momentum if 'momentum' in locals() else 0
            })
        
        trends_df = pd.DataFrame(customer_trends)
        predictive_rfm = rfm.merge(trends_df, on=self.customer_col)
        
        return predictive_rfm
    
    def _create_predictive_segments(self, clv_data: pd.DataFrame) -> pd.DataFrame:
        """Create segments based on predictive metrics"""
        
        # Define segments based on probability alive and predicted CLV
        def assign_predictive_segment(row):
            prob_alive = row.get('probability_alive', 0)
            pred_clv = row.get('predicted_clv_90d', 0)
            
            if prob_alive >= 0.7 and pred_clv >= np.percentile(clv_data['predicted_clv_90d'], 75):
                return 'High Value Active'
            elif prob_alive >= 0.7 and pred_clv >= np.percentile(clv_data['predicted_clv_90d'], 50):
                return 'Medium Value Active' 
            elif prob_alive >= 0.7:
                return 'Active'
            elif prob_alive >= 0.3:
                return 'At Risk'
            else:
                return 'Lost'
        
        clv_data['Predictive_Segment'] = clv_data.apply(assign_predictive_segment, axis=1)
        return clv_data
    
    def _calculate_seasonal_preference(self, customer_data: pd.DataFrame) -> float:
        """Calculate customer's seasonal purchase preference"""
        
        customer_data['Month'] = customer_data[self.date_col].dt.month
        monthly_purchases = customer_data['Month'].value_counts()
        
        # Calculate coefficient of variation (higher = more seasonal)
        if len(monthly_purchases) > 1:
            return monthly_purchases.std() / monthly_purchases.mean()
        return 0
    
    def _calculate_weekend_preference(self, customer_data: pd.DataFrame) -> float:
        """Calculate customer's weekend shopping preference"""
        
        customer_data['DayOfWeek'] = customer_data[self.date_col].dt.dayofweek
        weekend_purchases = customer_data[customer_data['DayOfWeek'].isin([5, 6])].shape[0]
        total_purchases = customer_data.shape[0]
        
        return weekend_purchases / total_purchases if total_purchases > 0 else 0
    
    def _calculate_price_sensitivity(self, customer_data: pd.DataFrame) -> float:
        """Calculate customer's price sensitivity"""
        
        if 'UnitPrice' in customer_data.columns and len(customer_data) > 1:
            price_std = customer_data['UnitPrice'].std()
            price_mean = customer_data['UnitPrice'].mean()
            return price_std / price_mean if price_mean > 0 else 0
        return 0
    
    def _calculate_product_loyalty(self, customer_data: pd.DataFrame) -> float:
        """Calculate customer's product loyalty"""
        
        product_counts = customer_data[self.product_col].value_counts()
        
        if len(product_counts) > 0:
            # Herfindahl index for concentration
            proportions = product_counts / product_counts.sum()
            return (proportions ** 2).sum()
        return 0
    
    def _calculate_behavioral_scores(self, behavioral_rfm: pd.DataFrame) -> pd.DataFrame:
        """Calculate behavioral scores"""
        
        behavioral_metrics = [
            'Purchase_Regularity', 'Seasonal_Preference', 'Weekend_Preference',
            'Bulk_Purchase_Tendency', 'Price_Sensitivity', 'Product_Loyalty'
        ]
        
        # Normalize behavioral metrics to 1-5 scale
        for metric in behavioral_metrics:
            if metric in behavioral_rfm.columns:
                behavioral_rfm[f'{metric}_Score'] = pd.qcut(
                    behavioral_rfm[metric], q=5, labels=[1,2,3,4,5], duplicates='drop'
                ).astype(int)
        
        return behavioral_rfm
    
    def _generate_actionable_recommendations(self, insights: Dict) -> List[str]:
        """Generate actionable business recommendations"""
        
        recommendations = []
        
        # Based on segment analysis
        if 'traditional' in insights['summary_statistics']:
            stats = insights['summary_statistics']['traditional']
            
            if stats['avg_recency'] > 60:
                recommendations.append(
                    "Launch a win-back campaign for customers who haven't purchased in 60+ days"
                )
            
            if stats['avg_frequency'] < 2:
                recommendations.append(
                    "Implement a loyalty program to encourage repeat purchases"
                )
        
        # High-value customer recommendations
        recommendations.append(
            "Create VIP program for top 10% customers by monetary value"
        )
        
        # At-risk customer recommendations
        recommendations.append(
            "Deploy targeted retention campaigns for high-value customers with recent low activity"
        )
        
        # New customer recommendations
        recommendations.append(
            "Develop onboarding sequence for single-purchase customers to drive second purchase"
        )
        
        return recommendations
