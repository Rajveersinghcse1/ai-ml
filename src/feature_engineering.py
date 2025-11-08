"""
Advanced Feature Engineering Module
Comprehensive feature extraction for customer segmentation in e-commerce
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from operator import attrgetter
import warnings
warnings.filterwarnings('ignore')


class CustomerFeatureEngineer:
    """
    Advanced feature engineering for customer segmentation
    """
    
    def __init__(self, df, customer_id='CustomerID', date_col='InvoiceDate', 
                 amount_col='TotalAmount', quantity_col='Quantity', 
                 product_col='Description'):
        """
        Initialize the feature engineer
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        customer_id : str
            Customer ID column name
        date_col : str
            Date column name
        amount_col : str
            Amount/price column name
        quantity_col : str
            Quantity column name
        product_col : str
            Product description column name
        """
        self.df = df.copy()
        self.customer_id = customer_id
        self.date_col = date_col
        self.amount_col = amount_col
        self.quantity_col = quantity_col
        self.product_col = product_col
        self.reference_date = df[date_col].max() + pd.Timedelta(days=1)
        
    def calculate_clv_features(self):
        """
        Calculate Customer Lifetime Value (CLV) features
        
        Returns:
        --------
        pd.DataFrame
            CLV features for each customer
        """
        # Historical CLV (total value to date)
        historical_clv = self.df.groupby(self.customer_id).agg({
            self.amount_col: 'sum'
        }).rename(columns={self.amount_col: 'Historical_CLV'})
        
        # Average order value
        avg_order_value = self.df.groupby(self.customer_id).agg({
            self.amount_col: 'mean'
        }).rename(columns={self.amount_col: 'Avg_Order_Value'})
        
        # Purchase frequency
        purchase_frequency = self.df.groupby(self.customer_id).size().to_frame('Purchase_Frequency')
        
        # Customer lifespan in days
        customer_lifespan = self.df.groupby(self.customer_id)[self.date_col].agg(
            lambda x: (x.max() - x.min()).days
        ).to_frame('Customer_Lifespan_Days')
        
        # Predicted CLV (simple model: AOV * Purchase Frequency * Expected Lifespan)
        clv_features = pd.concat([historical_clv, avg_order_value, purchase_frequency, customer_lifespan], axis=1)
        
        # Predict future value (next 12 months)
        # If customer active < 365 days, annualize the metrics
        clv_features['Days_Since_First_Purchase'] = (self.reference_date - 
            self.df.groupby(self.customer_id)[self.date_col].min()).dt.days
        
        clv_features['Purchase_Rate'] = clv_features['Purchase_Frequency'] / (
            clv_features['Customer_Lifespan_Days'] + 1
        )
        
        # Predicted 12-month CLV
        clv_features['Predicted_12M_CLV'] = (
            clv_features['Avg_Order_Value'] * 
            clv_features['Purchase_Rate'] * 365
        )
        
        # CLV category
        clv_features['CLV_Segment'] = pd.qcut(
            clv_features['Predicted_12M_CLV'], 
            q=5, 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
            duplicates='drop'
        )
        
        return clv_features
    
    def calculate_behavioral_features(self):
        """
        Calculate behavioral features from transaction patterns
        
        Returns:
        --------
        pd.DataFrame
            Behavioral features for each customer
        """
        behavioral = pd.DataFrame()
        
        # Distinct products purchased
        behavioral['Unique_Products'] = self.df.groupby(self.customer_id)[self.product_col].nunique()
        
        # Average items per order
        behavioral['Avg_Items_Per_Order'] = self.df.groupby(self.customer_id)[self.quantity_col].mean()
        
        # Total items purchased
        behavioral['Total_Items'] = self.df.groupby(self.customer_id)[self.quantity_col].sum()
        
        # Std of purchase amounts (variability)
        behavioral['Purchase_Amount_Std'] = self.df.groupby(self.customer_id)[self.amount_col].std()
        behavioral['Purchase_Amount_Std'] = behavioral['Purchase_Amount_Std'].fillna(0)
        
        # Coefficient of variation
        mean_amount = self.df.groupby(self.customer_id)[self.amount_col].mean()
        behavioral['Purchase_Variability'] = behavioral['Purchase_Amount_Std'] / (mean_amount + 0.01)
        
        # Max single purchase
        behavioral['Max_Single_Purchase'] = self.df.groupby(self.customer_id)[self.amount_col].max()
        
        # Min single purchase
        behavioral['Min_Single_Purchase'] = self.df.groupby(self.customer_id)[self.amount_col].min()
        
        # Purchase concentration (Gini coefficient approximation)
        def gini_coefficient(x):
            """Calculate Gini coefficient for purchase concentration"""
            x_sorted = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(x_sorted)
            return (2 * np.sum((np.arange(1, n+1)) * x_sorted)) / (n * cumsum[-1]) - (n + 1) / n
        
        behavioral['Purchase_Concentration'] = self.df.groupby(self.customer_id)[self.amount_col].apply(
            lambda x: gini_coefficient(x.values) if len(x) > 1 else 0
        )
        
        return behavioral
    
    def calculate_temporal_features(self):
        """
        Calculate time-based features
        
        Returns:
        --------
        pd.DataFrame
            Temporal features for each customer
        """
        temporal = pd.DataFrame()
        
        # Ensure date column is datetime
        self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        
        # Add time components
        self.df['DayOfWeek'] = self.df[self.date_col].dt.dayofweek
        self.df['Month'] = self.df[self.date_col].dt.month
        self.df['Hour'] = self.df[self.date_col].dt.hour
        self.df['Quarter'] = self.df[self.date_col].dt.quarter
        
        # Favorite day of week
        temporal['Favorite_DayOfWeek'] = self.df.groupby(self.customer_id)['DayOfWeek'].agg(
            lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
        )
        
        # Weekend vs weekday ratio
        self.df['Is_Weekend'] = self.df['DayOfWeek'].isin([5, 6]).astype(int)
        weekend_purchases = self.df.groupby(self.customer_id)['Is_Weekend'].sum()
        total_purchases = self.df.groupby(self.customer_id).size()
        temporal['Weekend_Purchase_Ratio'] = weekend_purchases / total_purchases
        
        # Favorite hour (if available)
        if 'Hour' in self.df.columns and self.df['Hour'].notna().any():
            temporal['Favorite_Hour'] = self.df.groupby(self.customer_id)['Hour'].agg(
                lambda x: x.mode()[0] if len(x.mode()) > 0 else -1
            )
        
        # Average days between purchases
        def avg_days_between_purchases(dates):
            if len(dates) < 2:
                return np.nan
            dates_sorted = sorted(dates)
            diffs = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
            return np.mean(diffs)
        
        temporal['Avg_Days_Between_Purchases'] = self.df.groupby(self.customer_id)[self.date_col].apply(
            avg_days_between_purchases
        )
        temporal['Avg_Days_Between_Purchases'] = temporal['Avg_Days_Between_Purchases'].fillna(365)
        
        # Purchase regularity (std of days between purchases)
        def std_days_between_purchases(dates):
            if len(dates) < 2:
                return 0
            dates_sorted = sorted(dates)
            diffs = [(dates_sorted[i+1] - dates_sorted[i]).days for i in range(len(dates_sorted)-1)]
            return np.std(diffs)
        
        temporal['Purchase_Regularity_Std'] = self.df.groupby(self.customer_id)[self.date_col].apply(
            std_days_between_purchases
        )
        
        # Days since first purchase
        temporal['Days_Since_First_Purchase'] = (
            self.reference_date - self.df.groupby(self.customer_id)[self.date_col].min()
        ).dt.days
        
        # Days since last purchase
        temporal['Days_Since_Last_Purchase'] = (
            self.reference_date - self.df.groupby(self.customer_id)[self.date_col].max()
        ).dt.days
        
        # Active months
        temporal['Active_Months'] = self.df.groupby(self.customer_id)[self.date_col].apply(
            lambda x: x.dt.to_period('M').nunique()
        )
        
        # Trend analysis (is customer increasing or decreasing spend?)
        def calculate_trend(group):
            """Calculate spending trend using linear regression"""
            if len(group) < 3:
                return 0
            dates = pd.to_numeric(group[self.date_col])
            amounts = group[self.amount_col].values
            if len(dates) > 0 and dates.std() > 0:
                slope, _, _, _, _ = stats.linregress(dates, amounts)
                return slope
            return 0
        
        temporal['Spending_Trend'] = self.df.groupby(self.customer_id).apply(calculate_trend)
        
        return temporal
    
    def calculate_cohort_features(self):
        """
        Calculate cohort-based features
        
        Returns:
        --------
        pd.DataFrame
            Cohort features for each customer
        """
        cohort = pd.DataFrame()
        
        # First purchase month (cohort)
        cohort['First_Purchase_Month'] = self.df.groupby(self.customer_id)[self.date_col].min().dt.to_period('M')
        
        # First purchase quarter
        cohort['First_Purchase_Quarter'] = self.df.groupby(self.customer_id)[self.date_col].min().dt.to_period('Q')
        
        # First purchase year
        cohort['First_Purchase_Year'] = self.df.groupby(self.customer_id)[self.date_col].min().dt.year
        
        # Cohort age in months
        cohort['Cohort_Age_Months'] = (
            self.reference_date.to_period('M') - cohort['First_Purchase_Month']
        ).apply(lambda x: x.n)
        
        # Purchases in first month
        first_month_df = self.df.merge(
            cohort['First_Purchase_Month'].reset_index(),
            on=self.customer_id
        )
        first_month_df['Purchase_Month'] = first_month_df[self.date_col].dt.to_period('M')
        first_month_purchases = first_month_df[
            first_month_df['Purchase_Month'] == first_month_df['First_Purchase_Month']
        ].groupby(self.customer_id).size()
        
        cohort['First_Month_Purchases'] = first_month_purchases
        cohort['First_Month_Purchases'] = cohort['First_Month_Purchases'].fillna(1)
        
        return cohort
    
    def calculate_engagement_features(self):
        """
        Calculate customer engagement features
        
        Returns:
        --------
        pd.DataFrame
            Engagement features for each customer
        """
        engagement = pd.DataFrame()
        
        # Total transactions
        engagement['Total_Transactions'] = self.df.groupby(self.customer_id).size()
        
        # Engagement score (transactions per month active)
        active_months = self.df.groupby(self.customer_id)[self.date_col].apply(
            lambda x: x.dt.to_period('M').nunique()
        )
        engagement['Engagement_Score'] = engagement['Total_Transactions'] / (active_months + 0.1)
        
        # Recent activity (last 30, 60, 90 days)
        cutoff_30 = self.reference_date - pd.Timedelta(days=30)
        cutoff_60 = self.reference_date - pd.Timedelta(days=60)
        cutoff_90 = self.reference_date - pd.Timedelta(days=90)
        
        engagement['Purchases_Last_30D'] = self.df[self.df[self.date_col] >= cutoff_30].groupby(
            self.customer_id
        ).size()
        engagement['Purchases_Last_30D'] = engagement['Purchases_Last_30D'].fillna(0)
        
        engagement['Purchases_Last_60D'] = self.df[self.df[self.date_col] >= cutoff_60].groupby(
            self.customer_id
        ).size()
        engagement['Purchases_Last_60D'] = engagement['Purchases_Last_60D'].fillna(0)
        
        engagement['Purchases_Last_90D'] = self.df[self.df[self.date_col] >= cutoff_90].groupby(
            self.customer_id
        ).size()
        engagement['Purchases_Last_90D'] = engagement['Purchases_Last_90D'].fillna(0)
        
        # Activity momentum (recent vs historical)
        total_purchases = self.df.groupby(self.customer_id).size()
        engagement['Activity_Momentum'] = engagement['Purchases_Last_90D'] / (total_purchases + 1)
        
        # Churn risk indicator (days since last purchase / average purchase interval)
        avg_interval = self.df.groupby(self.customer_id)[self.date_col].apply(
            lambda x: (x.max() - x.min()).days / max(len(x) - 1, 1)
        )
        days_since_last = (self.reference_date - self.df.groupby(self.customer_id)[self.date_col].max()).dt.days
        engagement['Churn_Risk_Score'] = days_since_last / (avg_interval + 1)
        
        return engagement
    
    def calculate_product_affinity_features(self):
        """
        Calculate product affinity and diversity features
        
        Returns:
        --------
        pd.DataFrame
            Product affinity features for each customer
        """
        affinity = pd.DataFrame()
        
        # Product diversity (unique products / total purchases)
        unique_products = self.df.groupby(self.customer_id)[self.product_col].nunique()
        total_purchases = self.df.groupby(self.customer_id).size()
        affinity['Product_Diversity'] = unique_products / total_purchases
        
        # Most frequent product
        affinity['Top_Product_Frequency'] = self.df.groupby(self.customer_id)[self.product_col].apply(
            lambda x: x.value_counts().iloc[0] if len(x) > 0 else 0
        )
        
        # Top product concentration (how much customer focuses on top product)
        affinity['Top_Product_Concentration'] = affinity['Top_Product_Frequency'] / total_purchases
        
        # Product exploration rate (new products per purchase)
        def exploration_rate(group):
            """Calculate rate at which customer tries new products"""
            if len(group) < 2:
                return 0
            seen_products = set()
            new_product_count = 0
            for product in group[self.product_col]:
                if product not in seen_products:
                    new_product_count += 1
                    seen_products.add(product)
            return new_product_count / len(group)
        
        affinity['Product_Exploration_Rate'] = self.df.groupby(self.customer_id).apply(exploration_rate)
        
        return affinity
    
    def calculate_advanced_clv_features(self):
        """
        Calculate advanced CLV features using probabilistic models
        """
        try:
            # Import lifelines for CLV modeling
            from lifelines import BetaGeoFitter, GammaGammaFitter
            
            # Prepare data for CLV calculation
            clv_data = self.df.groupby(self.customer_id).agg({
                self.date_col: ['min', 'max', 'count'],
                self.amount_col: 'sum'
            })
            
            clv_data.columns = ['first_purchase', 'last_purchase', 'frequency', 'monetary_value']
            
            # Calculate T (age of customer in days)
            clv_data['T'] = (self.reference_date - clv_data['first_purchase']).dt.days
            
            # Calculate recency in days
            clv_data['recency'] = (clv_data['last_purchase'] - clv_data['first_purchase']).dt.days
            
            # Adjust frequency (number of repeat purchases)
            clv_data['frequency'] = clv_data['frequency'] - 1
            clv_data = clv_data[clv_data['frequency'] >= 0]
            
            # Calculate average order value
            clv_data['avg_order_value'] = clv_data['monetary_value'] / (clv_data['frequency'] + 1)
            
            # Fit BG/NBD model for purchase prediction
            bgf = BetaGeoFitter(penalizer_coef=0.001)
            bgf.fit(clv_data['frequency'], clv_data['recency'], clv_data['T'])
            
            # Predict future purchases
            clv_data['predicted_purchases_30d'] = bgf.predict(
                30, clv_data['frequency'], clv_data['recency'], clv_data['T']
            )
            clv_data['predicted_purchases_90d'] = bgf.predict(
                90, clv_data['frequency'], clv_data['recency'], clv_data['T']
            )
            clv_data['predicted_purchases_365d'] = bgf.predict(
                365, clv_data['frequency'], clv_data['recency'], clv_data['T']
            )
            
            # Calculate probability of being alive
            clv_data['prob_alive'] = bgf.conditional_probability_alive(
                clv_data['frequency'], clv_data['recency'], clv_data['T']
            )
            
            # Fit Gamma-Gamma model for monetary value prediction
            customers_with_multiple_purchases = clv_data[clv_data['frequency'] > 0]
            
            if len(customers_with_multiple_purchases) > 10:
                ggf = GammaGammaFitter(penalizer_coef=0.001)
                ggf.fit(
                    customers_with_multiple_purchases['frequency'],
                    customers_with_multiple_purchases['avg_order_value']
                )
                
                # Predict CLV
                clv_data['predicted_clv_30d'] = 0
                clv_data['predicted_clv_90d'] = 0
                clv_data['predicted_clv_365d'] = 0
                
                for period, days in [('30d', 30), ('90d', 90), ('365d', 365)]:
                    clv_data.loc[customers_with_multiple_purchases.index, f'predicted_clv_{period}'] = (
                        ggf.customer_lifetime_value(
                            bgf,
                            customers_with_multiple_purchases['frequency'],
                            customers_with_multiple_purchases['recency'],
                            customers_with_multiple_purchases['T'],
                            customers_with_multiple_purchases['avg_order_value'],
                            time=days,
                            freq='D'
                        )
                    )
            
            return clv_data[['predicted_purchases_30d', 'predicted_purchases_90d', 'predicted_purchases_365d',
                           'prob_alive', 'predicted_clv_30d', 'predicted_clv_90d', 'predicted_clv_365d']]
        
        except ImportError:
            print("lifelines library not available. Install with: pip install lifelines")
            # Return empty DataFrame with expected columns
            empty_df = pd.DataFrame(index=self.df[self.customer_id].unique())
            for col in ['predicted_purchases_30d', 'predicted_purchases_90d', 'predicted_purchases_365d',
                       'prob_alive', 'predicted_clv_30d', 'predicted_clv_90d', 'predicted_clv_365d']:
                empty_df[col] = 0
            return empty_df
        except Exception as e:
            print(f"Error in advanced CLV calculation: {e}")
            # Return empty DataFrame with expected columns
            empty_df = pd.DataFrame(index=self.df[self.customer_id].unique())
            for col in ['predicted_purchases_30d', 'predicted_purchases_90d', 'predicted_purchases_365d',
                       'prob_alive', 'predicted_clv_30d', 'predicted_clv_90d', 'predicted_clv_365d']:
                empty_df[col] = 0
            return empty_df
    
    def calculate_churn_prediction_features(self):
        """
        Calculate advanced churn prediction features
        """
        churn_features = pd.DataFrame()
        
        # Days since last purchase
        churn_features['Days_Since_Last_Purchase'] = (
            self.reference_date - self.df.groupby(self.customer_id)[self.date_col].max()
        ).dt.days
        
        # Expected purchase interval
        def calculate_purchase_interval(group):
            if len(group) <= 1:
                return 365  # Default to 1 year for single purchases
            dates = sorted(group)
            intervals = [(dates[i+1] - dates[i]).days for i in range(len(dates)-1)]
            return np.mean(intervals)
        
        churn_features['Expected_Purchase_Interval'] = self.df.groupby(self.customer_id)[self.date_col].apply(
            calculate_purchase_interval
        )
        
        # Churn risk score (days overdue / expected interval)
        churn_features['Churn_Risk_Score'] = (
            churn_features['Days_Since_Last_Purchase'] / churn_features['Expected_Purchase_Interval']
        ).fillna(0)
        
        # Trend in purchase frequency
        def calculate_frequency_trend(group):
            if len(group) < 4:
                return 0
            
            # Group by month and count purchases
            monthly_counts = group.dt.to_period('M').value_counts().sort_index()
            if len(monthly_counts) < 3:
                return 0
            
            # Calculate trend slope
            x = np.arange(len(monthly_counts))
            y = monthly_counts.values
            if np.var(x) > 0:
                slope, _, _, _, _ = stats.linregress(x, y)
                return slope
            return 0
        
        churn_features['Purchase_Frequency_Trend'] = self.df.groupby(self.customer_id)[self.date_col].apply(
            calculate_frequency_trend
        )
        
        # Trend in spending
        def calculate_spending_trend(group):
            if len(group) < 4:
                return 0
            
            # Group by month and sum spending
            monthly_spending = group.groupby(group[self.date_col].dt.to_period('M'))[self.amount_col].sum()
            if len(monthly_spending) < 3:
                return 0
            
            # Calculate trend slope
            x = np.arange(len(monthly_spending))
            y = monthly_spending.values
            if np.var(x) > 0:
                slope, _, _, _, _ = stats.linregress(x, y)
                return slope
            return 0
        
        customer_groups = self.df.groupby(self.customer_id)
        churn_features['Spending_Trend'] = customer_groups.apply(calculate_spending_trend)
        
        # Engagement decline indicator
        recent_30d = self.reference_date - pd.Timedelta(days=30)
        recent_90d = self.reference_date - pd.Timedelta(days=90)
        
        purchases_30d = self.df[self.df[self.date_col] >= recent_30d].groupby(self.customer_id).size()
        purchases_90d = self.df[self.df[self.date_col] >= recent_90d].groupby(self.customer_id).size()
        
        churn_features['Purchases_30d'] = purchases_30d.reindex(churn_features.index, fill_value=0)
        churn_features['Purchases_90d'] = purchases_90d.reindex(churn_features.index, fill_value=0)
        
        # Engagement momentum (recent activity vs historical average)
        total_purchases = self.df.groupby(self.customer_id).size()
        customer_lifetime_days = (
            self.df.groupby(self.customer_id)[self.date_col].max() - 
            self.df.groupby(self.customer_id)[self.date_col].min()
        ).dt.days + 1
        
        historical_rate = total_purchases / (customer_lifetime_days / 30)  # purchases per month
        current_rate = churn_features['Purchases_30d']
        
        churn_features['Engagement_Momentum'] = (current_rate / (historical_rate + 0.1)).fillna(0)
        
        return churn_features
    
    def calculate_product_recommendation_features(self):
        """
        Calculate features for product recommendation system
        """
        # Create customer-product matrix
        customer_product = self.df.pivot_table(
            index=self.customer_id, 
            columns=self.product_col, 
            values=self.quantity_col,
            aggfunc='sum',
            fill_value=0
        )
        
        rec_features = pd.DataFrame(index=customer_product.index)
        
        # Calculate similarity with other customers
        similarity_matrix = cosine_similarity(customer_product.fillna(0))
        
        # Average similarity with top 5 most similar customers
        for i, customer_id in enumerate(customer_product.index):
            similarities = similarity_matrix[i]
            # Exclude self-similarity
            similarities[i] = -1
            top_5_similarities = np.partition(similarities, -5)[-5:]
            rec_features.loc[customer_id, 'Avg_Customer_Similarity'] = np.mean(top_5_similarities[top_5_similarities > 0])
        
        rec_features['Avg_Customer_Similarity'] = rec_features['Avg_Customer_Similarity'].fillna(0)
        
        # Product portfolio diversity (entropy)
        def calculate_entropy(row):
            # Normalize to probabilities
            probs = row / row.sum() if row.sum() > 0 else row
            probs = probs[probs > 0]  # Remove zeros for log calculation
            if len(probs) <= 1:
                return 0
            return -np.sum(probs * np.log2(probs))
        
        rec_features['Product_Portfolio_Entropy'] = customer_product.apply(calculate_entropy, axis=1)
        
        # Cross-selling potential (products purchased by similar customers)
        rec_features['Cross_Sell_Potential'] = 0
        for i, customer_id in enumerate(customer_product.index):
            customer_products = customer_product.iloc[i] > 0
            similarities = similarity_matrix[i]
            similarities[i] = -1  # Exclude self
            
            # Find top 10 similar customers
            top_similar_indices = np.argsort(similarities)[-10:]
            similar_customers = customer_product.iloc[top_similar_indices]
            
            # Products bought by similar customers but not by this customer
            potential_products = similar_customers.mean(axis=0)
            potential_products = potential_products[~customer_products]
            
            rec_features.loc[customer_id, 'Cross_Sell_Potential'] = potential_products.sum()
        
        return rec_features
    
    def create_all_features(self, include_advanced_clv=True, include_churn_features=True, include_recommendation_features=True):
        """
        Create all features and combine them
        
        Parameters:
        -----------
        include_advanced_clv : bool
            Whether to include advanced CLV features (requires lifelines)
        include_churn_features : bool
            Whether to include churn prediction features
        include_recommendation_features : bool
            Whether to include product recommendation features
        
        Returns:
        --------
        pd.DataFrame
            Complete feature set for each customer
        """
        print("Starting comprehensive feature engineering...")
        
        print("Calculating basic CLV features...")
        clv_features = self.calculate_clv_features()
        
        print("Calculating behavioral features...")
        behavioral_features = self.calculate_behavioral_features()
        
        print("Calculating temporal features...")
        temporal_features = self.calculate_temporal_features()
        
        print("Calculating cohort features...")
        cohort_features = self.calculate_cohort_features()
        
        print("Calculating engagement features...")
        engagement_features = self.calculate_engagement_features()
        
        print("Calculating product affinity features...")
        affinity_features = self.calculate_product_affinity_features()
        
        feature_sets = [
            clv_features,
            behavioral_features,
            temporal_features,
            cohort_features,
            engagement_features,
            affinity_features
        ]
        
        # Advanced CLV features
        if include_advanced_clv:
            print("Calculating advanced CLV features...")
            try:
                advanced_clv_features = self.calculate_advanced_clv_features()
                feature_sets.append(advanced_clv_features)
            except Exception as e:
                print(f"Skipping advanced CLV features: {e}")
        
        # Churn prediction features
        if include_churn_features:
            print("Calculating churn prediction features...")
            try:
                churn_features = self.calculate_churn_prediction_features()
                feature_sets.append(churn_features)
            except Exception as e:
                print(f"Skipping churn features: {e}")
        
        # Product recommendation features
        if include_recommendation_features:
            print("Calculating product recommendation features...")
            try:
                rec_features = self.calculate_product_recommendation_features()
                feature_sets.append(rec_features)
            except Exception as e:
                print(f"Skipping recommendation features: {e}")
        
        # Combine all features
        all_features = pd.concat(feature_sets, axis=1)
        
        # Handle any remaining NaN values
        all_features = all_features.fillna(0)
        
        # Convert period columns to strings for compatibility
        for col in all_features.columns:
            if all_features[col].dtype == 'period[M]' or all_features[col].dtype == 'period[Q]':
                all_features[col] = all_features[col].astype(str)
        
        # Remove infinite values
        all_features = all_features.replace([np.inf, -np.inf], 0)
        
        print(f"\nUltra-advanced feature engineering complete!")
        print(f"Created {len(all_features.columns)} features for {len(all_features)} customers")
        
        return all_features


def calculate_customer_journey_metrics(df, customer_id='CustomerID', date_col='InvoiceDate'):
    """
    Calculate customer journey metrics including stage identification
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
    customer_id : str
        Customer ID column name
    date_col : str
        Date column name
    
    Returns:
    --------
    pd.DataFrame
        Customer journey metrics
    """
    reference_date = df[date_col].max() + pd.Timedelta(days=1)
    
    journey = pd.DataFrame()
    journey[customer_id] = df[customer_id].unique()
    journey = journey.set_index(customer_id)
    
    # Calculate recency, frequency
    recency = (reference_date - df.groupby(customer_id)[date_col].max()).dt.days
    frequency = df.groupby(customer_id).size()
    
    # Determine customer lifecycle stage
    def determine_stage(r, f):
        """Determine customer lifecycle stage"""
        if f == 1:
            if r <= 30:
                return 'New'
            else:
                return 'One-Time'
        elif f >= 2 and f <= 5:
            if r <= 30:
                return 'Active'
            elif r <= 90:
                return 'Cooling'
            else:
                return 'At Risk'
        else:  # f > 5
            if r <= 30:
                return 'Loyal'
            elif r <= 60:
                return 'Slipping'
            else:
                return 'Lost'
    
    journey['Lifecycle_Stage'] = [determine_stage(r, f) for r, f in zip(recency, frequency)]
    
    # Days in each stage (simplified)
    journey['Days_In_Stage'] = recency.values
    
    # Predicted next stage
    def predict_next_stage(stage):
        """Predict likely next stage based on current stage"""
        transitions = {
            'New': 'Active',
            'Active': 'Loyal',
            'Loyal': 'Loyal',
            'Cooling': 'At Risk',
            'At Risk': 'Lost',
            'Slipping': 'Lost',
            'One-Time': 'Lost',
            'Lost': 'Lost'
        }
        return transitions.get(stage, 'Unknown')
    
    journey['Predicted_Next_Stage'] = journey['Lifecycle_Stage'].apply(predict_next_stage)
    
    return journey.reset_index()


def create_rfm_enhanced_features(rfm_df):
    """
    Create enhanced features from RFM data
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with Recency, Frequency, Monetary
    
    Returns:
    --------
    pd.DataFrame
        Enhanced RFM features
    """
    enhanced = rfm_df.copy()
    
    # Log transformations for skewed distributions
    enhanced['Recency_Log'] = np.log1p(enhanced['Recency'])
    enhanced['Frequency_Log'] = np.log1p(enhanced['Frequency'])
    enhanced['Monetary_Log'] = np.log1p(enhanced['Monetary'])
    
    # Polynomial features
    enhanced['RF_Interaction'] = enhanced['Recency'] * enhanced['Frequency']
    enhanced['FM_Interaction'] = enhanced['Frequency'] * enhanced['Monetary']
    enhanced['RM_Interaction'] = enhanced['Recency'] * enhanced['Monetary']
    enhanced['RFM_Interaction'] = enhanced['Recency'] * enhanced['Frequency'] * enhanced['Monetary']
    
    # Ratio features
    enhanced['Monetary_Per_Frequency'] = enhanced['Monetary'] / (enhanced['Frequency'] + 1)
    enhanced['Frequency_Per_Day'] = enhanced['Frequency'] / (enhanced['Recency'] + 1)
    
    # Quartile features
    enhanced['Recency_Quartile'] = pd.qcut(enhanced['Recency'], q=4, labels=False, duplicates='drop')
    enhanced['Frequency_Quartile'] = pd.qcut(enhanced['Frequency'], q=4, labels=False, duplicates='drop')
    enhanced['Monetary_Quartile'] = pd.qcut(enhanced['Monetary'], q=4, labels=False, duplicates='drop')
    
    return enhanced
