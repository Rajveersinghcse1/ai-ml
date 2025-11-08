"""
Ultra-Advanced Analytics Module
Comprehensive predictive analytics and machine learning for customer segmentation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import pickle
import json

# Data preprocessing and feature engineering
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Machine Learning Models
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Advanced ML libraries (with fallbacks)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Neural Networks
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# Model evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix, roc_curve
)

# Statistics and survival analysis
from scipy import stats
from scipy.stats import chi2_contingency
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False

# Clustering
from sklearn.cluster import KMeans, DBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Model interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class ChurnPredictionModel:
    """
    Advanced churn prediction with multiple algorithms and ensemble methods
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.is_fitted = False
        
    def prepare_churn_labels(self, df: pd.DataFrame, customer_id: str = 'CustomerID', 
                           date_col: str = 'InvoiceDate', threshold_days: int = 90) -> pd.DataFrame:
        """
        Create churn labels based on recency threshold
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        customer_id : str
            Customer ID column
        date_col : str
            Date column
        threshold_days : int
            Days threshold for churn definition
        
        Returns:
        --------
        pd.DataFrame
            Customer features with churn labels
        """
        reference_date = df[date_col].max()
        
        # Calculate days since last purchase
        last_purchase = df.groupby(customer_id)[date_col].max()
        days_since_last = (reference_date - last_purchase).dt.days
        
        # Create churn labels
        churn_labels = (days_since_last > threshold_days).astype(int)
        
        return pd.DataFrame({
            customer_id: churn_labels.index,
            'is_churned': churn_labels.values,
            'days_since_last_purchase': days_since_last.values
        }).set_index(customer_id)
    
    def build_models(self):
        """Build ensemble of churn prediction models"""
        
        # Base models
        self.models = {
            'logistic_regression': LogisticRegression(random_state=self.random_state),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(random_state=self.random_state),
            'svm': SVC(probability=True, random_state=self.random_state),
            'knn': KNeighborsClassifier(),
            'naive_bayes': GaussianNB()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state, eval_metric='logloss'
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state, verbose=-1
            )
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_name: str):
        """Optimize hyperparameters for specific model"""
        
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Using default parameters.")
            return
        
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                model = RandomForestClassifier(random_state=self.random_state, **params)
            
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
                }
                model = xgb.XGBClassifier(
                    random_state=self.random_state, eval_metric='logloss', **params
                )
            else:
                return 0  # Skip optimization for other models
            
            # Cross-validation score
            scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            return scores.mean()
        
        # Optimize
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)
        
        # Update model with best parameters
        if model_name in self.models:
            if model_name == 'random_forest':
                self.models[model_name].set_params(**study.best_params)
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                self.models[model_name].set_params(**study.best_params)
    
    def train(self, X: pd.DataFrame, y: pd.Series, optimize_hyperparams: bool = True,
              feature_selection: bool = True, test_size: float = 0.2):
        """
        Train churn prediction models
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable (churn labels)
        optimize_hyperparams : bool
            Whether to optimize hyperparameters
        feature_selection : bool
            Whether to perform feature selection
        test_size : float
            Test set size for validation
        """
        
        print("Training churn prediction models...")
        
        # Build models
        self.build_models()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        if feature_selection:
            self.feature_selector = SelectKBest(
                f_classif, k=min(50, X_train.shape[1])  # Select top 50 features or all if fewer
            )
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            X_test_scaled = self.feature_selector.transform(X_test_scaled)
        
        # Hyperparameter optimization
        if optimize_hyperparams:
            print("Optimizing hyperparameters...")
            for model_name in ['random_forest', 'xgboost']:
                if model_name in self.models:
                    self.optimize_hyperparameters(
                        pd.DataFrame(X_train_scaled), y_train, model_name
                    )
        
        # Train all models
        self.model_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
                self.model_scores[name] = auc_score
                print(f"{name} AUC: {auc_score:.4f}")
            except Exception as e:
                print(f"Error training {name}: {e}")
                del self.models[name]
        
        # Create ensemble model
        if len(self.models) > 1:
            ensemble_models = [(name, model) for name, model in self.models.items()]
            self.ensemble_model = VotingClassifier(
                estimators=ensemble_models, voting='soft'
            )
            self.ensemble_model.fit(X_train_scaled, y_train)
            
            # Evaluate ensemble
            ensemble_pred_proba = self.ensemble_model.predict_proba(X_test_scaled)[:, 1]
            ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)
            self.model_scores['ensemble'] = ensemble_auc
            print(f"Ensemble AUC: {ensemble_auc:.4f}")
        
        self.is_fitted = True
        
        # Store feature names for prediction
        self.feature_names = X.columns.tolist()
        
        print("Churn prediction training completed!")
        return self.model_scores
    
    def predict_churn_probability(self, X: pd.DataFrame, use_ensemble: bool = True) -> np.ndarray:
        """
        Predict churn probability for customers
        
        Parameters:
        -----------
        X : pd.DataFrame
            Customer features
        use_ensemble : bool
            Whether to use ensemble model
        
        Returns:
        --------
        np.ndarray
            Churn probabilities
        """
        
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train() first.")
        
        # Ensure features match training data
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply feature selection if used during training
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        if use_ensemble and self.ensemble_model is not None:
            return self.ensemble_model.predict_proba(X_scaled)[:, 1]
        else:
            # Use best individual model
            best_model = max(self.models.items(), key=lambda x: self.model_scores.get(x[0], 0))[1]
            return best_model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """Get feature importance from trained models"""
        
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        importances = {}
        
        if model_name and model_name in self.models:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                importances[model_name] = model.feature_importances_
        else:
            # Get importance from all models that have it
            for name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importances[name] = model.feature_importances_
        
        if not importances:
            print("No feature importances available.")
            return pd.DataFrame()
        
        # Get feature names (after selection if applied)
        if self.feature_selector is not None:
            selected_features = np.array(self.feature_names)[self.feature_selector.get_support()]
        else:
            selected_features = self.feature_names
        
        importance_df = pd.DataFrame(importances, index=selected_features)
        
        if len(importances) > 1:
            importance_df['mean_importance'] = importance_df.mean(axis=1)
            importance_df = importance_df.sort_values('mean_importance', ascending=False)
        
        return importance_df


class CLVPredictionModel:
    """
    Advanced Customer Lifetime Value prediction model
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def prepare_clv_features(self, df: pd.DataFrame, customer_id: str = 'CustomerID',
                           date_col: str = 'InvoiceDate', amount_col: str = 'TotalAmount',
                           prediction_months: int = 12) -> pd.DataFrame:
        """
        Prepare features for CLV prediction
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        prediction_months : int
            Number of months to predict CLV for
        """
        
        reference_date = df[date_col].max()
        
        # Calculate historical metrics
        customer_metrics = df.groupby(customer_id).agg({
            amount_col: ['sum', 'mean', 'std', 'count'],
            date_col: ['min', 'max']
        })
        
        customer_metrics.columns = ['total_spent', 'avg_order_value', 'spending_std', 
                                   'purchase_frequency', 'first_purchase', 'last_purchase']
        
        # Calculate derived features
        customer_metrics['customer_age_days'] = (
            reference_date - customer_metrics['first_purchase']
        ).dt.days
        
        customer_metrics['recency_days'] = (
            reference_date - customer_metrics['last_purchase']
        ).dt.days
        
        customer_metrics['purchase_rate_per_day'] = (
            customer_metrics['purchase_frequency'] / customer_metrics['customer_age_days']
        ).fillna(0)
        
        customer_metrics['spending_consistency'] = (
            customer_metrics['spending_std'] / customer_metrics['avg_order_value']
        ).fillna(0)
        
        # Seasonal features
        df_with_season = df.copy()
        df_with_season['month'] = df_with_season[date_col].dt.month
        df_with_season['quarter'] = df_with_season[date_col].dt.quarter
        
        seasonal_spending = df_with_season.groupby([customer_id, 'quarter'])[amount_col].sum().unstack(fill_value=0)
        seasonal_spending.columns = [f'Q{col}_spending' for col in seasonal_spending.columns]
        
        # Combine features
        clv_features = customer_metrics.join(seasonal_spending, how='left').fillna(0)
        
        return clv_features
    
    def build_models(self):
        """Build ensemble of CLV prediction models"""
        
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.random_state),
            'svr': SVR()
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBRegressor(random_state=self.random_state)
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMRegressor(
                random_state=self.random_state, verbose=-1
            )
    
    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        Train CLV prediction models
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target CLV values
        """
        
        print("Training CLV prediction models...")
        
        # Build models
        self.build_models()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Feature scaling
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train all models
        self.model_scores = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                self.model_scores[name] = {
                    'r2': r2,
                    'rmse': rmse
                }
                
                print(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.2f}")
                
            except Exception as e:
                print(f"Error training {name}: {e}")
                del self.models[name]
        
        self.is_fitted = True
        self.feature_names = X.columns.tolist()
        
        print("CLV prediction training completed!")
        return self.model_scores
    
    def predict_clv(self, X: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Predict CLV for customers
        
        Parameters:
        -----------
        X : pd.DataFrame
            Customer features
        model_name : str
            Specific model to use (if None, uses best model)
        """
        
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call train() first.")
        
        # Ensure features match training data
        X = X[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        if model_name and model_name in self.models:
            model = self.models[model_name]
        else:
            # Use best model based on R²
            best_model = max(
                self.models.items(), 
                key=lambda x: self.model_scores.get(x[0], {}).get('r2', -np.inf)
            )[1]
            model = best_model
        
        return model.predict(X_scaled)


class NextPurchasePrediction:
    """
    Predict when customers will make their next purchase
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.survival_model = None
        self.regression_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def prepare_survival_data(self, df: pd.DataFrame, customer_id: str = 'CustomerID',
                            date_col: str = 'InvoiceDate') -> pd.DataFrame:
        """
        Prepare data for survival analysis (time to next purchase)
        """
        
        # Calculate time between consecutive purchases for each customer
        survival_data = []
        
        for customer in df[customer_id].unique():
            customer_data = df[df[customer_id] == customer].copy()
            customer_data = customer_data.sort_values(date_col)
            
            if len(customer_data) > 1:
                # Calculate time differences between purchases
                time_diffs = customer_data[date_col].diff().dt.days.dropna()
                
                for i, time_diff in enumerate(time_diffs):
                    # Create survival record
                    survival_data.append({
                        customer_id: customer,
                        'duration': time_diff,
                        'event': 1,  # Purchase occurred (not censored)
                        'purchase_number': i + 2,  # Second purchase onwards
                        'previous_amount': customer_data.iloc[i]['TotalAmount'],
                        'days_since_first': (
                            customer_data.iloc[i+1][date_col] - customer_data.iloc[0][date_col]
                        ).days
                    })
        
        return pd.DataFrame(survival_data)
    
    def train_survival_model(self, survival_data: pd.DataFrame, features: List[str]):
        """
        Train survival model for next purchase prediction
        """
        
        if not LIFELINES_AVAILABLE:
            print("lifelines library not available. Skipping survival analysis.")
            return False
        
        print("Training survival model for next purchase prediction...")
        
        try:
            # Prepare data for Cox Proportional Hazards model
            model_data = survival_data[['duration', 'event'] + features].copy()
            model_data = model_data.dropna()
            
            if len(model_data) == 0:
                print("No valid data for survival model.")
                return False
            
            # Fit Cox model
            self.survival_model = CoxPHFitter()
            self.survival_model.fit(model_data, duration_col='duration', event_col='event')
            
            # Calculate concordance index
            concordance = concordance_index(
                model_data['duration'],
                -self.survival_model.predict_partial_hazard(model_data[features]),
                model_data['event']
            )
            
            print(f"Survival model concordance index: {concordance:.4f}")
            return True
            
        except Exception as e:
            print(f"Error training survival model: {e}")
            return False
    
    def train_regression_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train regression model as alternative to survival analysis
        """
        
        print("Training regression model for next purchase prediction...")
        
        # Use XGBoost if available, otherwise Random Forest
        if XGBOOST_AVAILABLE:
            self.regression_model = xgb.XGBRegressor(random_state=self.random_state)
        else:
            self.regression_model = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.regression_model.fit(X_scaled, y)
        
        # Evaluate
        y_pred = self.regression_model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"Next purchase regression model - R²: {r2:.4f}, RMSE: {rmse:.2f} days")
        
        self.is_fitted = True
        return {'r2': r2, 'rmse': rmse}
    
    def predict_next_purchase(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict days until next purchase
        """
        
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        if self.regression_model is not None:
            X_scaled = self.scaler.transform(X)
            return self.regression_model.predict(X_scaled)
        else:
            raise ValueError("No trained model available.")


class SentimentAnalyzer:
    """
    Sentiment analysis for customer feedback and reviews
    """
    
    def __init__(self):
        self.vectorizer = None
        self.sentiment_model = None
        self.is_fitted = False
    
    def prepare_text_features(self, texts: List[str]) -> pd.DataFrame:
        """
        Prepare text features for sentiment analysis
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # Initialize vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
            
            # Transform texts
            text_features = self.vectorizer.fit_transform(texts)
            
            # Convert to DataFrame
            feature_names = self.vectorizer.get_feature_names_out()
            return pd.DataFrame(text_features.toarray(), columns=feature_names)
            
        except Exception as e:
            print(f"Error preparing text features: {e}")
            return pd.DataFrame()
    
    def train_sentiment_model(self, texts: List[str], sentiments: List[int]):
        """
        Train sentiment classification model
        
        Parameters:
        -----------
        texts : List[str]
            List of text reviews/feedback
        sentiments : List[int]
            Sentiment labels (0: negative, 1: positive)
        """
        
        print("Training sentiment analysis model...")
        
        # Prepare text features
        X = self.prepare_text_features(texts)
        y = np.array(sentiments)
        
        if len(X) == 0:
            print("No text features available.")
            return False
        
        # Train sentiment classifier
        self.sentiment_model = LogisticRegression(random_state=42)
        self.sentiment_model.fit(X, y)
        
        # Evaluate
        y_pred = self.sentiment_model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        print(f"Sentiment model accuracy: {accuracy:.4f}")
        
        self.is_fitted = True
        return accuracy
    
    def predict_sentiment(self, texts: List[str]) -> np.ndarray:
        """
        Predict sentiment for new texts
        """
        
        if not self.is_fitted:
            raise ValueError("Model not fitted yet.")
        
        # Transform texts using fitted vectorizer
        text_features = self.vectorizer.transform(texts)
        
        # Predict sentiment probabilities
        return self.sentiment_model.predict_proba(text_features.toarray())[:, 1]


class AdvancedAnalyticsEngine:
    """
    Main engine combining all advanced analytics capabilities
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.churn_model = ChurnPredictionModel(random_state)
        self.clv_model = CLVPredictionModel(random_state)
        self.next_purchase_model = NextPurchasePrediction(random_state)
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def run_comprehensive_analysis(self, df: pd.DataFrame, features: pd.DataFrame,
                                 customer_id: str = 'CustomerID',
                                 date_col: str = 'InvoiceDate',
                                 amount_col: str = 'TotalAmount'):
        """
        Run comprehensive advanced analytics pipeline
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        features : pd.DataFrame
            Customer features
        """
        
        results = {}
        
        print("\n" + "="*60)
        print("ULTRA-ADVANCED ANALYTICS PIPELINE")
        print("="*60)
        
        # 1. Churn Prediction
        print("\n1. CHURN PREDICTION ANALYSIS")
        print("-" * 30)
        
        try:
            churn_labels = self.churn_model.prepare_churn_labels(df, customer_id, date_col)
            
            # Align features with churn labels
            common_customers = features.index.intersection(churn_labels.index)
            X_churn = features.loc[common_customers]
            y_churn = churn_labels.loc[common_customers, 'is_churned']
            
            if len(X_churn) > 50:  # Minimum samples for training
                churn_scores = self.churn_model.train(X_churn, y_churn)
                
                # Predict churn probabilities
                churn_probs = self.churn_model.predict_churn_probability(X_churn)
                
                results['churn_analysis'] = {
                    'model_scores': churn_scores,
                    'churn_probabilities': pd.Series(churn_probs, index=common_customers),
                    'feature_importance': self.churn_model.get_feature_importance()
                }
                
                print(f"✓ Churn models trained on {len(X_churn)} customers")
                print(f"✓ Best model AUC: {max(churn_scores.values()):.4f}")
                
            else:
                print("✗ Insufficient data for churn prediction")
                results['churn_analysis'] = None
                
        except Exception as e:
            print(f"✗ Churn prediction failed: {e}")
            results['churn_analysis'] = None
        
        # 2. CLV Prediction
        print("\n2. CUSTOMER LIFETIME VALUE PREDICTION")
        print("-" * 40)
        
        try:
            clv_features = self.clv_model.prepare_clv_features(df, customer_id, date_col, amount_col)
            
            # Use future CLV as target (simplified approach)
            # In real scenario, you would have actual future CLV data
            current_clv = df.groupby(customer_id)[amount_col].sum()
            
            # Align features
            common_customers = clv_features.index.intersection(current_clv.index)
            X_clv = clv_features.loc[common_customers]
            y_clv = current_clv.loc[common_customers]
            
            if len(X_clv) > 50:
                clv_scores = self.clv_model.train(X_clv, y_clv)
                
                # Predict CLV
                clv_predictions = self.clv_model.predict_clv(X_clv)
                
                results['clv_analysis'] = {
                    'model_scores': clv_scores,
                    'clv_predictions': pd.Series(clv_predictions, index=common_customers),
                    'clv_features': X_clv
                }
                
                best_r2 = max([scores['r2'] for scores in clv_scores.values()])
                print(f"✓ CLV models trained on {len(X_clv)} customers")
                print(f"✓ Best model R²: {best_r2:.4f}")
                
            else:
                print("✗ Insufficient data for CLV prediction")
                results['clv_analysis'] = None
                
        except Exception as e:
            print(f"✗ CLV prediction failed: {e}")
            results['clv_analysis'] = None
        
        # 3. Next Purchase Prediction
        print("\n3. NEXT PURCHASE PREDICTION")
        print("-" * 30)
        
        try:
            survival_data = self.next_purchase_model.prepare_survival_data(df, customer_id, date_col)
            
            if len(survival_data) > 100:
                # Train survival model if possible
                survival_features = ['purchase_number', 'previous_amount', 'days_since_first']
                available_features = [f for f in survival_features if f in survival_data.columns]
                
                if LIFELINES_AVAILABLE and len(available_features) > 0:
                    survival_success = self.next_purchase_model.train_survival_model(
                        survival_data, available_features
                    )
                else:
                    survival_success = False
                
                # Train regression model as alternative
                if len(available_features) > 0:
                    X_next = survival_data[available_features]
                    y_next = survival_data['duration']
                    
                    next_purchase_scores = self.next_purchase_model.train_regression_model(X_next, y_next)
                    
                    results['next_purchase_analysis'] = {
                        'model_scores': next_purchase_scores,
                        'survival_model_available': survival_success
                    }
                    
                    print(f"✓ Next purchase model trained on {len(survival_data)} purchase intervals")
                    print(f"✓ Model R²: {next_purchase_scores['r2']:.4f}")
                    
                else:
                    print("✗ No valid features for next purchase prediction")
                    results['next_purchase_analysis'] = None
                    
            else:
                print("✗ Insufficient data for next purchase prediction")
                results['next_purchase_analysis'] = None
                
        except Exception as e:
            print(f"✗ Next purchase prediction failed: {e}")
            results['next_purchase_analysis'] = None
        
        # 4. Generate Summary Report
        print("\n4. ANALYTICS SUMMARY")
        print("-" * 20)
        
        summary = self._generate_analytics_summary(results)
        results['summary'] = summary
        
        for line in summary:
            print(line)
        
        print("\n" + "="*60)
        print("ANALYTICS PIPELINE COMPLETED")
        print("="*60)
        
        return results
    
    def _generate_analytics_summary(self, results: Dict) -> List[str]:
        """Generate summary of analytics results"""
        
        summary = []
        
        # Churn Analysis Summary
        if results.get('churn_analysis'):
            churn_data = results['churn_analysis']
            best_auc = max(churn_data['model_scores'].values())
            high_risk_customers = (churn_data['churn_probabilities'] > 0.7).sum()
            
            summary.extend([
                f"🎯 CHURN ANALYSIS:",
                f"   • Best model AUC: {best_auc:.3f}",
                f"   • High-risk customers: {high_risk_customers}",
                f"   • Average churn probability: {churn_data['churn_probabilities'].mean():.3f}"
            ])
        
        # CLV Analysis Summary
        if results.get('clv_analysis'):
            clv_data = results['clv_analysis']
            best_r2 = max([scores['r2'] for scores in clv_data['model_scores'].values()])
            avg_clv = clv_data['clv_predictions'].mean()
            
            summary.extend([
                f"💰 CLV ANALYSIS:",
                f"   • Best model R²: {best_r2:.3f}",
                f"   • Average predicted CLV: ${avg_clv:.2f}",
                f"   • CLV range: ${clv_data['clv_predictions'].min():.2f} - ${clv_data['clv_predictions'].max():.2f}"
            ])
        
        # Next Purchase Summary
        if results.get('next_purchase_analysis'):
            next_data = results['next_purchase_analysis']
            
            summary.extend([
                f"📅 NEXT PURCHASE ANALYSIS:",
                f"   • Model R²: {next_data['model_scores']['r2']:.3f}",
                f"   • RMSE: {next_data['model_scores']['rmse']:.1f} days",
                f"   • Survival model: {'Available' if next_data['survival_model_available'] else 'Not available'}"
            ])
        
        if not summary:
            summary.append("❌ No successful analytics models trained")
        
        return summary
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        
        models_data = {
            'churn_model': self.churn_model if self.churn_model.is_fitted else None,
            'clv_model': self.clv_model if self.clv_model.is_fitted else None,
            'next_purchase_model': self.next_purchase_model if self.next_purchase_model.is_fitted else None,
            'sentiment_analyzer': self.sentiment_analyzer if self.sentiment_analyzer.is_fitted else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(models_data, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        
        with open(filepath, 'rb') as f:
            models_data = pickle.load(f)
        
        if models_data['churn_model']:
            self.churn_model = models_data['churn_model']
        if models_data['clv_model']:
            self.clv_model = models_data['clv_model']
        if models_data['next_purchase_model']:
            self.next_purchase_model = models_data['next_purchase_model']
        if models_data['sentiment_analyzer']:
            self.sentiment_analyzer = models_data['sentiment_analyzer']
        
        print(f"Models loaded from {filepath}")


def generate_predictive_insights(results: Dict, customer_features: pd.DataFrame) -> Dict:
    """
    Generate actionable insights from predictive analytics results
    """
    
    insights = {}
    
    # Churn insights
    if results.get('churn_analysis'):
        churn_probs = results['churn_analysis']['churn_probabilities']
        
        # Identify high-risk customers
        high_risk = churn_probs[churn_probs > 0.7].index.tolist()
        medium_risk = churn_probs[(churn_probs > 0.4) & (churn_probs <= 0.7)].index.tolist()
        
        insights['churn_insights'] = {
            'high_risk_customers': high_risk,
            'medium_risk_customers': medium_risk,
            'retention_priority': high_risk[:20],  # Top 20 priority customers
            'avg_risk_score': churn_probs.mean()
        }
    
    # CLV insights
    if results.get('clv_analysis'):
        clv_predictions = results['clv_analysis']['clv_predictions']
        
        # Segment by CLV
        clv_quantiles = clv_predictions.quantile([0.2, 0.4, 0.6, 0.8])
        
        insights['clv_insights'] = {
            'high_value_customers': clv_predictions[clv_predictions >= clv_quantiles[0.8]].index.tolist(),
            'medium_value_customers': clv_predictions[
                (clv_predictions >= clv_quantiles[0.4]) & (clv_predictions < clv_quantiles[0.8])
            ].index.tolist(),
            'investment_priorities': clv_predictions.nlargest(20).index.tolist(),
            'avg_predicted_clv': clv_predictions.mean()
        }
    
    # Combined insights
    if 'churn_insights' in insights and 'clv_insights' in insights:
        high_value_at_risk = set(insights['clv_insights']['high_value_customers']).intersection(
            set(insights['churn_insights']['high_risk_customers'])
        )
        
        insights['strategic_insights'] = {
            'high_value_at_risk': list(high_value_at_risk),
            'retention_roi_priorities': list(high_value_at_risk)[:10]
        }
    
    return insights


# Example usage and testing functions
def run_analytics_demo(df: pd.DataFrame, features: pd.DataFrame):
    """
    Demonstration of the advanced analytics capabilities
    """
    
    print("🚀 RUNNING ULTRA-ADVANCED ANALYTICS DEMO")
    print("="*50)
    
    # Initialize analytics engine
    analytics_engine = AdvancedAnalyticsEngine(random_state=42)
    
    # Run comprehensive analysis
    results = analytics_engine.run_comprehensive_analysis(df, features)
    
    # Generate insights
    insights = generate_predictive_insights(results, features)
    
    # Print insights
    print("\n📊 STRATEGIC INSIGHTS")
    print("-"*20)
    
    for category, data in insights.items():
        print(f"\n{category.upper()}:")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"  • {key}: {len(value)} customers")
            else:
                print(f"  • {key}: {value}")
    
    return results, insights
