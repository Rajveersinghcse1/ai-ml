"""
Ultra-Advanced Model Evaluation and Validation Module
Comprehensive metrics, cross-validation, and A/B testing frameworks for customer segmentation models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import warnings
from collections import defaultdict
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

# Core ML libraries
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold, 
    TimeSeriesSplit, ParameterGrid, GridSearchCV, RandomizedSearchCV
)
from sklearn.metrics import (
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, roc_curve, precision_recall_curve,
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error,
    # Clustering metrics
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, adjusted_mutual_info_score, homogeneity_score, completeness_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Advanced statistical tests
try:
    from scipy.stats import cramers_v, chi2, norm
except ImportError:
    pass

# Optional advanced libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Model interpretability will be limited.")

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not available. Experiment tracking will be limited.")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some visualizations will be limited.")

warnings.filterwarnings('ignore')


class UltraAdvancedModelEvaluation:
    """
    Ultra-advanced model evaluation and validation engine
    """
    
    def __init__(self, experiment_tracking: bool = True, random_state: int = 42):
        """
        Initialize the model evaluation engine
        
        Parameters:
        -----------
        experiment_tracking : bool
            Whether to enable MLflow experiment tracking
        random_state : int
            Random state for reproducibility
        """
        self.random_state = random_state
        self.experiment_tracking = experiment_tracking and MLFLOW_AVAILABLE
        self.evaluation_results = {}
        self.ab_test_results = {}
        self.model_performance_history = defaultdict(list)
        
        if self.experiment_tracking:
            try:
                mlflow.set_tracking_uri("sqlite:///model_experiments.db")
                mlflow.set_experiment("Customer_Segmentation_Models")
            except Exception as e:
                print(f"MLflow setup failed: {e}")
                self.experiment_tracking = False
    
    def evaluate_classification_model(self, model: Any, X_test: np.ndarray, 
                                    y_test: np.ndarray, model_name: str = "Model",
                                    class_names: List[str] = None) -> Dict:
        """
        Comprehensive evaluation of classification models
        
        Parameters:
        -----------
        model : Any
            Trained classification model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test labels
        model_name : str
            Name of the model for tracking
        class_names : List[str]
            Names of the classes
        
        Returns:
        --------
        Dict
            Comprehensive evaluation metrics
        """
        print(f"🔍 Evaluating Classification Model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = None
        
        try:
            y_pred_proba = model.predict_proba(X_test)
        except:
            pass
        
        # Core metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # AUC metrics
        auc_score = None
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:  # Binary classification
                    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:  # Multi-class classification
                    auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except:
                pass
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=class_names, 
                                           output_dict=True, zero_division=0)
        
        # Advanced metrics
        evaluation_results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'n_samples': len(y_test),
            'n_features': X_test.shape[1] if len(X_test.shape) > 1 else 1,
            'n_classes': len(np.unique(y_test)),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Model interpretability with SHAP
        if SHAP_AVAILABLE:
            try:
                shap_values = self._calculate_shap_values(model, X_test, model_name)
                evaluation_results['shap_values'] = shap_values
            except Exception as e:
                print(f"SHAP analysis failed: {e}")
        
        # Log to MLflow
        if self.experiment_tracking:
            self._log_classification_metrics(evaluation_results)
        
        # Store results
        self.evaluation_results[f'{model_name}_classification'] = evaluation_results
        self.model_performance_history[model_name].append(evaluation_results)
        
        print(f"✅ Classification Evaluation Complete:")
        print(f"   📊 Accuracy: {accuracy:.4f}")
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   🔍 Recall: {recall:.4f}")
        print(f"   ⚖️ F1-Score: {f1:.4f}")
        if auc_score:
            print(f"   📈 AUC: {auc_score:.4f}")
        
        return evaluation_results
    
    def evaluate_regression_model(self, model: Any, X_test: np.ndarray, 
                                y_test: np.ndarray, model_name: str = "Model") -> Dict:
        """
        Comprehensive evaluation of regression models
        
        Parameters:
        -----------
        model : Any
            Trained regression model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        model_name : str
            Name of the model for tracking
        
        Returns:
        --------
        Dict
            Comprehensive evaluation metrics
        """
        print(f"🔍 Evaluating Regression Model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Core metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Advanced metrics
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100
        
        # Residual analysis
        residuals = y_test - y_pred
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        
        # Prediction intervals
        prediction_intervals = self._calculate_prediction_intervals(y_pred, residuals)
        
        evaluation_results = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'prediction_intervals': prediction_intervals,
            'n_samples': len(y_test),
            'n_features': X_test.shape[1] if len(X_test.shape) > 1 else 1,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Model interpretability with SHAP
        if SHAP_AVAILABLE:
            try:
                shap_values = self._calculate_shap_values(model, X_test, model_name)
                evaluation_results['shap_values'] = shap_values
            except Exception as e:
                print(f"SHAP analysis failed: {e}")
        
        # Log to MLflow
        if self.experiment_tracking:
            self._log_regression_metrics(evaluation_results)
        
        # Store results
        self.evaluation_results[f'{model_name}_regression'] = evaluation_results
        self.model_performance_history[model_name].append(evaluation_results)
        
        print(f"✅ Regression Evaluation Complete:")
        print(f"   📊 R² Score: {r2:.4f}")
        print(f"   📉 RMSE: {rmse:.4f}")
        print(f"   📏 MAE: {mae:.4f}")
        print(f"   📈 MAPE: {mape:.2f}%")
        
        return evaluation_results
    
    def evaluate_clustering_model(self, model: Any, X: np.ndarray, 
                                labels: np.ndarray, model_name: str = "Model") -> Dict:
        """
        Comprehensive evaluation of clustering models
        
        Parameters:
        -----------
        model : Any
            Trained clustering model
        X : np.ndarray
            Features used for clustering
        labels : np.ndarray
            Cluster labels
        model_name : str
            Name of the model for tracking
        
        Returns:
        --------
        Dict
            Comprehensive evaluation metrics
        """
        print(f"🔍 Evaluating Clustering Model: {model_name}")
        
        # Core clustering metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        
        # Cluster statistics
        n_clusters = len(np.unique(labels))
        cluster_sizes = np.bincount(labels[labels >= 0])  # Exclude noise points (-1)
        cluster_balance = np.std(cluster_sizes) / np.mean(cluster_sizes) if len(cluster_sizes) > 0 else 0
        
        # Inertia (for K-means type algorithms)
        inertia = None
        try:
            inertia = model.inertia_
        except:
            pass
        
        evaluation_results = {
            'model_name': model_name,
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes.tolist() if cluster_sizes.size > 0 else [],
            'cluster_balance': cluster_balance,
            'inertia': inertia,
            'n_samples': len(labels),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Log to MLflow
        if self.experiment_tracking:
            self._log_clustering_metrics(evaluation_results)
        
        # Store results
        self.evaluation_results[f'{model_name}_clustering'] = evaluation_results
        self.model_performance_history[model_name].append(evaluation_results)
        
        print(f"✅ Clustering Evaluation Complete:")
        print(f"   🎯 Silhouette Score: {silhouette:.4f}")
        print(f"   📊 Davies-Bouldin Score: {davies_bouldin:.4f}")
        print(f"   📈 Calinski-Harabasz Score: {calinski_harabasz:.2f}")
        print(f"   🔢 Number of Clusters: {n_clusters}")
        
        return evaluation_results
    
    def perform_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray,
                               cv_method: str = 'stratified_kfold', n_splits: int = 5,
                               scoring: Union[str, List[str]] = None,
                               model_name: str = "Model") -> Dict:
        """
        Perform comprehensive cross-validation analysis
        
        Parameters:
        -----------
        model : Any
            Model to cross-validate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        cv_method : str
            Cross-validation method: 'stratified_kfold', 'kfold', 'time_series'
        n_splits : int
            Number of CV splits
        scoring : Union[str, List[str]]
            Scoring metrics
        model_name : str
            Name of the model
        
        Returns:
        --------
        Dict
            Cross-validation results
        """
        print(f"🔄 Performing Cross-Validation: {model_name}")
        
        # Set up cross-validation strategy
        if cv_method == 'stratified_kfold':
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        elif cv_method == 'kfold':
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        elif cv_method == 'time_series':
            cv = TimeSeriesSplit(n_splits=n_splits)
        else:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Set up scoring metrics
        if scoring is None:
            # Auto-detect based on target type
            if len(np.unique(y)) <= 10 and np.issubdtype(y.dtype, np.integer):
                # Classification
                scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            else:
                # Regression
                scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        # Perform cross-validation
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                                   return_train_score=True, return_estimator=True)
        
        # Calculate statistics
        results = {
            'model_name': model_name,
            'cv_method': cv_method,
            'n_splits': n_splits,
            'cv_results': {}
        }
        
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results['cv_results'][metric] = {
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores),
                'test_scores': test_scores.tolist(),
                'train_mean': np.mean(train_scores),
                'train_std': np.std(train_scores),
                'train_scores': train_scores.tolist(),
                'overfitting_score': np.mean(train_scores) - np.mean(test_scores)
            }
        
        results['evaluation_timestamp'] = datetime.now().isoformat()
        
        # Log to MLflow
        if self.experiment_tracking:
            self._log_cross_validation_metrics(results)
        
        # Store results
        self.evaluation_results[f'{model_name}_cv'] = results
        
        print(f"✅ Cross-Validation Complete:")
        for metric, scores in results['cv_results'].items():
            print(f"   {metric}: {scores['test_mean']:.4f} (±{scores['test_std']:.4f})")
        
        return results
    
    def conduct_ab_test(self, control_group: pd.DataFrame, treatment_group: pd.DataFrame,
                       metric_column: str, test_name: str = "AB_Test",
                       confidence_level: float = 0.95) -> Dict:
        """
        Conduct comprehensive A/B testing analysis
        
        Parameters:
        -----------
        control_group : pd.DataFrame
            Control group data
        treatment_group : pd.DataFrame
            Treatment group data
        metric_column : str
            Column name containing the metric to test
        test_name : str
            Name of the A/B test
        confidence_level : float
            Confidence level for statistical tests
        
        Returns:
        --------
        Dict
            A/B test results
        """
        print(f"🧪 Conducting A/B Test: {test_name}")
        
        # Extract metrics
        control_metric = control_group[metric_column].dropna()
        treatment_metric = treatment_group[metric_column].dropna()
        
        # Basic statistics
        control_stats = {
            'mean': control_metric.mean(),
            'std': control_metric.std(),
            'median': control_metric.median(),
            'count': len(control_metric),
            'q25': control_metric.quantile(0.25),
            'q75': control_metric.quantile(0.75)
        }
        
        treatment_stats = {
            'mean': treatment_metric.mean(),
            'std': treatment_metric.std(),
            'median': treatment_metric.median(),
            'count': len(treatment_metric),
            'q25': treatment_metric.quantile(0.25),
            'q75': treatment_metric.quantile(0.75)
        }
        
        # Effect size calculations
        absolute_effect = treatment_stats['mean'] - control_stats['mean']
        relative_effect = (absolute_effect / control_stats['mean']) * 100 if control_stats['mean'] != 0 else 0
        
        # Cohen's d (effect size)
        pooled_std = np.sqrt(((control_stats['count'] - 1) * control_stats['std']**2 + 
                             (treatment_stats['count'] - 1) * treatment_stats['std']**2) / 
                            (control_stats['count'] + treatment_stats['count'] - 2))
        cohens_d = absolute_effect / pooled_std if pooled_std != 0 else 0
        
        # Statistical tests
        alpha = 1 - confidence_level
        
        # T-test (parametric)
        t_stat, t_pvalue = ttest_ind(treatment_metric, control_metric, equal_var=False)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = mannwhitneyu(treatment_metric, control_metric, alternative='two-sided')
        
        # Kolmogorov-Smirnov test (distribution comparison)
        ks_stat, ks_pvalue = ks_2samp(treatment_metric, control_metric)
        
        # Power analysis and sample size
        power_analysis = self._calculate_power_analysis(
            control_metric, treatment_metric, alpha, cohens_d
        )
        
        # Confidence intervals
        ci_lower, ci_upper = self._calculate_confidence_interval(
            control_metric, treatment_metric, confidence_level
        )
        
        # Determine statistical significance
        is_significant_t = t_pvalue < alpha
        is_significant_u = u_pvalue < alpha
        is_significant_ks = ks_pvalue < alpha
        
        # Business significance thresholds
        minimum_detectable_effect = 0.02  # 2% minimum business impact
        is_business_significant = abs(relative_effect) >= minimum_detectable_effect * 100
        
        ab_test_results = {
            'test_name': test_name,
            'metric_column': metric_column,
            'confidence_level': confidence_level,
            'control_stats': control_stats,
            'treatment_stats': treatment_stats,
            'effect_analysis': {
                'absolute_effect': absolute_effect,
                'relative_effect': relative_effect,
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(cohens_d)
            },
            'statistical_tests': {
                't_test': {'statistic': t_stat, 'p_value': t_pvalue, 'significant': is_significant_t},
                'mann_whitney_u': {'statistic': u_stat, 'p_value': u_pvalue, 'significant': is_significant_u},
                'kolmogorov_smirnov': {'statistic': ks_stat, 'p_value': ks_pvalue, 'significant': is_significant_ks}
            },
            'confidence_interval': {
                'lower_bound': ci_lower,
                'upper_bound': ci_upper
            },
            'power_analysis': power_analysis,
            'significance_summary': {
                'statistically_significant': any([is_significant_t, is_significant_u]),
                'business_significant': is_business_significant,
                'recommended_action': self._get_ab_test_recommendation(
                    is_significant_t, is_business_significant, relative_effect
                )
            },
            'test_timestamp': datetime.now().isoformat()
        }
        
        # Log to MLflow
        if self.experiment_tracking:
            self._log_ab_test_metrics(ab_test_results)
        
        # Store results
        self.ab_test_results[test_name] = ab_test_results
        
        print(f"✅ A/B Test Complete:")
        print(f"   📊 Relative Effect: {relative_effect:.2f}%")
        print(f"   📈 Cohen's d: {cohens_d:.4f}")
        print(f"   🎯 Statistically Significant: {any([is_significant_t, is_significant_u])}")
        print(f"   💼 Business Significant: {is_business_significant}")
        print(f"   🎯 Recommendation: {ab_test_results['significance_summary']['recommended_action']}")
        
        return ab_test_results
    
    def compare_models(self, models_results: Dict[str, Dict], 
                      comparison_type: str = "classification") -> Dict:
        """
        Compare multiple models comprehensively
        
        Parameters:
        -----------
        models_results : Dict[str, Dict]
            Dictionary of model results from evaluations
        comparison_type : str
            Type of comparison: 'classification', 'regression', 'clustering'
        
        Returns:
        --------
        Dict
            Model comparison results
        """
        print(f"🏆 Comparing Models: {comparison_type}")
        
        if not models_results:
            print("No model results provided for comparison")
            return {}
        
        comparison_results = {
            'comparison_type': comparison_type,
            'models_compared': list(models_results.keys()),
            'comparison_timestamp': datetime.now().isoformat(),
            'rankings': {},
            'statistical_comparisons': {}
        }
        
        if comparison_type == "classification":
            # Key metrics for classification
            key_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'auc_score']
            
            for metric in key_metrics:
                metric_scores = {}
                for model_name, results in models_results.items():
                    if metric in results and results[metric] is not None:
                        metric_scores[model_name] = results[metric]
                
                if metric_scores:
                    # Rank models for this metric
                    sorted_models = sorted(metric_scores.items(), 
                                         key=lambda x: x[1], reverse=True)
                    comparison_results['rankings'][metric] = sorted_models
        
        elif comparison_type == "regression":
            # Key metrics for regression (lower is better for error metrics)
            error_metrics = ['mse', 'rmse', 'mae', 'mape']
            performance_metrics = ['r2_score']
            
            for metric in error_metrics:
                metric_scores = {}
                for model_name, results in models_results.items():
                    if metric in results and results[metric] is not None:
                        metric_scores[model_name] = results[metric]
                
                if metric_scores:
                    # Rank models (lower is better for error metrics)
                    sorted_models = sorted(metric_scores.items(), 
                                         key=lambda x: x[1], reverse=False)
                    comparison_results['rankings'][metric] = sorted_models
            
            for metric in performance_metrics:
                metric_scores = {}
                for model_name, results in models_results.items():
                    if metric in results and results[metric] is not None:
                        metric_scores[model_name] = results[metric]
                
                if metric_scores:
                    # Rank models (higher is better for performance metrics)
                    sorted_models = sorted(metric_scores.items(), 
                                         key=lambda x: x[1], reverse=True)
                    comparison_results['rankings'][metric] = sorted_models
        
        elif comparison_type == "clustering":
            # Key metrics for clustering
            key_metrics = ['silhouette_score', 'calinski_harabasz_score']  # Higher is better
            inverse_metrics = ['davies_bouldin_score']  # Lower is better
            
            for metric in key_metrics:
                metric_scores = {}
                for model_name, results in models_results.items():
                    if metric in results and results[metric] is not None:
                        metric_scores[model_name] = results[metric]
                
                if metric_scores:
                    sorted_models = sorted(metric_scores.items(), 
                                         key=lambda x: x[1], reverse=True)
                    comparison_results['rankings'][metric] = sorted_models
            
            for metric in inverse_metrics:
                metric_scores = {}
                for model_name, results in models_results.items():
                    if metric in results and results[metric] is not None:
                        metric_scores[model_name] = results[metric]
                
                if metric_scores:
                    sorted_models = sorted(metric_scores.items(), 
                                         key=lambda x: x[1], reverse=False)
                    comparison_results['rankings'][metric] = sorted_models
        
        # Overall ranking (aggregate across metrics)
        overall_scores = defaultdict(float)
        metric_count = defaultdict(int)
        
        for metric, rankings in comparison_results['rankings'].items():
            for rank, (model_name, score) in enumerate(rankings):
                # Inverse rank scoring (1st place = highest score)
                rank_score = len(rankings) - rank
                overall_scores[model_name] += rank_score
                metric_count[model_name] += 1
        
        # Normalize by number of metrics each model was evaluated on
        normalized_scores = {
            model: score / metric_count[model] if metric_count[model] > 0 else 0
            for model, score in overall_scores.items()
        }
        
        overall_ranking = sorted(normalized_scores.items(), 
                               key=lambda x: x[1], reverse=True)
        comparison_results['overall_ranking'] = overall_ranking
        
        # Best model recommendation
        if overall_ranking:
            best_model = overall_ranking[0][0]
            comparison_results['recommended_model'] = best_model
            comparison_results['recommendation_reason'] = (
                f"Best overall performance across {len(comparison_results['rankings'])} metrics"
            )
        
        print(f"✅ Model Comparison Complete:")
        print(f"   🏆 Recommended Model: {comparison_results.get('recommended_model', 'None')}")
        
        return comparison_results
    
    def generate_model_performance_report(self, model_name: str = None) -> Dict:
        """
        Generate comprehensive performance report
        
        Parameters:
        -----------
        model_name : str
            Specific model to report on (None for all models)
        
        Returns:
        --------
        Dict
            Performance report
        """
        print("📋 Generating Performance Report...")
        
        if model_name and model_name in self.model_performance_history:
            models_to_report = {model_name: self.model_performance_history[model_name]}
        else:
            models_to_report = self.model_performance_history
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'models_analyzed': list(models_to_report.keys()),
            'total_evaluations': sum(len(evals) for evals in models_to_report.values()),
            'model_summaries': {},
            'performance_trends': {},
            'recommendations': []
        }
        
        for model, evaluations in models_to_report.items():
            if not evaluations:
                continue
            
            # Latest evaluation
            latest_eval = evaluations[-1]
            
            # Performance trends (if multiple evaluations)
            trend_analysis = None
            if len(evaluations) > 1:
                trend_analysis = self._analyze_performance_trends(evaluations)
            
            report['model_summaries'][model] = {
                'latest_evaluation': latest_eval,
                'evaluation_count': len(evaluations),
                'trend_analysis': trend_analysis
            }
        
        # Global recommendations
        report['recommendations'] = self._generate_performance_recommendations(report)
        
        print(f"✅ Performance Report Generated for {len(models_to_report)} models")
        
        return report
    
    # Helper methods
    def _calculate_shap_values(self, model: Any, X_test: np.ndarray, 
                             model_name: str) -> Dict:
        """Calculate SHAP values for model interpretability"""
        try:
            # Create explainer based on model type
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test[:100])  # Limit for performance
            
            return {
                'available': True,
                'feature_importance': np.abs(shap_values.values).mean(0).tolist(),
                'sample_explanations': shap_values.values[:10].tolist()  # First 10 samples
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}
    
    def _calculate_prediction_intervals(self, y_pred: np.ndarray, 
                                      residuals: np.ndarray, 
                                      confidence: float = 0.95) -> Dict:
        """Calculate prediction intervals for regression models"""
        z_score = stats.norm.ppf(1 - (1 - confidence) / 2)
        std_error = np.std(residuals)
        
        lower_bound = y_pred - z_score * std_error
        upper_bound = y_pred + z_score * std_error
        
        return {
            'confidence_level': confidence,
            'lower_bound': lower_bound.tolist(),
            'upper_bound': upper_bound.tolist(),
            'width': (upper_bound - lower_bound).tolist()
        }
    
    def _calculate_power_analysis(self, control: np.ndarray, treatment: np.ndarray,
                                alpha: float, effect_size: float) -> Dict:
        """Calculate power analysis for A/B tests"""
        # Simplified power calculation
        n1, n2 = len(control), len(treatment)
        pooled_std = np.sqrt((np.var(control) + np.var(treatment)) / 2)
        
        # Effect size
        delta = abs(np.mean(treatment) - np.mean(control)) / pooled_std
        
        # Approximate power calculation
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = delta * np.sqrt(n1 * n2 / (n1 + n2)) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return {
            'statistical_power': max(0, min(1, power)),
            'effect_size': delta,
            'sample_size_control': n1,
            'sample_size_treatment': n2,
            'alpha': alpha
        }
    
    def _calculate_confidence_interval(self, control: np.ndarray, 
                                     treatment: np.ndarray,
                                     confidence_level: float) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means"""
        mean_diff = np.mean(treatment) - np.mean(control)
        
        # Standard error of difference
        se_diff = np.sqrt(np.var(control) / len(control) + np.var(treatment) / len(treatment))
        
        # Degrees of freedom (Welch's t-test)
        df = ((np.var(control) / len(control) + np.var(treatment) / len(treatment))**2 /
              ((np.var(control) / len(control))**2 / (len(control) - 1) +
               (np.var(treatment) / len(treatment))**2 / (len(treatment) - 1)))
        
        # Critical value
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha / 2, df)
        
        # Confidence interval
        margin_error = t_critical * se_diff
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        return ci_lower, ci_upper
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _get_ab_test_recommendation(self, is_statistically_significant: bool,
                                  is_business_significant: bool,
                                  relative_effect: float) -> str:
        """Generate A/B test recommendation"""
        if is_statistically_significant and is_business_significant:
            if relative_effect > 0:
                return "Implement treatment - statistically and business significant positive effect"
            else:
                return "Reject treatment - statistically and business significant negative effect"
        elif is_statistically_significant and not is_business_significant:
            return "Monitor further - statistically significant but limited business impact"
        elif not is_statistically_significant and is_business_significant:
            return "Extend test - potentially business significant but needs more data"
        else:
            return "No significant effect detected - consider alternative approaches"
    
    def _analyze_performance_trends(self, evaluations: List[Dict]) -> Dict:
        """Analyze performance trends across evaluations"""
        if len(evaluations) < 2:
            return None
        
        # Extract timestamps and a key metric
        timestamps = [eval.get('evaluation_timestamp', '') for eval in evaluations]
        
        # Try to find a consistent metric across evaluations
        key_metric = None
        values = []
        
        for metric in ['accuracy', 'f1_score', 'r2_score', 'silhouette_score']:
            if all(metric in eval and eval[metric] is not None for eval in evaluations):
                key_metric = metric
                values = [eval[metric] for eval in evaluations]
                break
        
        if not key_metric or not values:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend
        if len(values) >= 2:
            recent_avg = np.mean(values[-3:])  # Last 3 evaluations
            older_avg = np.mean(values[:-3]) if len(values) > 3 else values[0]
            
            if recent_avg > older_avg * 1.05:  # 5% improvement threshold
                trend = 'improving'
            elif recent_avg < older_avg * 0.95:  # 5% degradation threshold
                trend = 'degrading'
            else:
                trend = 'stable'
            
            return {
                'trend': trend,
                'key_metric': key_metric,
                'recent_performance': recent_avg,
                'baseline_performance': older_avg,
                'improvement_rate': (recent_avg - older_avg) / older_avg if older_avg != 0 else 0
            }
        
        return {'trend': 'insufficient_data'}
    
    def _generate_performance_recommendations(self, report: Dict) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Analyze model summaries for recommendations
        for model_name, summary in report['model_summaries'].items():
            latest_eval = summary['latest_evaluation']
            trend = summary.get('trend_analysis', {})
            
            # Check for performance issues
            if 'accuracy' in latest_eval and latest_eval['accuracy'] < 0.7:
                recommendations.append(
                    f"📊 {model_name}: Consider feature engineering or model complexity increase (accuracy: {latest_eval['accuracy']:.3f})"
                )
            
            if 'r2_score' in latest_eval and latest_eval['r2_score'] < 0.5:
                recommendations.append(
                    f"📈 {model_name}: Low R² score ({latest_eval['r2_score']:.3f}) - consider alternative algorithms or feature selection"
                )
            
            if trend.get('trend') == 'degrading':
                recommendations.append(
                    f"⚠️ {model_name}: Performance degrading - investigate data drift or model decay"
                )
            elif trend.get('trend') == 'improving':
                recommendations.append(
                    f"✅ {model_name}: Performance improving - consider current approach for other models"
                )
        
        if not recommendations:
            recommendations.append("✅ All models performing within expected ranges")
        
        return recommendations
    
    # MLflow logging methods
    def _log_classification_metrics(self, results: Dict):
        """Log classification metrics to MLflow"""
        if not self.experiment_tracking:
            return
        
        try:
            with mlflow.start_run(run_name=f"{results['model_name']}_classification"):
                mlflow.log_metrics({
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score']
                })
                if results['auc_score']:
                    mlflow.log_metric('auc_score', results['auc_score'])
                
                mlflow.log_params({
                    'n_samples': results['n_samples'],
                    'n_features': results['n_features'],
                    'n_classes': results['n_classes']
                })
        except Exception as e:
            print(f"MLflow logging failed: {e}")
    
    def _log_regression_metrics(self, results: Dict):
        """Log regression metrics to MLflow"""
        if not self.experiment_tracking:
            return
        
        try:
            with mlflow.start_run(run_name=f"{results['model_name']}_regression"):
                mlflow.log_metrics({
                    'mse': results['mse'],
                    'rmse': results['rmse'],
                    'mae': results['mae'],
                    'r2_score': results['r2_score'],
                    'mape': results['mape']
                })
                
                mlflow.log_params({
                    'n_samples': results['n_samples'],
                    'n_features': results['n_features']
                })
        except Exception as e:
            print(f"MLflow logging failed: {e}")
    
    def _log_clustering_metrics(self, results: Dict):
        """Log clustering metrics to MLflow"""
        if not self.experiment_tracking:
            return
        
        try:
            with mlflow.start_run(run_name=f"{results['model_name']}_clustering"):
                mlflow.log_metrics({
                    'silhouette_score': results['silhouette_score'],
                    'davies_bouldin_score': results['davies_bouldin_score'],
                    'calinski_harabasz_score': results['calinski_harabasz_score']
                })
                
                mlflow.log_params({
                    'n_clusters': results['n_clusters'],
                    'n_samples': results['n_samples'],
                    'n_features': results['n_features']
                })
        except Exception as e:
            print(f"MLflow logging failed: {e}")
    
    def _log_cross_validation_metrics(self, results: Dict):
        """Log cross-validation metrics to MLflow"""
        if not self.experiment_tracking:
            return
        
        try:
            with mlflow.start_run(run_name=f"{results['model_name']}_cv"):
                for metric, scores in results['cv_results'].items():
                    mlflow.log_metrics({
                        f'cv_{metric}_mean': scores['test_mean'],
                        f'cv_{metric}_std': scores['test_std'],
                        f'cv_{metric}_overfitting': scores['overfitting_score']
                    })
        except Exception as e:
            print(f"MLflow logging failed: {e}")
    
    def _log_ab_test_metrics(self, results: Dict):
        """Log A/B test metrics to MLflow"""
        if not self.experiment_tracking:
            return
        
        try:
            with mlflow.start_run(run_name=f"ABTest_{results['test_name']}"):
                mlflow.log_metrics({
                    'relative_effect': results['effect_analysis']['relative_effect'],
                    'cohens_d': results['effect_analysis']['cohens_d'],
                    't_test_pvalue': results['statistical_tests']['t_test']['p_value'],
                    'statistical_power': results['power_analysis']['statistical_power']
                })
                
                mlflow.log_params({
                    'confidence_level': results['confidence_level'],
                    'control_size': results['control_stats']['count'],
                    'treatment_size': results['treatment_stats']['count']
                })
        except Exception as e:
            print(f"MLflow logging failed: {e}")


# Additional utility functions for model evaluation
def calculate_business_metrics(predictions_df: pd.DataFrame, 
                             actual_revenue_col: str = 'actual_revenue',
                             predicted_revenue_col: str = 'predicted_revenue') -> Dict:
    """
    Calculate business-specific metrics for model evaluation
    
    Parameters:
    -----------
    predictions_df : pd.DataFrame
        DataFrame with actual and predicted values
    actual_revenue_col : str
        Column name for actual revenue
    predicted_revenue_col : str
        Column name for predicted revenue
    
    Returns:
    --------
    Dict
        Business metrics
    """
    
    actual = predictions_df[actual_revenue_col]
    predicted = predictions_df[predicted_revenue_col]
    
    # Revenue impact metrics
    total_actual_revenue = actual.sum()
    total_predicted_revenue = predicted.sum()
    revenue_error = total_predicted_revenue - total_actual_revenue
    revenue_error_pct = (revenue_error / total_actual_revenue) * 100 if total_actual_revenue != 0 else 0
    
    # Customer value prediction accuracy
    customer_accuracy = []
    for idx in predictions_df.index:
        actual_val = actual.loc[idx]
        predicted_val = predicted.loc[idx]
        
        if actual_val != 0:
            accuracy = 1 - abs(predicted_val - actual_val) / actual_val
            customer_accuracy.append(max(0, accuracy))  # Cap at 0
    
    avg_customer_accuracy = np.mean(customer_accuracy) if customer_accuracy else 0
    
    return {
        'total_actual_revenue': total_actual_revenue,
        'total_predicted_revenue': total_predicted_revenue,
        'revenue_error': revenue_error,
        'revenue_error_percentage': revenue_error_pct,
        'average_customer_accuracy': avg_customer_accuracy,
        'high_accuracy_customers_pct': np.sum(np.array(customer_accuracy) > 0.8) / len(customer_accuracy) * 100 if customer_accuracy else 0
    }


def validate_model_stability(model: Any, X: np.ndarray, y: np.ndarray,
                           n_iterations: int = 10, test_size: float = 0.3,
                           random_state: int = 42) -> Dict:
    """
    Validate model stability across different data splits
    
    Parameters:
    -----------
    model : Any
        Model to validate
    X : np.ndarray
        Features
    y : np.ndarray
        Target
    n_iterations : int
        Number of stability iterations
    test_size : float
        Test set size for each iteration
    
    Returns:
    --------
    Dict
        Stability metrics
    """
    
    scores = []
    
    for i in range(n_iterations):
        # Random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=random_state + i, stratify=y if len(np.unique(y)) <= 10 else None
        )
        
        # Train and evaluate
        model_clone = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
        model_clone.fit(X_train, y_train)
        
        # Score based on problem type
        if len(np.unique(y)) <= 10 and np.issubdtype(y.dtype, np.integer):
            # Classification
            score = accuracy_score(y_test, model_clone.predict(X_test))
        else:
            # Regression
            score = r2_score(y_test, model_clone.predict(X_test))
        
        scores.append(score)
    
    return {
        'stability_scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'stability_coefficient': np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else float('inf'),
        'is_stable': np.std(scores) < 0.1  # Arbitrary threshold for stability
    }