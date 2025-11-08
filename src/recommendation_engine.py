"""
Ultra-Advanced Recommendation Engine
Comprehensive recommendation system for e-commerce customer segmentation
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import pickle
import json
from collections import defaultdict

# Core ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

# Advanced ML models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Scipy for statistical functions
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform

# Advanced libraries (with fallbacks)
try:
    import implicit
    IMPLICIT_AVAILABLE = True
except ImportError:
    IMPLICIT_AVAILABLE = False

try:
    from surprise import Dataset, Reader, SVD, KNNBasic, accuracy
    from surprise.model_selection import cross_validate, train_test_split as surprise_split
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class CollaborativeFilteringEngine:
    """
    Advanced collaborative filtering recommendation system
    """
    
    def __init__(self, method='matrix_factorization', random_state=42):
        """
        Initialize collaborative filtering engine
        
        Parameters:
        -----------
        method : str
            CF method: 'matrix_factorization', 'user_based', 'item_based', 'deep_learning'
        """
        self.method = method
        self.random_state = random_state
        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.is_fitted = False
        self.user_item_matrix = None
        self.item_features = None
        
    def prepare_interaction_matrix(self, df: pd.DataFrame, 
                                 user_col: str = 'CustomerID',
                                 item_col: str = 'StockCode', 
                                 rating_col: str = 'Quantity',
                                 implicit_feedback: bool = True) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
        """
        Prepare user-item interaction matrix
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        implicit_feedback : bool
            Whether to treat as implicit feedback (binary) or explicit ratings
        """
        
        # Clean data
        interaction_df = df[[user_col, item_col, rating_col]].copy()
        interaction_df = interaction_df.dropna()
        
        # Remove users/items with very few interactions
        user_counts = interaction_df[user_col].value_counts()
        item_counts = interaction_df[item_col].value_counts()
        
        min_user_interactions = 3
        min_item_interactions = 3
        
        valid_users = user_counts[user_counts >= min_user_interactions].index
        valid_items = item_counts[item_counts >= min_item_interactions].index
        
        interaction_df = interaction_df[
            (interaction_df[user_col].isin(valid_users)) & 
            (interaction_df[item_col].isin(valid_items))
        ]
        
        print(f"Filtered to {len(interaction_df)} interactions between "
              f"{interaction_df[user_col].nunique()} users and {interaction_df[item_col].nunique()} items")
        
        # Encode users and items
        interaction_df['user_encoded'] = self.user_encoder.fit_transform(interaction_df[user_col])
        interaction_df['item_encoded'] = self.item_encoder.fit_transform(interaction_df[item_col])
        
        # Create ratings
        if implicit_feedback:
            # Convert to binary or use log-normalized quantities
            interaction_df['rating'] = np.log1p(interaction_df[rating_col])
            # Normalize to 0-5 scale
            interaction_df['rating'] = MinMaxScaler(feature_range=(1, 5)).fit_transform(
                interaction_df[['rating']]
            ).flatten()
        else:
            interaction_df['rating'] = interaction_df[rating_col]
        
        # Create sparse matrix
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        user_item_matrix = csr_matrix(
            (interaction_df['rating'], 
             (interaction_df['user_encoded'], interaction_df['item_encoded'])),
            shape=(n_users, n_items)
        )
        
        self.user_item_matrix = user_item_matrix
        
        return user_item_matrix, self.user_encoder.classes_, self.item_encoder.classes_
    
    def train_matrix_factorization(self, user_item_matrix: csr_matrix, n_factors: int = 50):
        """
        Train matrix factorization model
        """
        
        if IMPLICIT_AVAILABLE:
            print("Training ALS matrix factorization...")
            
            # Use implicit library for ALS
            self.model = implicit.als.AlternatingLeastSquares(
                factors=n_factors,
                regularization=0.01,
                iterations=20,
                random_state=self.random_state
            )
            
            # Implicit expects item-user matrix (transposed)
            item_user_matrix = user_item_matrix.T.tocsr()
            self.model.fit(item_user_matrix)
            
        elif SURPRISE_AVAILABLE:
            print("Training SVD matrix factorization...")
            
            # Convert to surprise format
            reader = Reader(rating_scale=(1, 5))
            
            # Create ratings dataframe
            rows, cols = user_item_matrix.nonzero()
            ratings_data = []
            for i in range(len(rows)):
                ratings_data.append({
                    'userID': rows[i],
                    'itemID': cols[i], 
                    'rating': user_item_matrix[rows[i], cols[i]]
                })
            
            ratings_df = pd.DataFrame(ratings_data)
            dataset = Dataset.load_from_df(ratings_df[['userID', 'itemID', 'rating']], reader)
            
            # Train model
            self.model = SVD(n_factors=n_factors, random_state=self.random_state)
            trainset = dataset.build_full_trainset()
            self.model.fit(trainset)
            
        else:
            print("Training custom NMF matrix factorization...")
            
            # Use sklearn NMF as fallback
            self.model = NMF(
                n_components=n_factors,
                init='random',
                random_state=self.random_state,
                max_iter=200
            )
            
            # Convert to dense for NMF
            user_item_dense = user_item_matrix.toarray()
            self.model.fit(user_item_dense)
        
        self.is_fitted = True
        print("Matrix factorization training completed!")
    
    def train_neighborhood_model(self, user_item_matrix: csr_matrix, 
                                method: str = 'user_based', k: int = 50):
        """
        Train neighborhood-based collaborative filtering
        
        Parameters:
        -----------
        method : str
            'user_based' or 'item_based'
        k : int
            Number of nearest neighbors
        """
        
        if SURPRISE_AVAILABLE:
            print(f"Training {method} neighborhood model...")
            
            # Convert to surprise format
            reader = Reader(rating_scale=(1, 5))
            
            rows, cols = user_item_matrix.nonzero()
            ratings_data = []
            for i in range(len(rows)):
                ratings_data.append({
                    'userID': rows[i],
                    'itemID': cols[i], 
                    'rating': user_item_matrix[rows[i], cols[i]]
                })
            
            ratings_df = pd.DataFrame(ratings_data)
            dataset = Dataset.load_from_df(ratings_df[['userID', 'itemID', 'rating']], reader)
            
            # Configure similarity and neighborhood
            sim_options = {
                'name': 'cosine',
                'user_based': method == 'user_based'
            }
            
            self.model = KNNBasic(k=k, sim_options=sim_options)
            trainset = dataset.build_full_trainset()
            self.model.fit(trainset)
            
        else:
            print(f"Training custom {method} neighborhood model...")
            
            # Custom implementation using cosine similarity
            if method == 'user_based':
                similarity_matrix = cosine_similarity(user_item_matrix)
            else:  # item_based
                similarity_matrix = cosine_similarity(user_item_matrix.T)
            
            self.model = {
                'similarity_matrix': similarity_matrix,
                'user_item_matrix': user_item_matrix,
                'method': method,
                'k': k
            }
        
        self.is_fitted = True
        print(f"{method.title()} neighborhood training completed!")
    
    def train_neural_collaborative_filtering(self, user_item_matrix: csr_matrix, 
                                           embedding_size: int = 50, epochs: int = 50):
        """
        Train neural collaborative filtering model
        """
        
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Falling back to matrix factorization.")
            self.train_matrix_factorization(user_item_matrix, embedding_size)
            return
        
        print("Training neural collaborative filtering...")
        
        # Prepare data
        rows, cols = user_item_matrix.nonzero()
        ratings = user_item_matrix.data
        
        n_users, n_items = user_item_matrix.shape
        
        # Create negative samples
        negative_samples = []
        for _ in range(len(rows)):
            user = np.random.randint(0, n_users)
            item = np.random.randint(0, n_items)
            if user_item_matrix[user, item] == 0:
                negative_samples.append((user, item, 0.0))
        
        # Combine positive and negative samples
        all_users = list(rows) + [x[0] for x in negative_samples]
        all_items = list(cols) + [x[1] for x in negative_samples]
        all_ratings = list(ratings) + [x[2] for x in negative_samples]
        
        # Build neural network
        user_input = keras.layers.Input(shape=(), name='user_id')
        item_input = keras.layers.Input(shape=(), name='item_id')
        
        user_embedding = keras.layers.Embedding(n_users, embedding_size)(user_input)
        item_embedding = keras.layers.Embedding(n_items, embedding_size)(item_input)
        
        user_vec = keras.layers.Flatten()(user_embedding)
        item_vec = keras.layers.Flatten()(item_embedding)
        
        # Neural MF
        concat = keras.layers.concatenate([user_vec, item_vec])
        dense1 = keras.layers.Dense(128, activation='relu')(concat)
        dense2 = keras.layers.Dense(64, activation='relu')(dense1)
        output = keras.layers.Dense(1, activation='sigmoid')(dense2)
        
        self.model = keras.Model(inputs=[user_input, item_input], outputs=output)
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        self.model.fit(
            [np.array(all_users), np.array(all_items)], 
            np.array(all_ratings),
            epochs=epochs,
            batch_size=256,
            validation_split=0.2,
            verbose=0
        )
        
        self.is_fitted = True
        print("Neural collaborative filtering training completed!")
    
    def get_user_recommendations(self, user_id: Union[str, int], n_recommendations: int = 10,
                               exclude_known: bool = True) -> List[Tuple[str, float]]:
        """
        Get recommendations for a specific user
        
        Parameters:
        -----------
        user_id : Union[str, int]
            User identifier
        n_recommendations : int
            Number of recommendations to return
        exclude_known : bool
            Whether to exclude items user has already interacted with
        """
        
        if not self.is_fitted:
            raise ValueError("Model not trained yet!")
        
        try:
            # Encode user ID
            if user_id in self.user_encoder.classes_:
                user_encoded = self.user_encoder.transform([user_id])[0]
            else:
                print(f"User {user_id} not found in training data")
                return []
            
            recommendations = []
            
            if IMPLICIT_AVAILABLE and hasattr(self.model, 'recommend'):
                # Use implicit library
                recommended_items, scores = self.model.recommend(
                    user_encoded, 
                    self.user_item_matrix,
                    N=n_recommendations,
                    filter_already_liked_items=exclude_known
                )
                
                # Decode item IDs
                for item_encoded, score in zip(recommended_items, scores):
                    item_id = self.item_encoder.inverse_transform([item_encoded])[0]
                    recommendations.append((item_id, float(score)))
            
            elif SURPRISE_AVAILABLE and hasattr(self.model, 'predict'):
                # Use surprise library
                n_items = len(self.item_encoder.classes_)
                
                # Get predictions for all items
                predictions = []
                for item_encoded in range(n_items):
                    pred = self.model.predict(user_encoded, item_encoded)
                    predictions.append((item_encoded, pred.est))
                
                # Sort by predicted rating
                predictions.sort(key=lambda x: x[1], reverse=True)
                
                # Filter known items if requested
                if exclude_known:
                    known_items = set(self.user_item_matrix[user_encoded].indices)
                    predictions = [(item, score) for item, score in predictions 
                                 if item not in known_items]
                
                # Get top N recommendations
                top_predictions = predictions[:n_recommendations]
                
                # Decode item IDs
                for item_encoded, score in top_predictions:
                    item_id = self.item_encoder.inverse_transform([item_encoded])[0]
                    recommendations.append((item_id, float(score)))
            
            else:
                # Custom implementation
                if isinstance(self.model, dict):
                    # Neighborhood model
                    recommendations = self._get_neighborhood_recommendations(
                        user_encoded, n_recommendations, exclude_known
                    )
                else:
                    # NMF model
                    recommendations = self._get_nmf_recommendations(
                        user_encoded, n_recommendations, exclude_known
                    )
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations for user {user_id}: {e}")
            return []
    
    def _get_neighborhood_recommendations(self, user_encoded: int, 
                                       n_recommendations: int, exclude_known: bool) -> List[Tuple[str, float]]:
        """Get recommendations using neighborhood model"""
        
        model_data = self.model
        similarity_matrix = model_data['similarity_matrix']
        user_item_matrix = model_data['user_item_matrix']
        method = model_data['method']
        k = model_data['k']
        
        if method == 'user_based':
            # Find similar users
            user_similarities = similarity_matrix[user_encoded]
            similar_users = np.argsort(user_similarities)[-k-1:-1]  # Exclude self
            
            # Calculate item scores based on similar users
            item_scores = np.zeros(user_item_matrix.shape[1])
            
            for similar_user in similar_users:
                similarity_score = user_similarities[similar_user]
                user_items = user_item_matrix[similar_user].toarray().flatten()
                item_scores += similarity_score * user_items
            
        else:  # item_based
            # Get items user has interacted with
            user_items = user_item_matrix[user_encoded].indices
            item_scores = np.zeros(user_item_matrix.shape[1])
            
            for user_item in user_items:
                # Find similar items
                item_similarities = similarity_matrix[user_item]
                similar_items = np.argsort(item_similarities)[-k:]
                
                for similar_item in similar_items:
                    similarity_score = item_similarities[similar_item]
                    item_scores[similar_item] += similarity_score
        
        # Get top recommendations
        if exclude_known:
            known_items = user_item_matrix[user_encoded].indices
            item_scores[known_items] = -1  # Set to negative to exclude
        
        top_items = np.argsort(item_scores)[-n_recommendations:][::-1]
        
        # Convert to recommendations with scores
        recommendations = []
        for item_encoded in top_items:
            if item_scores[item_encoded] > 0:
                item_id = self.item_encoder.inverse_transform([item_encoded])[0]
                recommendations.append((item_id, float(item_scores[item_encoded])))
        
        return recommendations
    
    def _get_nmf_recommendations(self, user_encoded: int, 
                               n_recommendations: int, exclude_known: bool) -> List[Tuple[str, float]]:
        """Get recommendations using NMF model"""
        
        # Get user factors and item factors
        user_factors = self.model.transform(self.user_item_matrix)[user_encoded]
        item_factors = self.model.components_.T
        
        # Calculate predicted ratings
        predicted_ratings = np.dot(user_factors, item_factors.T)
        
        # Exclude known items if requested
        if exclude_known:
            known_items = self.user_item_matrix[user_encoded].indices
            predicted_ratings[known_items] = -1
        
        # Get top recommendations
        top_items = np.argsort(predicted_ratings)[-n_recommendations:][::-1]
        
        recommendations = []
        for item_encoded in top_items:
            if predicted_ratings[item_encoded] > 0:
                item_id = self.item_encoder.inverse_transform([item_encoded])[0]
                recommendations.append((item_id, float(predicted_ratings[item_encoded])))
        
        return recommendations


class ContentBasedFilteringEngine:
    """
    Content-based filtering using item features and user preferences
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.item_features = None
        self.user_profiles = None
        self.tfidf_vectorizer = None
        self.is_fitted = False
        
    def prepare_item_features(self, df: pd.DataFrame, 
                            item_col: str = 'StockCode',
                            description_col: str = 'Description',
                            price_col: str = 'UnitPrice') -> pd.DataFrame:
        """
        Prepare item feature matrix from product data
        """
        
        print("Preparing item features for content-based filtering...")
        
        # Get unique items with their features
        item_data = df.groupby(item_col).agg({
            description_col: 'first',
            price_col: 'mean'
        }).reset_index()
        
        # Clean descriptions
        item_data[description_col] = item_data[description_col].fillna('').astype(str)
        
        # Create TF-IDF features from descriptions
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2
        )
        
        tfidf_features = self.tfidf_vectorizer.fit_transform(item_data[description_col])
        tfidf_df = pd.DataFrame(
            tfidf_features.toarray(),
            columns=self.tfidf_vectorizer.get_feature_names_out(),
            index=item_data[item_col]
        )
        
        # Add price features
        price_features = pd.DataFrame(index=item_data[item_col])
        price_features['price'] = item_data[price_col].values
        price_features['log_price'] = np.log1p(price_features['price'])
        
        # Normalize price features
        scaler = StandardScaler()
        price_features[['price_normalized', 'log_price_normalized']] = scaler.fit_transform(
            price_features[['price', 'log_price']]
        )
        
        # Combine all features
        self.item_features = pd.concat([tfidf_df, price_features], axis=1)
        
        print(f"Created {len(self.item_features.columns)} features for {len(self.item_features)} items")
        
        return self.item_features
    
    def build_user_profiles(self, df: pd.DataFrame,
                          user_col: str = 'CustomerID',
                          item_col: str = 'StockCode',
                          rating_col: str = 'Quantity'):
        """
        Build user profiles based on their interaction history
        """
        
        if self.item_features is None:
            raise ValueError("Item features not prepared yet!")
        
        print("Building user profiles...")
        
        self.user_profiles = {}
        
        for user_id in df[user_col].unique():
            user_data = df[df[user_col] == user_id]
            
            # Get items user has interacted with
            user_items = user_data[item_col].unique()
            user_items = [item for item in user_items if item in self.item_features.index]
            
            if len(user_items) == 0:
                continue
            
            # Get ratings/weights
            item_weights = user_data.groupby(item_col)[rating_col].sum()
            item_weights = item_weights.reindex(user_items, fill_value=1)
            
            # Normalize weights
            item_weights = item_weights / item_weights.sum()
            
            # Calculate weighted average of item features
            user_profile = np.zeros(len(self.item_features.columns))
            
            for item, weight in item_weights.items():
                if item in self.item_features.index:
                    item_feature_vector = self.item_features.loc[item].values
                    user_profile += weight * item_feature_vector
            
            self.user_profiles[user_id] = user_profile
        
        print(f"Created profiles for {len(self.user_profiles)} users")
        
        self.is_fitted = True
    
    def get_content_based_recommendations(self, user_id: Union[str, int],
                                        n_recommendations: int = 10,
                                        exclude_known: bool = True,
                                        df: pd.DataFrame = None) -> List[Tuple[str, float]]:
        """
        Get content-based recommendations for a user
        """
        
        if not self.is_fitted:
            raise ValueError("Model not trained yet!")
        
        if user_id not in self.user_profiles:
            print(f"User {user_id} not found in user profiles")
            return []
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate cosine similarity between user profile and all items
        similarities = cosine_similarity([user_profile], self.item_features.values)[0]
        
        # Get item similarities
        item_similarities = list(zip(self.item_features.index, similarities))
        item_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Exclude known items if requested
        if exclude_known and df is not None:
            user_items = set(df[df['CustomerID'] == user_id]['StockCode'].unique())
            item_similarities = [(item, sim) for item, sim in item_similarities 
                               if item not in user_items]
        
        # Return top N recommendations
        return item_similarities[:n_recommendations]


class HybridRecommendationEngine:
    """
    Hybrid recommendation system combining collaborative and content-based filtering
    """
    
    def __init__(self, cf_weight: float = 0.7, cb_weight: float = 0.3, random_state: int = 42):
        """
        Initialize hybrid recommendation engine
        
        Parameters:
        -----------
        cf_weight : float
            Weight for collaborative filtering recommendations
        cb_weight : float
            Weight for content-based recommendations
        """
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight
        self.random_state = random_state
        
        self.cf_engine = CollaborativeFilteringEngine(random_state=random_state)
        self.cb_engine = ContentBasedFilteringEngine(random_state=random_state)
        
        self.is_fitted = False
    
    def train(self, df: pd.DataFrame,
              user_col: str = 'CustomerID',
              item_col: str = 'StockCode',
              rating_col: str = 'Quantity',
              description_col: str = 'Description',
              price_col: str = 'UnitPrice',
              cf_method: str = 'matrix_factorization'):
        """
        Train both collaborative filtering and content-based models
        """
        
        print("\n" + "="*50)
        print("TRAINING HYBRID RECOMMENDATION ENGINE")
        print("="*50)
        
        # Train collaborative filtering
        print("\n1. COLLABORATIVE FILTERING")
        print("-" * 30)
        
        try:
            user_item_matrix, users, items = self.cf_engine.prepare_interaction_matrix(
                df, user_col, item_col, rating_col
            )
            
            if cf_method == 'matrix_factorization':
                self.cf_engine.train_matrix_factorization(user_item_matrix)
            elif cf_method == 'user_based':
                self.cf_engine.train_neighborhood_model(user_item_matrix, 'user_based')
            elif cf_method == 'item_based':
                self.cf_engine.train_neighborhood_model(user_item_matrix, 'item_based')
            elif cf_method == 'neural':
                self.cf_engine.train_neural_collaborative_filtering(user_item_matrix)
            
            print("✓ Collaborative filtering trained successfully")
            
        except Exception as e:
            print(f"✗ Collaborative filtering training failed: {e}")
            return False
        
        # Train content-based filtering
        print("\n2. CONTENT-BASED FILTERING")
        print("-" * 30)
        
        try:
            self.cb_engine.prepare_item_features(df, item_col, description_col, price_col)
            self.cb_engine.build_user_profiles(df, user_col, item_col, rating_col)
            
            print("✓ Content-based filtering trained successfully")
            
        except Exception as e:
            print(f"✗ Content-based filtering training failed: {e}")
            return False
        
        self.is_fitted = True
        
        print("\n" + "="*50)
        print("HYBRID RECOMMENDATION ENGINE TRAINING COMPLETED")
        print("="*50)
        
        return True
    
    def get_hybrid_recommendations(self, user_id: Union[str, int],
                                 n_recommendations: int = 10,
                                 df: pd.DataFrame = None) -> List[Tuple[str, float]]:
        """
        Get hybrid recommendations combining CF and CB approaches
        """
        
        if not self.is_fitted:
            raise ValueError("Model not trained yet!")
        
        # Get collaborative filtering recommendations
        try:
            cf_recommendations = self.cf_engine.get_user_recommendations(
                user_id, n_recommendations * 2  # Get more to have options for merging
            )
        except:
            cf_recommendations = []
        
        # Get content-based recommendations
        try:
            cb_recommendations = self.cb_engine.get_content_based_recommendations(
                user_id, n_recommendations * 2, df=df
            )
        except:
            cb_recommendations = []
        
        # Combine recommendations
        hybrid_scores = defaultdict(float)
        
        # Add CF scores
        for item, score in cf_recommendations:
            hybrid_scores[item] += self.cf_weight * score
        
        # Add CB scores
        max_cb_score = max([score for _, score in cb_recommendations], default=1.0)
        for item, score in cb_recommendations:
            # Normalize CB scores to similar scale as CF
            normalized_score = score / max_cb_score if max_cb_score > 0 else 0
            hybrid_scores[item] += self.cb_weight * normalized_score
        
        # Sort by combined score and return top N
        hybrid_recommendations = sorted(
            hybrid_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return hybrid_recommendations[:n_recommendations]
    
    def get_diversified_recommendations(self, user_id: Union[str, int],
                                     n_recommendations: int = 10,
                                     diversity_factor: float = 0.2,
                                     df: pd.DataFrame = None) -> List[Tuple[str, float]]:
        """
        Get diversified recommendations to avoid over-concentration
        """
        
        # Get more recommendations than needed
        candidates = self.get_hybrid_recommendations(
            user_id, n_recommendations * 3, df
        )
        
        if not candidates:
            return []
        
        # Start with top recommendation
        selected = [candidates[0]]
        candidates = candidates[1:]
        
        while len(selected) < n_recommendations and candidates:
            best_candidate = None
            best_score = -1
            
            for candidate_item, candidate_score in candidates:
                # Calculate diversity penalty
                diversity_penalty = 0
                
                if self.cb_engine.item_features is not None:
                    try:
                        candidate_features = self.cb_engine.item_features.loc[candidate_item].values
                        
                        for selected_item, _ in selected:
                            if selected_item in self.cb_engine.item_features.index:
                                selected_features = self.cb_engine.item_features.loc[selected_item].values
                                similarity = cosine_similarity([candidate_features], [selected_features])[0][0]
                                diversity_penalty += similarity
                        
                        diversity_penalty /= len(selected)
                    except:
                        diversity_penalty = 0
                
                # Adjust score with diversity
                adjusted_score = candidate_score * (1 - diversity_factor * diversity_penalty)
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_candidate = (candidate_item, candidate_score)
            
            if best_candidate:
                selected.append(best_candidate)
                candidates = [c for c in candidates if c[0] != best_candidate[0]]
            else:
                break
        
        return selected
    
    def evaluate_recommendations(self, df: pd.DataFrame, 
                               user_col: str = 'CustomerID',
                               item_col: str = 'StockCode',
                               test_ratio: float = 0.2) -> Dict[str, float]:
        """
        Evaluate recommendation quality using holdout test set
        """
        
        print("Evaluating recommendation quality...")
        
        # Split data chronologically
        df_sorted = df.sort_values('InvoiceDate')
        split_idx = int(len(df_sorted) * (1 - test_ratio))
        
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        
        # Get users who appear in both train and test
        train_users = set(train_df[user_col].unique())
        test_users = set(test_df[user_col].unique())
        common_users = train_users.intersection(test_users)
        
        if len(common_users) < 10:
            print("Not enough common users for evaluation")
            return {}
        
        # Sample users for evaluation
        eval_users = np.random.choice(list(common_users), min(100, len(common_users)), replace=False)
        
        # Retrain on training data
        temp_engine = HybridRecommendationEngine(
            self.cf_weight, self.cb_weight, self.random_state
        )
        temp_engine.train(train_df)
        
        # Evaluate
        hit_rates = []
        precisions = []
        recalls = []
        
        for user in eval_users:
            # Get test items for user
            test_items = set(test_df[test_df[user_col] == user][item_col].unique())
            
            if len(test_items) == 0:
                continue
            
            # Get recommendations
            try:
                recommendations = temp_engine.get_hybrid_recommendations(user, 10, train_df)
                rec_items = set([item for item, _ in recommendations])
                
                # Calculate metrics
                hits = len(rec_items.intersection(test_items))
                
                hit_rate = 1 if hits > 0 else 0
                precision = hits / len(rec_items) if len(rec_items) > 0 else 0
                recall = hits / len(test_items) if len(test_items) > 0 else 0
                
                hit_rates.append(hit_rate)
                precisions.append(precision)
                recalls.append(recall)
                
            except:
                continue
        
        # Calculate average metrics
        metrics = {}
        if hit_rates:
            metrics['hit_rate'] = np.mean(hit_rates)
            metrics['precision'] = np.mean(precisions)
            metrics['recall'] = np.mean(recalls)
            
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
            else:
                metrics['f1_score'] = 0
        
        print(f"Evaluation completed on {len(hit_rates)} users")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics


def create_recommendation_dashboard_data(engine: HybridRecommendationEngine, 
                                       df: pd.DataFrame,
                                       top_users: List[str] = None,
                                       n_recommendations: int = 5) -> Dict:
    """
    Create comprehensive recommendation data for dashboard display
    """
    
    if not engine.is_fitted:
        raise ValueError("Recommendation engine not trained!")
    
    dashboard_data = {}
    
    # Get users to analyze
    if top_users is None:
        # Select top users by transaction volume
        user_transactions = df.groupby('CustomerID').size().nlargest(20)
        top_users = user_transactions.index.tolist()
    
    # Generate recommendations for each user
    user_recommendations = {}
    
    for user_id in top_users:
        try:
            # Get different types of recommendations
            hybrid_recs = engine.get_hybrid_recommendations(user_id, n_recommendations, df)
            diversified_recs = engine.get_diversified_recommendations(user_id, n_recommendations, df=df)
            
            # Get user's purchase history
            user_history = df[df['CustomerID'] == user_id].groupby('StockCode').agg({
                'Quantity': 'sum',
                'Description': 'first',
                'UnitPrice': 'mean'
            }).nlargest(5, 'Quantity')
            
            user_recommendations[user_id] = {
                'hybrid_recommendations': hybrid_recs,
                'diversified_recommendations': diversified_recs,
                'purchase_history': user_history.to_dict('records'),
                'total_purchases': len(df[df['CustomerID'] == user_id]),
                'total_spent': df[df['CustomerID'] == user_id]['TotalAmount'].sum()
            }
            
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {e}")
            continue
    
    dashboard_data['user_recommendations'] = user_recommendations
    
    # Global recommendation statistics
    all_items = df['StockCode'].unique()
    item_popularity = df.groupby('StockCode').agg({
        'Quantity': 'sum',
        'CustomerID': 'nunique',
        'Description': 'first'
    }).sort_values('CustomerID', ascending=False)
    
    dashboard_data['popular_items'] = item_popularity.head(20).to_dict('records')
    
    # Recommendation coverage
    recommended_items = set()
    for user_data in user_recommendations.values():
        for item, _ in user_data['hybrid_recommendations']:
            recommended_items.add(item)
    
    dashboard_data['recommendation_stats'] = {
        'total_users_analyzed': len(user_recommendations),
        'total_items_in_catalog': len(all_items),
        'items_recommended': len(recommended_items),
        'catalog_coverage': len(recommended_items) / len(all_items) if len(all_items) > 0 else 0
    }
    
    return dashboard_data


# Example usage functions
def run_recommendation_demo(df: pd.DataFrame) -> Dict:
    """
    Demonstration of the recommendation engine capabilities
    """
    
    print("🚀 RUNNING ULTRA-ADVANCED RECOMMENDATION ENGINE DEMO")
    print("="*60)
    
    # Initialize and train hybrid engine
    engine = HybridRecommendationEngine(cf_weight=0.7, cb_weight=0.3)
    
    # Train the engine
    success = engine.train(
        df=df,
        cf_method='matrix_factorization'  # Try different methods: 'user_based', 'item_based', 'neural'
    )
    
    if not success:
        print("❌ Training failed!")
        return {}
    
    # Get sample users for demo
    top_users = df.groupby('CustomerID').size().nlargest(5).index.tolist()
    
    print(f"\n📋 GENERATING RECOMMENDATIONS FOR TOP {len(top_users)} USERS")
    print("-" * 50)
    
    demo_results = {}
    
    for user_id in top_users:
        print(f"\n👤 User: {user_id}")
        
        # Get hybrid recommendations
        hybrid_recs = engine.get_hybrid_recommendations(user_id, 5, df)
        print("🔀 Hybrid Recommendations:")
        for i, (item, score) in enumerate(hybrid_recs, 1):
            print(f"   {i}. Item {item} (Score: {score:.3f})")
        
        # Get diversified recommendations
        diversified_recs = engine.get_diversified_recommendations(user_id, 5, df=df)
        print("🎯 Diversified Recommendations:")
        for i, (item, score) in enumerate(diversified_recs, 1):
            print(f"   {i}. Item {item} (Score: {score:.3f})")
        
        demo_results[user_id] = {
            'hybrid': hybrid_recs,
            'diversified': diversified_recs
        }
    
    # Evaluate recommendations
    print(f"\n📊 EVALUATING RECOMMENDATION QUALITY")
    print("-" * 35)
    
    evaluation_metrics = engine.evaluate_recommendations(df)
    
    print("\n✅ RECOMMENDATION ENGINE DEMO COMPLETED!")
    
    return {
        'engine': engine,
        'recommendations': demo_results,
        'evaluation': evaluation_metrics
    }


def save_recommendation_model(engine: HybridRecommendationEngine, filepath: str):
    """Save trained recommendation engine"""
    
    with open(filepath, 'wb') as f:
        pickle.dump(engine, f)
    
    print(f"Recommendation engine saved to {filepath}")


def load_recommendation_model(filepath: str) -> HybridRecommendationEngine:
    """Load trained recommendation engine"""
    
    with open(filepath, 'rb') as f:
        engine = pickle.load(f)
    
    print(f"Recommendation engine loaded from {filepath}")
    return engine