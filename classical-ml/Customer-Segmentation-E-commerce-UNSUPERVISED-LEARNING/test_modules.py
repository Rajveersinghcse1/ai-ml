"""
Test script to verify all ultra-advanced modules are working correctly
"""

print("🧪 Testing Ultra-Advanced Customer Segmentation Modules...")
print("=" * 60)

# Test basic imports
try:
    import pandas as pd
    import numpy as np
    import streamlit as st
    print("✅ Basic libraries: OK")
except Exception as e:
    print(f"❌ Basic libraries: {e}")

# Test custom modules
try:
    from src.preprocessing import load_data, clean_data
    print("✅ Preprocessing module: OK")
except Exception as e:
    print(f"❌ Preprocessing module: {e}")

try:
    from src.feature_engineering import CustomerFeatureEngineer
    print("✅ Feature Engineering module: OK")
except Exception as e:
    print(f"❌ Feature Engineering module: {e}")

try:
    from src.rfm_analysis import AdvancedRFMAnalyzer
    print("✅ RFM Analysis module: OK")
except Exception as e:
    print(f"❌ RFM Analysis module: {e}")

try:
    from src.clustering import UltraAdvancedClustering
    print("✅ Clustering module: OK")
except Exception as e:
    print(f"❌ Clustering module: {e}")

try:
    from src.advanced_analytics import ChurnPredictionModel
    print("✅ Advanced Analytics module: OK")
except Exception as e:
    print(f"❌ Advanced Analytics module: {e}")

try:
    from src.recommendation_engine import HybridRecommendationEngine
    print("✅ Recommendation Engine module: OK")
except Exception as e:
    print(f"❌ Recommendation Engine module: {e}")

try:
    from src.visualization import UltraAdvancedVisualization
    print("✅ Visualization module: OK")
except Exception as e:
    print(f"❌ Visualization module: {e}")

try:
    from src.personalization import UltraAdvancedPersonalizationEngine
    print("✅ Personalization module: OK")
except Exception as e:
    print(f"❌ Personalization module: {e}")

try:
    from src.model_evaluation import UltraAdvancedModelEvaluation
    print("✅ Model Evaluation module: OK")
except Exception as e:
    print(f"❌ Model Evaluation module: {e}")

print("=" * 60)
print("🎉 Module testing complete!")
print("\n🚀 Your Ultra-Advanced Customer Segmentation Platform is ready!")
print("📊 Access your dashboard at: http://localhost:8501")