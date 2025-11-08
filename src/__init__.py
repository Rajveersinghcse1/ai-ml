"""
E-Commerce Customer Segmentation Package

This package provides tools for customer segmentation analysis including:
- Data preprocessing and cleaning
- RFM (Recency, Frequency, Monetary) analysis
- Machine learning clustering algorithms
- Visualization and reporting

Author: Customer Analytics Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Customer Analytics Team"

from . import preprocessing
from . import rfm_analysis
from . import clustering
from . import visualization

__all__ = [
    'preprocessing',
    'rfm_analysis',
    'clustering',
    'visualization'
]
