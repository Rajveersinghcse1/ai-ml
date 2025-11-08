"""
Data Preprocessing Module
Functions for cleaning and preparing e-commerce data for analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    """
    Load e-commerce transaction data
    
    Parameters:
    -----------
    filepath : str
        Path to the data file (CSV, Excel, etc.)
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, encoding='utf-8')
        elif filepath.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def clean_data(df):
    """
    Clean the dataset by handling missing values and duplicates
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataframe
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe
    """
    print("Initial shape:", df.shape)
    print("\nMissing values:\n", df.isnull().sum())
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"\nAfter removing duplicates: {df.shape}")
    
    # Handle missing values in CustomerID (critical field)
    if 'CustomerID' in df.columns:
        df = df.dropna(subset=['CustomerID'])
    elif 'Customer ID' in df.columns:
        df = df.dropna(subset=['Customer ID'])
    
    return df

def parse_dates(df, date_column):
    """
    Parse date columns to datetime format
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with date column
    date_column : str
        Name of the date column
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with parsed dates
    """
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])
    return df

def remove_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Remove outliers from specified columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of column names to check for outliers
    method : str
        Method to use ('iqr' or 'zscore')
    threshold : float
        Threshold for outlier detection
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with outliers removed
    """
    df_clean = df.copy()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df_clean[col]))
            df_clean = df_clean[z_scores < threshold]
    
    print(f"Shape after outlier removal: {df_clean.shape}")
    return df_clean

def create_transaction_features(df):
    """
    Create additional features from transaction data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
    
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional features
    """
    # Add day of week
    if 'InvoiceDate' in df.columns:
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Month'] = df['InvoiceDate'].dt.month
        df['Year'] = df['InvoiceDate'].dt.year
        df['Hour'] = df['InvoiceDate'].dt.hour
    
    # Create total amount if not exists
    if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
        df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
    
    return df

def prepare_for_segmentation(df):
    """
    Prepare data specifically for customer segmentation
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction dataframe
    
    Returns:
    --------
    pd.DataFrame
        Prepared dataframe ready for segmentation
    """
    # Remove cancelled orders (negative quantities)
    if 'Quantity' in df.columns:
        df = df[df['Quantity'] > 0]
    
    # Remove negative prices
    if 'UnitPrice' in df.columns:
        df = df[df['UnitPrice'] > 0]
    
    # Remove records with missing customer information
    customer_col = 'CustomerID' if 'CustomerID' in df.columns else 'Customer ID'
    df = df.dropna(subset=[customer_col])
    
    return df
