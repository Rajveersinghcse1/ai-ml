"""
Visualization Module
Functions for creating insightful visualizations for customer segmentation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from plotly.graph_objs import Sankey
from datetime import datetime, timedelta
from operator import attrgetter
import warnings
from typing import List, Dict, Optional, Union, Tuple
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import networkx as nx

# Optional advanced visualization imports
try:
    import plotly.graph_objects as go
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Some advanced visualizations will be limited.")

try:
    import bokeh
    from bokeh.plotting import figure, show, output_file
    from bokeh.models import HoverTool, ColorBar, LinearColorMapper
    from bokeh.layouts import row, column
    from bokeh.palettes import Viridis256, Category20
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
    warnings.warn("Bokeh not available. Some advanced visualizations will be limited.")

# Set style
sns.set_style('whitegrid')
plt.style.use('seaborn-v0_8-darkgrid')
warnings.filterwarnings('ignore')

def plot_rfm_distribution(rfm_df, figsize=(15, 5)):
    """
    Plot distribution of RFM metrics
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Recency
    axes[0].hist(rfm_df['Recency'], bins=50, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Recency (days)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Recency Distribution')
    axes[0].axvline(rfm_df['Recency'].median(), color='red', linestyle='--', label=f"Median: {rfm_df['Recency'].median():.0f}")
    axes[0].legend()
    
    # Frequency
    axes[1].hist(rfm_df['Frequency'], bins=50, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('Frequency (purchases)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Frequency Distribution')
    axes[1].axvline(rfm_df['Frequency'].median(), color='red', linestyle='--', label=f"Median: {rfm_df['Frequency'].median():.0f}")
    axes[1].legend()
    
    # Monetary
    axes[2].hist(rfm_df['Monetary'], bins=50, color='salmon', edgecolor='black')
    axes[2].set_xlabel('Monetary (total spent)')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Monetary Distribution')
    axes[2].axvline(rfm_df['Monetary'].median(), color='red', linestyle='--', label=f"Median: ${rfm_df['Monetary'].median():.2f}")
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

def plot_rfm_scatter(rfm_df, segment_col='Segment'):
    """
    Create 3D scatter plot of RFM metrics
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with segments
    segment_col : str
        Name of segment column
    """
    fig = px.scatter_3d(
        rfm_df, 
        x='Recency', 
        y='Frequency', 
        z='Monetary',
        color=segment_col,
        title='Customer Segmentation - 3D RFM Scatter Plot',
        labels={'Recency': 'Recency (days)', 'Frequency': 'Frequency', 'Monetary': 'Monetary ($)'},
        height=700
    )
    fig.show()

def plot_segment_distribution(rfm_df, segment_col='Segment', figsize=(12, 6)):
    """
    Plot segment distribution
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with segments
    segment_col : str
        Name of segment column
    figsize : tuple
        Figure size
    """
    segment_counts = rfm_df[segment_col].value_counts()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Bar chart
    segment_counts.plot(kind='bar', ax=axes[0], color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Segment')
    axes[0].set_ylabel('Number of Customers')
    axes[0].set_title('Customer Segment Distribution')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(segment_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', va='bottom')
    
    # Pie chart
    axes[1].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Customer Segment Percentage')
    
    plt.tight_layout()
    plt.show()

def plot_segment_rfm_comparison(rfm_df, segment_col='Segment', figsize=(15, 5)):
    """
    Compare RFM metrics across segments
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with segments
    segment_col : str
        Name of segment column
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Recency by segment
    rfm_df.boxplot(column='Recency', by=segment_col, ax=axes[0])
    axes[0].set_xlabel('Segment')
    axes[0].set_ylabel('Recency (days)')
    axes[0].set_title('Recency by Segment')
    axes[0].tick_params(axis='x', rotation=45)
    plt.sca(axes[0])
    plt.xticks(rotation=45, ha='right')
    
    # Frequency by segment
    rfm_df.boxplot(column='Frequency', by=segment_col, ax=axes[1])
    axes[1].set_xlabel('Segment')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Frequency by Segment')
    axes[1].tick_params(axis='x', rotation=45)
    plt.sca(axes[1])
    plt.xticks(rotation=45, ha='right')
    
    # Monetary by segment
    rfm_df.boxplot(column='Monetary', by=segment_col, ax=axes[2])
    axes[2].set_xlabel('Segment')
    axes[2].set_ylabel('Monetary ($)')
    axes[2].set_title('Monetary by Segment')
    axes[2].tick_params(axis='x', rotation=45)
    plt.sca(axes[2])
    plt.xticks(rotation=45, ha='right')
    
    # Remove the automatic 'Boxplot grouped by' title
    plt.suptitle('')
    
    plt.tight_layout()
    plt.show()

def plot_cluster_analysis(rfm_df, cluster_col='Cluster', figsize=(12, 8)):
    """
    Visualize cluster analysis results
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with cluster labels
    cluster_col : str
        Name of cluster column
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Scatter: Recency vs Frequency
    for cluster in rfm_df[cluster_col].unique():
        if cluster != -1:  # Exclude noise in DBSCAN
            cluster_data = rfm_df[rfm_df[cluster_col] == cluster]
            axes[0, 0].scatter(cluster_data['Recency'], cluster_data['Frequency'], 
                             label=f'Cluster {cluster}', alpha=0.6)
    axes[0, 0].set_xlabel('Recency (days)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Recency vs Frequency')
    axes[0, 0].legend()
    
    # Scatter: Frequency vs Monetary
    for cluster in rfm_df[cluster_col].unique():
        if cluster != -1:
            cluster_data = rfm_df[rfm_df[cluster_col] == cluster]
            axes[0, 1].scatter(cluster_data['Frequency'], cluster_data['Monetary'], 
                             label=f'Cluster {cluster}', alpha=0.6)
    axes[0, 1].set_xlabel('Frequency')
    axes[0, 1].set_ylabel('Monetary ($)')
    axes[0, 1].set_title('Frequency vs Monetary')
    axes[0, 1].legend()
    
    # Scatter: Recency vs Monetary
    for cluster in rfm_df[cluster_col].unique():
        if cluster != -1:
            cluster_data = rfm_df[rfm_df[cluster_col] == cluster]
            axes[1, 0].scatter(cluster_data['Recency'], cluster_data['Monetary'], 
                             label=f'Cluster {cluster}', alpha=0.6)
    axes[1, 0].set_xlabel('Recency (days)')
    axes[1, 0].set_ylabel('Monetary ($)')
    axes[1, 0].set_title('Recency vs Monetary')
    axes[1, 0].legend()
    
    # Cluster size distribution
    cluster_counts = rfm_df[cluster_col].value_counts().sort_index()
    axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color='steelblue', edgecolor='black')
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Number of Customers')
    axes[1, 1].set_title('Cluster Size Distribution')
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(rfm_df, figsize=(8, 6)):
    """
    Plot correlation heatmap for RFM metrics
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe
    figsize : tuple
        Figure size
    """
    # Select numeric columns
    numeric_cols = ['Recency', 'Frequency', 'Monetary']
    if 'R_Score' in rfm_df.columns:
        numeric_cols.extend(['R_Score', 'F_Score', 'M_Score'])
    
    correlation = rfm_df[numeric_cols].corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('RFM Metrics Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def plot_customer_lifetime_value(rfm_df, segment_col='Segment', top_n=10):
    """
    Plot top customers by lifetime value
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe
    segment_col : str
        Name of segment column
    top_n : int
        Number of top customers to display
    """
    top_customers = rfm_df.nlargest(top_n, 'Monetary')
    
    fig = px.bar(
        top_customers,
        x='CustomerID',
        y='Monetary',
        color=segment_col,
        title=f'Top {top_n} Customers by Lifetime Value',
        labels={'Monetary': 'Total Spent ($)', 'CustomerID': 'Customer ID'}
    )
    fig.show()

def create_interactive_dashboard(rfm_df, segment_col='Segment'):
    """
    Create an interactive dashboard with multiple visualizations
    
    Parameters:
    -----------
    rfm_df : pd.DataFrame
        RFM dataframe with segments
    segment_col : str
        Name of segment column
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Segment Distribution', 'RFM by Segment', 
                       'Recency vs Monetary', 'Frequency Distribution'),
        specs=[[{'type': 'pie'}, {'type': 'bar'}],
               [{'type': 'scatter'}, {'type': 'histogram'}]]
    )
    
    # Segment distribution pie chart
    segment_counts = rfm_df[segment_col].value_counts()
    fig.add_trace(
        go.Pie(labels=segment_counts.index, values=segment_counts.values, name='Segments'),
        row=1, col=1
    )
    
    # RFM summary bar chart
    segment_summary = rfm_df.groupby(segment_col).agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    }).reset_index()
    
    for metric in ['Recency', 'Frequency', 'Monetary']:
        fig.add_trace(
            go.Bar(x=segment_summary[segment_col], y=segment_summary[metric], name=metric),
            row=1, col=2
        )
    
    # Scatter plot
    for segment in rfm_df[segment_col].unique():
        segment_data = rfm_df[rfm_df[segment_col] == segment]
        fig.add_trace(
            go.Scatter(x=segment_data['Recency'], y=segment_data['Monetary'], 
                      mode='markers', name=segment, opacity=0.6),
            row=2, col=1
        )
    
    # Frequency histogram
    fig.add_trace(
        go.Histogram(x=rfm_df['Frequency'], name='Frequency'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, title_text="Customer Segmentation Dashboard")
    fig.show()


class UltraAdvancedVisualization:
    """
    Ultra-advanced visualization engine for customer segmentation analytics
    """
    
    def __init__(self, theme: str = 'plotly', figure_size: Tuple[int, int] = (12, 8)):
        """
        Initialize the advanced visualization engine
        
        Parameters:
        -----------
        theme : str
            Visualization theme: 'plotly', 'seaborn', 'bokeh'
        figure_size : Tuple[int, int]
            Default figure size
        """
        self.theme = theme
        self.figure_size = figure_size
        self.color_palettes = {
            'segments': px.colors.qualitative.Set3,
            'sequential': px.colors.sequential.Viridis,
            'diverging': px.colors.diverging.RdYlBu,
            'categorical': px.colors.qualitative.Plotly
        }
        
    def create_cohort_analysis_heatmap(self, df: pd.DataFrame, 
                                     customer_col: str = 'CustomerID',
                                     date_col: str = 'InvoiceDate',
                                     revenue_col: str = 'TotalAmount',
                                     period_type: str = 'M') -> go.Figure:
        """
        Create advanced cohort analysis heatmap with retention and revenue metrics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        customer_col : str
            Customer ID column name
        date_col : str
            Date column name
        revenue_col : str
            Revenue column name
        period_type : str
            Period type: 'M' (monthly), 'W' (weekly), 'D' (daily)
        """
        
        # Prepare data
        df_cohort = df.copy()
        df_cohort[date_col] = pd.to_datetime(df_cohort[date_col])
        
        # Create cohort tables
        cohort_data = self._create_cohort_table(df_cohort, customer_col, date_col, period_type)
        
        # Create retention and revenue cohort tables
        retention_table, revenue_table = self._calculate_cohort_metrics(
            df_cohort, cohort_data, customer_col, revenue_col
        )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Customer Retention Cohort Analysis', 'Revenue Cohort Analysis'),
            vertical_spacing=0.15
        )
        
        # Retention heatmap
        retention_heatmap = go.Heatmap(
            z=retention_table.values,
            x=[f'Period {i}' for i in range(retention_table.shape[1])],
            y=[f'Cohort {i}' for i in range(retention_table.shape[0])],
            colorscale='RdYlGn',
            showscale=True,
            hovertemplate='Cohort: %{y}<br>Period: %{x}<br>Retention Rate: %{z:.1%}<extra></extra>',
            colorbar=dict(title='Retention Rate', y=0.75, len=0.4)
        )
        
        fig.add_trace(retention_heatmap, row=1, col=1)
        
        # Revenue heatmap
        revenue_heatmap = go.Heatmap(
            z=revenue_table.values,
            x=[f'Period {i}' for i in range(revenue_table.shape[1])],
            y=[f'Cohort {i}' for i in range(revenue_table.shape[0])],
            colorscale='Blues',
            showscale=True,
            hovertemplate='Cohort: %{y}<br>Period: %{x}<br>Avg Revenue: $%{z:.2f}<extra></extra>',
            colorbar=dict(title='Avg Revenue ($)', y=0.25, len=0.4)
        )
        
        fig.add_trace(revenue_heatmap, row=2, col=1)
        
        fig.update_layout(
            height=800,
            title='Advanced Cohort Analysis Dashboard',
            font=dict(size=12)
        )
        
        return fig
    
    def create_customer_journey_map(self, df: pd.DataFrame,
                                  customer_col: str = 'CustomerID',
                                  date_col: str = 'InvoiceDate',
                                  touchpoint_col: str = 'Category',
                                  value_col: str = 'TotalAmount') -> go.Figure:
        """
        Create interactive customer journey map visualization
        
        Parameters:
        -----------
        df : pd.DataFrame
            Transaction dataframe
        customer_col : str
            Customer ID column name
        date_col : str
            Date column name
        touchpoint_col : str
            Touchpoint/category column name
        value_col : str
            Transaction value column name
        """
        
        # Prepare journey data
        journey_data = self._prepare_journey_data(df, customer_col, date_col, touchpoint_col, value_col)
        
        # Create Sankey diagram for customer journey flow
        fig = self._create_journey_sankey(journey_data)
        
        return fig
    
    def create_advanced_sankey_diagram(self, df: pd.DataFrame,
                                     source_col: str,
                                     target_col: str,
                                     value_col: str,
                                     title: str = "Customer Flow Analysis") -> go.Figure:
        """
        Create advanced Sankey diagram for flow analysis
        
        Parameters:
        -----------
        df : pd.DataFrame
            Flow dataframe
        source_col : str
            Source node column
        target_col : str
            Target node column
        value_col : str
            Flow value column
        title : str
            Chart title
        """
        
        # Prepare Sankey data
        sankey_data = self._prepare_sankey_data(df, source_col, target_col, value_col)
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=sankey_data['labels'],
                color=sankey_data['node_colors']
            ),
            link=dict(
                source=sankey_data['source'],
                target=sankey_data['target'],
                value=sankey_data['values'],
                color=sankey_data['link_colors']
            )
        )])
        
        fig.update_layout(
            title_text=title,
            font_size=10,
            height=600
        )
        
        return fig
    
    def create_hierarchical_treemap(self, df: pd.DataFrame,
                                  path_cols: List[str],
                                  value_col: str,
                                  color_col: str = None,
                                  title: str = "Hierarchical Customer Segments") -> go.Figure:
        """
        Create interactive hierarchical treemap visualization
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with hierarchical data
        path_cols : List[str]
            Columns defining the hierarchy path
        value_col : str
            Value column for sizing
        color_col : str
            Column for coloring
        title : str
            Chart title
        """
        
        if color_col is None:
            color_col = value_col
        
        # Create treemap
        fig = px.treemap(
            df,
            path=path_cols,
            values=value_col,
            color=color_col,
            title=title,
            color_continuous_scale='Viridis'
        )
        
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Value: %{value}<br>Color: %{color}<extra></extra>'
        )
        
        fig.update_layout(
            height=600,
            font_size=12
        )
        
        return fig
    
    def create_3d_cluster_visualization(self, df: pd.DataFrame,
                                      feature_cols: List[str],
                                      cluster_col: str = 'Cluster',
                                      method: str = 'pca') -> go.Figure:
        """
        Create advanced 3D cluster visualization with dimensionality reduction
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with features and clusters
        feature_cols : List[str]
            Feature columns for visualization
        cluster_col : str
            Cluster column name
        method : str
            Dimensionality reduction method: 'pca', 'tsne'
        """
        
        # Prepare data
        X = df[feature_cols].values
        clusters = df[cluster_col].values
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=3)
            X_reduced = reducer.fit_transform(X)
            explained_var = reducer.explained_variance_ratio_
            axis_labels = [f'PC{i+1} ({explained_var[i]:.1%})' for i in range(3)]
        elif method == 'tsne':
            reducer = TSNE(n_components=3, random_state=42)
            X_reduced = reducer.fit_transform(X)
            axis_labels = ['t-SNE 1', 't-SNE 2', 't-SNE 3']
        else:
            # Use first 3 features directly
            X_reduced = X[:, :3]
            axis_labels = feature_cols[:3]
        
        # Create 3D scatter plot
        unique_clusters = np.unique(clusters)
        colors = px.colors.qualitative.Plotly
        
        fig = go.Figure()
        
        for i, cluster in enumerate(unique_clusters):
            if cluster == -1:  # Noise points
                cluster_name = 'Noise'
                color = 'gray'
            else:
                cluster_name = f'Cluster {cluster}'
                color = colors[i % len(colors)]
            
            mask = clusters == cluster
            
            fig.add_trace(go.Scatter3d(
                x=X_reduced[mask, 0],
                y=X_reduced[mask, 1],
                z=X_reduced[mask, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                ),
                name=cluster_name,
                hovertemplate=f'<b>{cluster_name}</b><br>' +
                            f'{axis_labels[0]}: %{{x:.2f}}<br>' +
                            f'{axis_labels[1]}: %{{y:.2f}}<br>' +
                            f'{axis_labels[2]}: %{{z:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'3D Cluster Visualization ({method.upper()})',
            scene=dict(
                xaxis_title=axis_labels[0],
                yaxis_title=axis_labels[1],
                zaxis_title=axis_labels[2],
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=700,
            showlegend=True
        )
        
        return fig
    
    def create_network_analysis_plot(self, df: pd.DataFrame,
                                   source_col: str,
                                   target_col: str,
                                   weight_col: str = None,
                                   title: str = "Customer Network Analysis") -> go.Figure:
        """
        Create network analysis visualization for customer relationships
        
        Parameters:
        -----------
        df : pd.DataFrame
            Edge dataframe
        source_col : str
            Source node column
        target_col : str
            Target node column  
        weight_col : str
            Edge weight column
        title : str
            Chart title
        """
        
        try:
            import networkx as nx
        except ImportError:
            print("NetworkX not available. Install with: pip install networkx")
            return None
        
        # Create network graph
        G = nx.Graph()
        
        for _, row in df.iterrows():
            if weight_col:
                G.add_edge(row[source_col], row[target_col], weight=row[weight_col])
            else:
                G.add_edge(row[source_col], row[target_col])
        
        # Calculate layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract node and edge information
        node_trace, edge_trace = self._create_network_traces(G, pos, weight_col)
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Network visualization of customer relationships",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="#888", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
        
        return fig
    
    def create_advanced_correlation_matrix(self, df: pd.DataFrame,
                                         feature_cols: List[str] = None,
                                         method: str = 'pearson',
                                         cluster_features: bool = True) -> go.Figure:
        """
        Create advanced correlation matrix with clustering and annotations
        
        Parameters:
        -----------
        df : pd.DataFrame
            Dataframe with features
        feature_cols : List[str]
            Feature columns to include
        method : str
            Correlation method: 'pearson', 'spearman', 'kendall'
        cluster_features : bool
            Whether to cluster similar features
        """
        
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr(method=method)
        
        # Cluster features if requested
        if cluster_features and len(feature_cols) > 2:
            # Hierarchical clustering of features
            distance_matrix = 1 - np.abs(corr_matrix)
            linkage_matrix = linkage(distance_matrix, method='ward')
            dendrogram_info = dendrogram(linkage_matrix, no_plot=True)
            clustered_order = dendrogram_info['leaves']
            
            # Reorder correlation matrix
            corr_matrix = corr_matrix.iloc[clustered_order, clustered_order]
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            title=f'Advanced Correlation Matrix ({method.title()})',
            color_continuous_scale='RdBu',
            aspect='auto',
            text_auto=True
        )
        
        fig.update_layout(
            height=max(400, len(feature_cols) * 25),
            width=max(400, len(feature_cols) * 25)
        )
        
        return fig
    
    def create_time_series_analysis(self, df: pd.DataFrame,
                                  date_col: str,
                                  value_col: str,
                                  segment_col: str = None,
                                  aggregation: str = 'D',
                                  show_trend: bool = True) -> go.Figure:
        """
        Create advanced time series analysis with trend decomposition
        
        Parameters:
        -----------
        df : pd.DataFrame
            Time series dataframe
        date_col : str
            Date column name
        value_col : str
            Value column name
        segment_col : str
            Segment column for grouping
        aggregation : str
            Time aggregation: 'D', 'W', 'M', 'Q'
        show_trend : bool
            Whether to show trend lines
        """
        
        # Prepare time series data
        ts_data = df.copy()
        ts_data[date_col] = pd.to_datetime(ts_data[date_col])
        ts_data = ts_data.set_index(date_col)
        
        if segment_col:
            # Group by segment
            grouped = ts_data.groupby(segment_col)[value_col].resample(aggregation).sum().reset_index()
            
            fig = px.line(
                grouped,
                x=date_col,
                y=value_col,
                color=segment_col,
                title=f'Time Series Analysis by {segment_col}',
                labels={value_col: value_col.title(), date_col: 'Date'}
            )
        else:
            # Single time series
            aggregated = ts_data[value_col].resample(aggregation).sum().reset_index()
            
            fig = px.line(
                aggregated,
                x=date_col,
                y=value_col,
                title='Time Series Analysis',
                labels={value_col: value_col.title(), date_col: 'Date'}
            )
        
        # Add trend lines if requested
        if show_trend:
            # Add polynomial trend line
            fig.update_traces(line=dict(width=2))
            
            # Add moving average
            if segment_col:
                for segment in grouped[segment_col].unique():
                    segment_data = grouped[grouped[segment_col] == segment]
                    if len(segment_data) > 3:
                        moving_avg = segment_data[value_col].rolling(window=min(7, len(segment_data)//2), center=True).mean()
                        fig.add_trace(go.Scatter(
                            x=segment_data[date_col],
                            y=moving_avg,
                            mode='lines',
                            name=f'{segment} Trend',
                            line=dict(dash='dash', width=1),
                            opacity=0.7
                        ))
        
        fig.update_layout(
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_customer_lifetime_value_analysis(self, clv_df: pd.DataFrame,
                                              segment_col: str = 'Segment',
                                              clv_col: str = 'CLV',
                                              probability_col: str = 'Probability_Alive') -> go.Figure:
        """
        Create comprehensive CLV analysis dashboard
        
        Parameters:
        -----------
        clv_df : pd.DataFrame
            CLV dataframe with predictions
        segment_col : str
            Segment column name
        clv_col : str
            CLV prediction column name
        probability_col : str
            Customer alive probability column name
        """
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CLV Distribution by Segment', 'CLV vs Probability Alive',
                          'Segment CLV Summary', 'Top Value Customers'),
            specs=[[{'type': 'violin'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # CLV distribution by segment (violin plot)
        for i, segment in enumerate(clv_df[segment_col].unique()):
            segment_data = clv_df[clv_df[segment_col] == segment]
            
            fig.add_trace(go.Violin(
                y=segment_data[clv_col],
                name=segment,
                box_visible=True,
                meanline_visible=True,
                showlegend=False
            ), row=1, col=1)
        
        # CLV vs Probability scatter
        colors = px.colors.qualitative.Plotly
        for i, segment in enumerate(clv_df[segment_col].unique()):
            segment_data = clv_df[clv_df[segment_col] == segment]
            
            fig.add_trace(go.Scatter(
                x=segment_data[probability_col],
                y=segment_data[clv_col],
                mode='markers',
                name=segment,
                marker=dict(color=colors[i % len(colors)], opacity=0.6),
                showlegend=False
            ), row=1, col=2)
        
        # Segment CLV summary
        segment_summary = clv_df.groupby(segment_col)[clv_col].agg(['mean', 'median', 'std']).reset_index()
        
        fig.add_trace(go.Bar(
            x=segment_summary[segment_col],
            y=segment_summary['mean'],
            name='Mean CLV',
            showlegend=False
        ), row=2, col=1)
        
        # Top value customers
        top_customers = clv_df.nlargest(10, clv_col)
        
        fig.add_trace(go.Bar(
            x=top_customers.index.astype(str),
            y=top_customers[clv_col],
            name='Top CLV',
            showlegend=False
        ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            title='Customer Lifetime Value Analysis Dashboard',
            showlegend=True
        )
        
        return fig
    
    def create_churn_risk_dashboard(self, churn_df: pd.DataFrame,
                                  churn_prob_col: str = 'Churn_Probability',
                                  segment_col: str = 'Segment',
                                  feature_cols: List[str] = None) -> go.Figure:
        """
        Create comprehensive churn risk analysis dashboard
        
        Parameters:
        -----------
        churn_df : pd.DataFrame
            Churn prediction dataframe
        churn_prob_col : str
            Churn probability column name
        segment_col : str
            Segment column name
        feature_cols : List[str]
            Important feature columns for analysis
        """
        
        if feature_cols is None:
            feature_cols = ['Recency', 'Frequency', 'Monetary']
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Churn Risk Distribution', 'Risk by Segment',
                          'Feature Impact on Churn', 'High Risk Customers'),
            specs=[[{'type': 'histogram'}, {'type': 'box'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # Churn risk distribution
        fig.add_trace(go.Histogram(
            x=churn_df[churn_prob_col],
            nbinsx=30,
            name='Churn Risk Distribution',
            showlegend=False
        ), row=1, col=1)
        
        # Risk by segment
        for segment in churn_df[segment_col].unique():
            segment_data = churn_df[churn_df[segment_col] == segment]
            
            fig.add_trace(go.Box(
                y=segment_data[churn_prob_col],
                name=segment,
                showlegend=False
            ), row=1, col=2)
        
        # Feature impact (correlation with churn risk)
        if feature_cols and all(col in churn_df.columns for col in feature_cols):
            correlations = []
            for feature in feature_cols:
                corr = churn_df[feature].corr(churn_df[churn_prob_col])
                correlations.append(abs(corr))
            
            fig.add_trace(go.Bar(
                x=feature_cols,
                y=correlations,
                name='Feature Impact',
                showlegend=False
            ), row=2, col=1)
        
        # High risk customers scatter
        if feature_cols and len(feature_cols) >= 2:
            high_risk = churn_df[churn_df[churn_prob_col] > 0.7]
            
            fig.add_trace(go.Scatter(
                x=high_risk[feature_cols[0]],
                y=high_risk[feature_cols[1]],
                mode='markers',
                marker=dict(
                    color=high_risk[churn_prob_col],
                    colorscale='Reds',
                    showscale=True,
                    colorbar=dict(title='Churn Risk')
                ),
                name='High Risk Customers',
                showlegend=False
            ), row=2, col=2)
        
        fig.update_layout(
            height=800,
            title='Churn Risk Analysis Dashboard',
            showlegend=True
        )
        
        return fig
    
    def create_recommendation_performance_analysis(self, recommendations_df: pd.DataFrame,
                                                 actual_purchases_df: pd.DataFrame = None) -> go.Figure:
        """
        Create recommendation system performance analysis
        
        Parameters:
        -----------
        recommendations_df : pd.DataFrame
            Recommendations dataframe
        actual_purchases_df : pd.DataFrame
            Actual purchase data for validation
        """
        
        # Create performance metrics visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Recommendation Distribution', 'Category Performance',
                          'User Engagement', 'Success Metrics')
        )
        
        # Recommendation distribution
        if 'Category' in recommendations_df.columns:
            category_counts = recommendations_df['Category'].value_counts()
            
            fig.add_trace(go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                name='Recommendations by Category',
                showlegend=False
            ), row=1, col=1)
        
        # Add more performance visualizations based on available data
        if 'Score' in recommendations_df.columns:
            fig.add_trace(go.Histogram(
                x=recommendations_df['Score'],
                name='Recommendation Scores',
                showlegend=False
            ), row=1, col=2)
        
        fig.update_layout(
            height=800,
            title='Recommendation System Performance Analysis'
        )
        
        return fig
    
    # Helper methods
    def _create_cohort_table(self, df: pd.DataFrame, customer_col: str, 
                           date_col: str, period_type: str) -> pd.DataFrame:
        """Create cohort table for analysis"""
        
        df = df.copy()
        df['OrderPeriod'] = df[date_col].dt.to_period(period_type)
        df['CohortGroup'] = df.groupby(customer_col)[date_col].transform('min').dt.to_period(period_type)
        
        df_cohort = df.groupby(['CohortGroup', 'OrderPeriod']).agg({
            customer_col: 'nunique'
        }).reset_index()
        
        df_cohort['Period'] = (df_cohort['OrderPeriod'] - df_cohort['CohortGroup']).apply(attrgetter('n'))
        
        return df_cohort
    
    def _calculate_cohort_metrics(self, df: pd.DataFrame, cohort_data: pd.DataFrame,
                                customer_col: str, revenue_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate retention and revenue cohort metrics"""
        
        # Calculate cohort sizes
        cohort_sizes = df.groupby('CohortGroup')[customer_col].nunique().reset_index()
        cohort_sizes.columns = ['CohortGroup', 'TotalCustomers']
        
        # Merge with cohort data
        cohort_data = cohort_data.merge(cohort_sizes, on='CohortGroup')
        
        # Calculate retention rates
        cohort_data['RetentionRate'] = cohort_data[customer_col] / cohort_data['TotalCustomers']
        
        # Create retention table
        retention_table = cohort_data.pivot_table(
            index='CohortGroup', 
            columns='Period', 
            values='RetentionRate'
        ).fillna(0)
        
        # Calculate revenue metrics
        revenue_data = df.groupby(['CohortGroup', 'OrderPeriod']).agg({
            revenue_col: 'mean'
        }).reset_index()
        
        revenue_data['Period'] = (revenue_data['OrderPeriod'] - revenue_data['CohortGroup']).apply(attrgetter('n'))
        
        revenue_table = revenue_data.pivot_table(
            index='CohortGroup',
            columns='Period',
            values=revenue_col
        ).fillna(0)
        
        return retention_table, revenue_table
    
    def _prepare_journey_data(self, df: pd.DataFrame, customer_col: str,
                            date_col: str, touchpoint_col: str, value_col: str) -> pd.DataFrame:
        """Prepare customer journey data"""
        
        # Sort by customer and date
        journey_df = df.sort_values([customer_col, date_col])
        
        # Create journey sequences
        journey_df['NextTouchpoint'] = journey_df.groupby(customer_col)[touchpoint_col].shift(-1)
        journey_df['JourneyStep'] = journey_df.groupby(customer_col).cumcount() + 1
        
        # Remove last touchpoints (no next step)
        journey_df = journey_df.dropna(subset=['NextTouchpoint'])
        
        return journey_df
    
    def _create_journey_sankey(self, journey_data: pd.DataFrame) -> go.Figure:
        """Create Sankey diagram for customer journey"""
        
        # Aggregate journey flows
        flows = journey_data.groupby(['Touchpoint', 'NextTouchpoint']).size().reset_index(name='Count')
        
        # Create node list
        all_touchpoints = list(set(flows['Touchpoint'].tolist() + flows['NextTouchpoint'].tolist()))
        node_dict = {touchpoint: i for i, touchpoint in enumerate(all_touchpoints)}
        
        # Prepare Sankey data
        sankey_data = {
            'labels': all_touchpoints,
            'source': [node_dict[tp] for tp in flows['Touchpoint']],
            'target': [node_dict[tp] for tp in flows['NextTouchpoint']],
            'values': flows['Count'].tolist(),
            'node_colors': px.colors.qualitative.Plotly[:len(all_touchpoints)],
            'link_colors': ['rgba(31, 119, 180, 0.4)'] * len(flows)
        }
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=sankey_data['labels'],
                color=sankey_data['node_colors']
            ),
            link=dict(
                source=sankey_data['source'],
                target=sankey_data['target'],
                value=sankey_data['values'],
                color=sankey_data['link_colors']
            )
        )])
        
        fig.update_layout(
            title_text="Customer Journey Flow Analysis",
            font_size=10,
            height=600
        )
        
        return fig
    
    def _prepare_sankey_data(self, df: pd.DataFrame, source_col: str,
                           target_col: str, value_col: str) -> Dict:
        """Prepare data for Sankey diagram"""
        
        # Get unique nodes
        sources = df[source_col].unique()
        targets = df[target_col].unique()
        all_nodes = list(set(list(sources) + list(targets)))
        
        # Create node mapping
        node_dict = {node: i for i, node in enumerate(all_nodes)}
        
        # Prepare Sankey data
        sankey_data = {
            'labels': all_nodes,
            'source': [node_dict[source] for source in df[source_col]],
            'target': [node_dict[target] for target in df[target_col]],
            'values': df[value_col].tolist(),
            'node_colors': px.colors.qualitative.Set3[:len(all_nodes)],
            'link_colors': ['rgba(0,0,255,0.4)'] * len(df)
        }
        
        return sankey_data
    
    def _create_network_traces(self, G, pos: Dict, weight_col: str = None) -> Tuple:
        """Create network visualization traces"""
        
        # Edge trace
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            if weight_col and G.edges[edge].get('weight'):
                edge_info.append(f"Weight: {G.edges[edge]['weight']}")
            else:
                edge_info.append("Connection")
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=0.5, color='#888'),
                               hoverinfo='none',
                               mode='lines')
        
        # Node trace
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
            
            adjacencies = list(G.neighbors(node))
            node_info.append(f'Node: {node}<br>Connections: {len(adjacencies)}')
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               text=node_text,
                               textposition="middle center",
                               hovertext=node_info,
                               marker=dict(showscale=True,
                                         colorscale='YlGnBu',
                                         reversescale=True,
                                         color=[],
                                         size=10,
                                         colorbar=dict(
                                             thickness=15,
                                             len=0.5,
                                             x=0.1,
                                             title="Node Connections"
                                         ),
                                         line=dict(width=2)))
        
        # Color nodes by number of connections
        node_adjacencies = []
        for node in G.nodes():
            node_adjacencies.append(len(list(G.neighbors(node))))
        
        node_trace.marker.color = node_adjacencies
        
        return node_trace, edge_trace
