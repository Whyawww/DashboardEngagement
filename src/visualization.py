import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def plot_correlation_heatmap(data: pd.DataFrame):
    """
    Membuat heatmap korelasi antar fitur termasuk engagement_rate jika tersedia.

    Args:
        data (pd.DataFrame): Data numerik.

    Returns:
        matplotlib.figure.Figure: Heatmap figure.
    """
    correlation = data.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Heatmap Korelasi Fitur Engagement", fontsize=14)
    plt.tight_layout()
    return fig

def plot_pca_clusters(data_pca: pd.DataFrame, labels):
    """
    Membuat scatter plot 2D hasil PCA berdasarkan label cluster.

    Args:
        data_pca (pd.DataFrame): Data hasil PCA (PC1, PC2).
        labels (array-like): Label cluster untuk tiap data point.

    Returns:
        matplotlib.figure.Figure: Scatter plot 2D figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=data_pca.iloc[:, 0],
        y=data_pca.iloc[:, 1],
        hue=labels,
        palette="viridis",
        s=60,
        edgecolor='black'
    )
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Visualisasi Cluster (PCA 2D)', fontsize=14)
    ax.legend(title='Cluster', loc='best')
    plt.tight_layout()
    return fig

def plot_3d_scatter(data: pd.DataFrame, labels):
    """
    Membuat scatter plot 3D untuk views, likes, engagement_rate berdasarkan label cluster.

    Args:
        data (pd.DataFrame): Data asli sebelum PCA.
        labels (array-like): Label cluster untuk tiap data point.

    Returns:
        plotly.graph_objects.Figure: 3D scatter plot figure.
    """
    # Gunakan engagement_rate sebagai dimensi z, bukan comments
    fig = px.scatter_3d(
        data,
        x="views",
        y="likes",
        z="engagement_rate",
        color=labels.astype(str),
        title="Visualisasi Cluster 3D (Views-Likes-Engagement)",
        labels={"color": "Cluster"},
        height=700
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        legend_title_text='Cluster',
        scene=dict(
            xaxis_title='Views',
            yaxis_title='Likes',
            zaxis_title='Engagement Rate'
        )
    )
    return fig