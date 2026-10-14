"""Generic Plotly heatmap / distribution figures.

Ported from policy_doctor's ``plotting.plotly.heatmaps``. The influence- and
curation-specific figures (influence matrices, curation selection diagnostics,
ranking charts, ...) were policy_doctor-only features and were dropped in the
port; what remains are the generic histogram / correlation / embedding /
distribution figures. No Streamlit imports.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go


def create_correlation_matrix_heatmap(
    correlation_matrix: Any,  # pandas DataFrame
    title: str = "Correlation Matrix",
) -> go.Figure:
    """Create a heatmap for a correlation matrix.

    Args:
        correlation_matrix: pandas DataFrame containing correlation coefficients
        title: Plot title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(correlation_matrix.values, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        height=600,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0),
    )

    return fig


def create_density_plots(
    success_values: np.ndarray,
    failure_values: np.ndarray,
    stat_name: str,
    plot_type: str = "histogram",  # "histogram", "normalized", "cdf"
    title: str | None = None,
) -> go.Figure:
    """Create density plots comparing success vs. failure distributions.

    Args:
        success_values: Array of values for successful rollouts
        failure_values: Array of values for failed rollouts
        stat_name: Name of the statistic being plotted
        plot_type: Type of plot ("histogram", "normalized", or "cdf")
        title: Optional custom title

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    if plot_type == "cdf":
        success_sorted = np.sort(success_values)
        failure_sorted = np.sort(failure_values)

        if len(success_sorted) > 0:
            success_cdf = np.arange(1, len(success_sorted) + 1) / len(success_sorted)
            fig.add_trace(
                go.Scatter(
                    x=success_sorted,
                    y=success_cdf,
                    name="Success",
                    line=dict(color="green", width=2),
                    mode="lines",
                    hovertemplate=f"{stat_name}: %{{x:.6f}}<br>Cumulative Prob: %{{y:.4f}}<extra></extra>",
                )
            )

        if len(failure_sorted) > 0:
            failure_cdf = np.arange(1, len(failure_sorted) + 1) / len(failure_sorted)
            fig.add_trace(
                go.Scatter(
                    x=failure_sorted,
                    y=failure_cdf,
                    name="Failure",
                    line=dict(color="red", width=2),
                    mode="lines",
                    hovertemplate=f"{stat_name}: %{{x:.6f}}<br>Cumulative Prob: %{{y:.4f}}<extra></extra>",
                )
            )

        fig.update_layout(
            title=title or f"Cumulative Distribution: {stat_name}",
            xaxis_title=stat_name,
            yaxis_title="Cumulative Probability",
            hovermode="x unified",
        )
    else:
        histnorm = "probability" if plot_type == "normalized" else None

        fig.add_trace(
            go.Histogram(
                x=success_values,
                name="Success",
                opacity=0.6,
                marker=dict(color="green"),
                nbinsx=50,
                histnorm=histnorm,
            )
        )

        fig.add_trace(
            go.Histogram(
                x=failure_values,
                name="Failure",
                opacity=0.6,
                marker=dict(color="red"),
                nbinsx=50,
                histnorm=histnorm,
            )
        )

        fig.update_layout(
            title=title or f"{'Normalized ' if histnorm else ''}Distribution: {stat_name}",
            xaxis_title=stat_name,
            yaxis_title="Probability" if histnorm else "Count",
            barmode="overlay",
        )

    fig.update_layout(height=500, showlegend=True)
    return fig


def create_embedding_plot(
    embeddings_2d: np.ndarray,
    groups: list[dict],
    title: str = "t-SNE Embeddings",
    height: int = 600,
) -> go.Figure:
    """Create an interactive scatter plot for 2D embeddings.

    Args:
        embeddings_2d: 2D array of shape (N, 2)
        groups: List of dictionaries, each containing:
            - 'name': Group name for legend
            - 'indices': Indices of points in this group
            - 'color': Color for this group
            - 'hover_texts': List of hover text strings for points in this group
        title: Plot title
        height: Plot height

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    for group in groups:
        indices = group["indices"]
        name = group["name"]
        color = group["color"]
        hover_texts = group["hover_texts"]

        fig.add_trace(
            go.Scatter(
                x=embeddings_2d[indices, 0],
                y=embeddings_2d[indices, 1],
                mode="markers",
                name=name,
                marker=dict(
                    size=10,
                    color=color,
                    line=dict(width=1, color="white"),
                    opacity=0.7,
                ),
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        height=height,
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
    )

    return fig


def create_behavior_pie_chart(
    labels: list[str],
    values: list[int],
    title: str = "Behavior Distribution",
) -> go.Figure:
    """Create a pie chart for behavior distribution.

    Args:
        labels: Group labels
        values: Group values
        title: Plot title

    Returns:
        Plotly Figure object
    """
    import plotly.express as px

    fig = px.pie(
        names=labels,
        values=values,
        title=title,
        hole=0.3,  # Donut chart
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(showlegend=True)

    return fig


def create_histogram(
    values: np.ndarray,
    title: str = "Histogram",
    xaxis_title: str = "Value",
    yaxis_title: str = "Count",
    color: str = "blue",
    nbins: int = 30,
    vline_at: float | None = None,
    vline_label: str | None = None,
) -> go.Figure:
    """Create a simple histogram with optional vertical line.

    Args:
        values: Array of values
        title: Plot title
        xaxis_title: X-axis title
        yaxis_title: Y-axis title
        color: Bar color
        nbins: Number of bins
        vline_at: Optional x-position for vertical line
        vline_label: Optional label for vertical line

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=nbins,
            marker=dict(color=color, opacity=0.7),
        )
    )

    if vline_at is not None:
        fig.add_vline(
            x=vline_at,
            line_dash="dash",
            line_color="white",
            annotation_text=vline_label or f"Line at {vline_at:.4f}",
            annotation_position="top right",
        )

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=400,
    )

    return fig


def create_temporal_patterns_subplots(
    data_list: list[dict],
    title: str = "Temporal Patterns",
    subplot_titles: tuple[str, str] = ("Left", "Right"),
) -> go.Figure:
    """Create side-by-side temporal pattern plots.

    Args:
        data_list: List of dicts, each with 'x', 'y', 'name', 'col' (1 or 2)
        title: Main plot title
        subplot_titles: Titles for the two subplots

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=subplot_titles,
    )

    for entry in data_list:
        fig.add_trace(
            go.Scatter(
                x=entry["x"],
                y=entry["y"],
                mode="lines",
                name=entry["name"],
                opacity=0.6,
            ),
            row=1,
            col=entry["col"],
        )

    fig.update_layout(
        height=500,
        showlegend=True,
        title_text=title,
    )

    return fig


def create_comparison_histograms(
    train_data: np.ndarray,
    rollout_data: np.ndarray,
    labels: list[str],
    title: str = "Distribution Comparison",
) -> go.Figure:
    """Create a grid of histograms comparing two datasets.

    Args:
        train_data: Array of shape (n_samples, n_dims)
        rollout_data: Array of shape (n_samples, n_dims)
        labels: Labels for each dimension
        title: Overall plot title

    Returns:
        Plotly Figure object
    """
    from plotly.subplots import make_subplots

    num_dims = train_data.shape[1]
    cols = 3
    rows = (num_dims + cols - 1) // cols

    max_vertical_spacing = 1.0 / (rows - 1) if rows > 1 else 0.1
    vertical_spacing = min(0.15, max_vertical_spacing * 0.9)
    horizontal_spacing = 0.1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=labels,
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    for i in range(num_dims):
        row = i // cols + 1
        col = i % cols + 1

        fig.add_trace(
            go.Histogram(
                x=train_data[:, i],
                name="Training",
                marker_color="blue",
                opacity=0.6,
                nbinsx=50,
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Histogram(
                x=rollout_data[:, i],
                name="Rollout",
                marker_color="red",
                opacity=0.6,
                nbinsx=50,
                showlegend=(i == 0),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        title=title,
        height=300 * rows,
        barmode="overlay",
        showlegend=True,
    )

    return fig


def create_generic_tsne_plot(
    embeddings_2d: np.ndarray,
    metadata: list[dict],
    color_by: str = "mean_influence",
    title: str = "t-SNE Visualization",
    categorical_color: bool = False,
) -> go.Figure:
    """Create a generic t-SNE scatter plot.

    Args:
        embeddings_2d: Array of shape (n_samples, 2)
        metadata: List of metadata dicts
        color_by: Key in metadata for coloring
        title: Plot title
        categorical_color: Whether color_by is a categorical variable

    Returns:
        Plotly Figure object
    """
    if categorical_color:
        unique_labels = sorted(set(m[color_by] for m in metadata))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        color_values = [label_to_idx[m[color_by]] for m in metadata]
        colorscale = "Viridis"
        showscale = False
    else:
        color_values = [m.get(color_by, 0) for m in metadata]
        colorscale = [[0, "red"], [0.5, "white"], [1, "green"]]
        showscale = True

    hover_texts = []
    for m in metadata:
        lines = [f"{k}: {v}" for k, v in m.items()]
        hover_texts.append("<br>".join(lines))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=embeddings_2d[:, 0],
            y=embeddings_2d[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color=color_values,
                colorscale=colorscale,
                showscale=showscale,
                colorbar=dict(title=color_by) if showscale else None,
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        hovermode="closest",
        height=600,
    )

    return fig
