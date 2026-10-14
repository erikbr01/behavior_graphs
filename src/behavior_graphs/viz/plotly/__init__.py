"""Pure Plotly figure modules ported from policy_doctor.

Each module exposes functions that accept preprocessed data and return
``plotly.graph_objects.Figure`` objects — no Streamlit imports. The newer
``behavior_graphs.viz.graph_plot`` / ``behavior_graphs.viz.clusters`` modules
coexist with this package; the ported apps use these legacy-figure modules.

Dropped in the port (policy_doctor-only features): influence/attribution
heatmaps, curation-selection diagnostics, and runtime-monitoring figures.
"""

from behavior_graphs.viz.plotly.behavior_graph import (
    create_advantage_matrix_heatmap,
    create_behavior_graph_plot,
    create_behavior_paths_plot,
    create_episode_q_advantage_timeline,
    create_episode_value_timeline,
    create_interactive_behavior_graph,
    create_node_value_bar_chart,
    create_q_value_distribution_plot,
    create_transition_matrix_heatmap,
    create_value_colored_interactive_graph,
    create_value_heatmap,
)
from behavior_graphs.viz.plotly.behavior_graph_timesteps import (
    create_timestep_colored_interactive_graph,
)
from behavior_graphs.viz.plotly.clusters import (
    CLUSTER_COLORS,
    NOISE_COLOR,
    create_cluster_scatter_2d,
    create_cluster_size_chart,
    create_cluster_timestep_distribution,
    create_confusion_matrix_plot,
    create_label_coherency_chart,
    create_silhouette_plot,
)
from behavior_graphs.viz.plotly.eef_trajectories import (
    create_eef_trajectory_figure,
    create_initial_eef_scatter_2d,
)
from behavior_graphs.viz.plotly.frames import (
    create_action_plot,
    create_annotated_frame,
    create_cluster_timeline,
    create_label_timeline,
    create_state_action_scatter,
)
from behavior_graphs.viz.plotly.heatmaps import (
    create_behavior_pie_chart,
    create_comparison_histograms,
    create_correlation_matrix_heatmap,
    create_density_plots,
    create_embedding_plot,
    create_generic_tsne_plot,
    create_histogram,
    create_temporal_patterns_subplots,
)
from behavior_graphs.viz.plotly.trajectory_tree import (
    create_trajectory_icicle,
    create_trajectory_node_edge_plotly,
    create_trajectory_sunburst,
    create_trajectory_treemap,
)

__all__ = [
    "CLUSTER_COLORS",
    "NOISE_COLOR",
    "create_advantage_matrix_heatmap",
    "create_action_plot",
    "create_annotated_frame",
    "create_behavior_graph_plot",
    "create_behavior_paths_plot",
    "create_behavior_pie_chart",
    "create_cluster_scatter_2d",
    "create_cluster_size_chart",
    "create_cluster_timeline",
    "create_cluster_timestep_distribution",
    "create_comparison_histograms",
    "create_confusion_matrix_plot",
    "create_correlation_matrix_heatmap",
    "create_density_plots",
    "create_eef_trajectory_figure",
    "create_embedding_plot",
    "create_episode_q_advantage_timeline",
    "create_episode_value_timeline",
    "create_generic_tsne_plot",
    "create_histogram",
    "create_initial_eef_scatter_2d",
    "create_interactive_behavior_graph",
    "create_label_coherency_chart",
    "create_label_timeline",
    "create_node_value_bar_chart",
    "create_q_value_distribution_plot",
    "create_silhouette_plot",
    "create_state_action_scatter",
    "create_temporal_patterns_subplots",
    "create_timestep_colored_interactive_graph",
    "create_trajectory_icicle",
    "create_trajectory_node_edge_plotly",
    "create_trajectory_sunburst",
    "create_trajectory_treemap",
    "create_transition_matrix_heatmap",
    "create_value_colored_interactive_graph",
    "create_value_heatmap",
]
