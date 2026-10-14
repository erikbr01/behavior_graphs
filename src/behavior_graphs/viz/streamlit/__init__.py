"""Streamlit-side rendering helpers (ported from policy_doctor's streamlit_app).

Strict separation is kept from the old repo: modules here own all ``st.*``
calls and data preprocessing, while ``behavior_graphs.viz.plotly`` holds the
pure figure functions.
"""
