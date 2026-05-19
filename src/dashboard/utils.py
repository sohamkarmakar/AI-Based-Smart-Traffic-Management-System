import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.database.db_manager import DatabaseManager
from src.prediction.predict import TrafficPredictor

# Dark-theme color palette
THEME_COLORS = {
    "background": "#0e1117",
    "card": "#1f2937",
    "primary": "#3b82f6",      # Blue
    "success": "#10b981",      # Emerald Green
    "warning": "#f59e0b",      # Amber Yellow
    "danger": "#ef4444",       # Rose Red
    "accent": "#8b5cf6"        # Violet Purple
}

def inject_custom_css():
    """Injects custom CSS to style the Streamlit dashboard with a premium glassmorphic dark theme."""
    st.markdown(
        """
        <style>
        /* General background */
        .stApp {
            background-color: #0e1117;
            color: #f3f4f6;
        }
        
        /* Metric cards styling */
        div[data-testid="stMetricValue"] {
            font-size: 2.2rem !important;
            font-weight: 700 !important;
            color: #60a5fa !important;
        }
        
        div[data-testid="stMetricLabel"] {
            font-size: 0.95rem !important;
            font-weight: 500 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #9ca3af !important;
        }
        
        /* Container boxes / cards */
        .traffic-card {
            background-color: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }
        
        .traffic-header {
            font-size: 1.25rem;
            font-weight: 600;
            color: #38bdf8;
            margin-bottom: 1rem;
            border-bottom: 1px solid #334155;
            padding-bottom: 0.5rem;
        }
        
        /* Signal lights */
        .light-box {
            display: inline-block;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            margin-right: 10px;
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        }
        
        .light-red {
            background-color: #ef4444;
            box-shadow: 0 0 20px #f43f5e;
        }
        
        .light-yellow {
            background-color: #eab308;
            box-shadow: 0 0 20px #eab308;
        }
        
        .light-green {
            background-color: #10b981;
            box-shadow: 0 0 20px #34d399;
        }
        
        /* Interactive tables */
        .dataframe {
            background-color: #1f2937 !important;
            color: #f3f4f6 !important;
            border-collapse: collapse;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

@st.cache_resource
def get_db_manager() -> DatabaseManager:
    """Singleton cached database manager."""
    return DatabaseManager()

@st.cache_resource
def get_predictor() -> TrafficPredictor:
    """Singleton cached predictor."""
    return TrafficPredictor()

def format_plotly_fig(fig):
    """Formats standard Plotly figures to align with the dashboard's premium dark layout."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#d1d5db",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        legend=dict(bgcolor="rgba(15, 23, 42, 0.6)", bordercolor="#334155")
    )
    return fig
