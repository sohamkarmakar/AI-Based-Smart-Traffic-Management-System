import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard.utils import format_plotly_fig, THEME_COLORS

def render_density_analysis(db_manager):
    st.title("📊 Traffic Density & Congestion Analysis")
    st.markdown("Quantifying road occupancy through Passenger Car Unit (PCU) density metrics.")
    
    # Query database
    records = db_manager.get_historical_records(limit=5000)
    if not records:
        st.warning("No data found in SQLite database.")
        return
        
    df = pd.DataFrame(records)
    
    # Preprocess
    if "Hour" not in df.columns:
        df["Hour"] = pd.to_datetime(df["timestamp"], format="%I:%M:%S %p", errors='coerce').dt.hour
        df["Hour"] = df["Hour"].fillna(0).astype(int)
        
    # Peak hour flag
    if "IsPeakHour" not in df.columns:
        df["TimeVal"] = df["Hour"]
        df["IsPeakHour"] = (((df["TimeVal"] >= 7) & (df["TimeVal"] <= 9)) | 
                            ((df["TimeVal"] >= 16) & (df["TimeVal"] <= 19))).astype(int)

    # 1. Peak hour comparison
    col_peak, col_days = st.columns(2)
    
    with col_peak:
        st.subheader("⏰ Peak vs Off-Peak Density Loads")
        peak_avg = df.groupby("IsPeakHour")["density_score"].mean().reset_index()
        peak_avg["Phase"] = peak_avg["IsPeakHour"].map({1: "Peak Hours", 0: "Off-Peak Hours"})
        
        fig_peak = px.bar(
            peak_avg, x="Phase", y="density_score",
            color="Phase",
            color_discrete_map={"Peak Hours": THEME_COLORS["danger"], "Off-Peak Hours": THEME_COLORS["success"]},
            labels={"density_score": "Average PCU Density"}
        )
        fig_peak.update_layout(showlegend=False)
        format_plotly_fig(fig_peak)
        st.plotly_chart(fig_peak, use_container_width=True)
        
    with col_days:
        st.subheader("📅 Weekly Density Profiles")
        day_avg = df.groupby("day_of_week")["density_score"].mean().reset_index()
        # Sort days order
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_avg["day_of_week"] = pd.Categorical(day_avg["day_of_week"], categories=day_order, ordered=True)
        day_avg = day_avg.sort_values("day_of_week")
        
        fig_days = px.line(
            day_avg, x="day_of_week", y="density_score",
            markers=True,
            line_shape="spline",
            labels={"density_score": "Avg PCU Density", "day_of_week": "Day of the Week"}
        )
        fig_days.update_traces(line=dict(color=THEME_COLORS["accent"], width=3))
        format_plotly_fig(fig_days)
        st.plotly_chart(fig_days, use_container_width=True)

    st.markdown("---")
    
    # 2. Heatmap Matrix
    st.subheader("🌡️ Hourly Traffic Gridlock Heatmap")
    st.markdown("Visualizing average density score by Hour of Day vs Day of Week. Highlights bottleneck times.")
    
    # Group by DayOfWeek and Hour
    heatmap_data = df.groupby(["day_of_week", "Hour"])["density_score"].mean().reset_index()
    
    # Pivot
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_df = heatmap_data.pivot(index="day_of_week", columns="Hour", values="density_score")
    pivot_df = pivot_df.reindex(day_order)
    
    # Plotly express heatmap
    fig_heat = px.imshow(
        pivot_df,
        labels=dict(x="Hour of Day", y="Day of Week", color="PCU Density"),
        x=pivot_df.columns,
        y=pivot_df.index,
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    
    fig_heat.update_xaxes(side="bottom", dtick=1)
    format_plotly_fig(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True)
