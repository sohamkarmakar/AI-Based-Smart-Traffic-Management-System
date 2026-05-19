import streamlit as st
import pandas as pd
import plotly.express as px
from src.dashboard.utils import format_plotly_fig, THEME_COLORS

def render_historical_trends(db_manager):
    st.title("📜 Historical Traffic Trends & Export")
    st.markdown("Search, query, and download historical city sensor records stored in local SQLite databases.")
    
    # 1. Filters
    st.subheader("🔍 Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        day_of_week = st.selectbox("Filter by Day of Week", ["All", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        
    with col2:
        min_density = st.slider("Minimum PCU Density", min_value=0.0, max_value=200.0, value=0.0, step=5.0)

    # 2. Query DB
    filter_day = None if day_of_week == "All" else day_of_week
    records = db_manager.get_historical_records(limit=10000, day_of_week=filter_day, min_density=min_density)
    
    if not records:
        st.warning("No records found matching the criteria.")
        return
        
    df = pd.DataFrame(records)
    
    st.success(f"Successfully loaded {len(df)} matching records from SQLite database.")
    
    # 3. Chart
    st.subheader("📈 Custom Time Trend Graph")
    y_axis = st.selectbox("Select variable to plot", ["total_volume", "density_score", "car_count", "bike_count", "bus_count", "truck_count"])
    
    fig = px.scatter(
        df, x="timestamp", y=y_axis,
        color="traffic_situation",
        color_discrete_map={"low": THEME_COLORS["success"], "normal": THEME_COLORS["primary"], "high": THEME_COLORS["warning"], "heavy": THEME_COLORS["danger"]},
        labels={"timestamp": "Observation Timestamp", y_axis: y_axis.replace("_", " ").title()}
    )
    format_plotly_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Table view
    st.subheader("📋 Query Results Table")
    # Show subset of columns for readability
    disp_cols = ["timestamp", "day_of_week", "car_count", "bike_count", "bus_count", "truck_count", "total_volume", "density_score", "traffic_situation"]
    st.dataframe(df[disp_cols].head(100), use_container_width=True)
    st.caption("Showing top 100 results. Use the download button below to export all rows.")
    
    # 5. Export as CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset as CSV",
        data=csv_data,
        file_name=f"traffic_telemetry_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
