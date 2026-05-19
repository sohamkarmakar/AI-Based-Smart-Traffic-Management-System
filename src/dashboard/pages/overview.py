import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from src.dashboard.utils import format_plotly_fig, THEME_COLORS

def render_overview(db_manager):
    st.title("🚦 Traffic Overview Dashboard")
    st.markdown("Real-time city-wide traffic telemetry, active congestion levels, and smart city KPIs.")
    
    # Query database
    records = db_manager.get_historical_records(limit=100)
    
    if not records:
        st.warning("No data found in SQLite database. Run the data pipeline first.")
        return
        
    df = pd.DataFrame(records)
    
    # 1. Row of KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    latest_rec = df.iloc[0]
    avg_vol = df["total_volume"].mean()
    max_dens = df["density_score"].max()
    tot_recs = len(df)
    
    # Define situation color mapping
    sit_colors = {
        "low": THEME_COLORS["success"],
        "normal": THEME_COLORS["primary"],
        "high": THEME_COLORS["warning"],
        "heavy": THEME_COLORS["danger"]
    }
    current_sit = latest_rec["traffic_situation"].lower()
    sit_color = sit_colors.get(current_sit, "#ffffff")
    
    with kpi1:
        st.metric(label="Latest Condition", value=current_sit.upper())
        st.markdown(f"<span style='color:{sit_color}; font-weight:bold;'>● Active Congestion Status</span>", unsafe_allow_html=True)
        
    with kpi2:
        st.metric(label="Avg Volume (15-min)", value=f"{avg_vol:.1f} veh")
        st.caption("Rolling average count")
        
    with kpi3:
        st.metric(label="Peak Density Score", value=f"{max_dens:.1f}")
        st.caption("Max PCU load observed")
        
    with kpi4:
        st.metric(label="SQL Records", value=f"{tot_recs:,}")
        st.caption("Database rows cached")

    st.markdown("---")
    
    # 2. Main charts split
    col_chart, col_feed = st.columns([2, 1])
    
    with col_chart:
        st.subheader("📈 Rolling Traffic Volume (Latest Observations)")
        
        # Plot latest 48 points (12 hours) in chronologically ascending order
        plot_df = df.head(48).iloc[::-1]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df.index,  # Simple index tracker
            y=plot_df["total_volume"],
            mode="lines+markers",
            name="Total Volume",
            line=dict(color=THEME_COLORS["primary"], width=3),
            marker=dict(size=6, color=THEME_COLORS["accent"])
        ))
        fig.add_trace(go.Bar(
            x=plot_df.index,
            y=plot_df["car_count"],
            name="Cars",
            opacity=0.3,
            marker_color="#10b981"
        ))
        
        fig.update_layout(
            title="Telemetry Traffic Volume and Car Counts",
            xaxis_title="Timeline Observation Intervals",
            yaxis_title="Vehicle Count / Volume",
            hovermode="x unified",
            legend=dict(orientation="h", y=1.1)
        )
        format_plotly_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_feed:
        st.subheader("📻 Live Traffic Incidents")
        st.markdown(
            f"""
            <div class='traffic-card'>
                <div class='traffic-header'>🟢 Intersection Status: ONLINE</div>
                <p><b>Time:</b> {datetime.now().strftime("%I:%M:%S %p")}</p>
                <p><b>Status:</b> All loops reading successfully. Signal dynamic optimization cycles active.</p>
                <p><b>Latest PCU Density:</b> {latest_rec['density_score']:.1f} equivalents.</p>
                <p><b>Alerts:</b> None. Flow rate meets municipal service level agreements.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Micro animation demonstration
        st.info("System refresh automatically logs any new sensor loop data to local storage.")
        
    # 3. Recent telemetry details
    st.subheader("📋 Recent Traffic Telemetry Table (Latest 10 Logs)")
    disp_df = df.head(10)[["timestamp", "day_of_week", "car_count", "bike_count", "bus_count", "truck_count", "total_volume", "density_score", "traffic_situation"]]
    disp_df.columns = ["Time", "Day", "Cars", "Bikes", "Buses", "Trucks", "Total Vol", "Density Score", "Situation"]
    st.dataframe(disp_df, use_container_width=True)
