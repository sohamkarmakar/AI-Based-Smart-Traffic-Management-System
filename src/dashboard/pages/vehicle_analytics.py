import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from src.dashboard.utils import format_plotly_fig, THEME_COLORS

def render_vehicle_analytics(db_manager):
    st.title("🚗 Vehicle Analytics")
    st.markdown("Detailed breakdown of traffic composition and vehicle occupancy trends.")
    
    # Query database
    records = db_manager.get_historical_records(limit=5000)
    if not records:
        st.warning("No data found in SQLite database.")
        return
        
    df = pd.DataFrame(records)
    
    # Parse time to extract Hour if missing
    if "Hour" not in df.columns:
        df["Hour"] = pd.to_datetime(df["timestamp"], format="%I:%M:%S %p", errors='coerce').dt.hour
        df["Hour"] = df["Hour"].fillna(0).astype(int)

    # 1. Sidebar filters specific to this page
    st.sidebar.subheader("Filter Vehicle Analytics")
    days = df["day_of_week"].unique()
    selected_days = st.sidebar.multiselect("Select Days of the week", days, default=days)
    
    # Filter dataset
    filtered_df = df[df["day_of_week"].isin(selected_days)]
    
    if filtered_df.empty:
        st.error("No records match the selected filters.")
        return

    # Sum of counts
    total_cars = filtered_df["car_count"].sum()
    total_bikes = filtered_df["bike_count"].sum()
    total_buses = filtered_df["bus_count"].sum()
    total_trucks = filtered_df["trucks_count"].sum() if "trucks_count" in filtered_df.columns else filtered_df["truck_count"].sum()
    
    # 2. Charts
    col_pie, col_stats = st.columns([1, 1])
    
    with col_pie:
        st.subheader("📊 Vehicle Type Distribution Mix")
        labels = ["Cars", "Bikes", "Buses", "Trucks"]
        values = [total_cars, total_bikes, total_buses, total_trucks]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.45,
            marker=dict(colors=[THEME_COLORS["primary"], THEME_COLORS["success"], THEME_COLORS["warning"], THEME_COLORS["danger"]])
        )])
        fig.update_layout(title="Total Vehicle Count Proportions")
        format_plotly_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    with col_stats:
        st.subheader("💡 Observations & Insights")
        # Compute vehicle percentage
        total_vehicles = sum(values)
        if total_vehicles > 0:
            car_pct = (total_cars / total_vehicles) * 100
            bike_pct = (total_bikes / total_vehicles) * 100
            bus_pct = (total_buses / total_vehicles) * 100
            truck_pct = (total_trucks / total_vehicles) * 100
        else:
            car_pct = bike_pct = bus_pct = truck_pct = 0
            
        st.markdown(
            f"""
            <div class='traffic-card'>
                <p>🟢 <b>Private Transport dominance:</b> Private cars represent <b>{car_pct:.1f}%</b> of total vehicles. This suggests high personal vehicle reliance.</p>
                <p>🔵 <b>Bicycle/Motorcycle share:</b> Commuters using two-wheelers make up <b>{bike_pct:.1f}%</b> of traffic.</p>
                <p>🟡 <b>Public Transport occupancy:</b> Buses form <b>{bus_pct:.1f}%</b> of the mix. Higher bus shares improve commuter throughput.</p>
                <p>🔴 <b>Logistics footprint:</b> Freight/commercial trucks represent <b>{truck_pct:.1f}%</b> of the traffic profile.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("---")
    
    # 3. Stacked Time plots
    st.subheader("🕒 Vehicle Distribution Trends by Hour of Day")
    # Group by Hour
    hourly_df = filtered_df.groupby("Hour")[["car_count", "bike_count", "bus_count", "truck_count"]].mean().reset_index()
    
    fig_time = go.Figure()
    fig_time.add_trace(go.Bar(
        x=hourly_df["Hour"], y=hourly_df["car_count"], name="Cars", marker_color=THEME_COLORS["primary"]
    ))
    fig_time.add_trace(go.Bar(
        x=hourly_df["Hour"], y=hourly_df["bike_count"], name="Bikes", marker_color=THEME_COLORS["success"]
    ))
    fig_time.add_trace(go.Bar(
        x=hourly_df["Hour"], y=hourly_df["bus_count"], name="Buses", marker_color=THEME_COLORS["warning"]
    ))
    fig_time.add_trace(go.Bar(
        x=hourly_df["Hour"], y=hourly_df["truck_count"], name="Trucks", marker_color=THEME_COLORS["danger"]
    ))
    
    fig_time.update_layout(
        barmode="stack",
        xaxis=dict(title="Hour of Day (0-23)", tickmode="linear", tick0=0, dtick=1),
        yaxis=dict(title="Average Vehicle Count"),
        title="Hourly Average Composition of Traffic Lanes"
    )
    format_plotly_fig(fig_time)
    st.plotly_chart(fig_time, use_container_width=True)
