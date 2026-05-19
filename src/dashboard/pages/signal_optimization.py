import streamlit as st
import pandas as pd
from datetime import datetime
from src.dashboard.utils import format_plotly_fig, THEME_COLORS
from src.optimization.signal_optimizer import TrafficSignalOptimizer

def render_signal_optimization(db_manager):
    st.title("🚦 Intelligent Signal Timing Optimizer")
    st.markdown("Dynamic traffic signal regulation based on relative road occupancy. Replaces fixed static intervals.")
    
    optimizer = TrafficSignalOptimizer(db_manager=db_manager)
    
    # 1. Intersection Simulation Inputs
    st.subheader("🌐 Step 1: Intersection Simulation Controller")
    st.markdown("Set vehicle counts crossing loops on the 4 lanes (North, South, East, West).")
    
    col_n, col_s, col_e, col_w = st.columns(4)
    
    with col_n:
        st.markdown("<h4 style='color:#60a5fa;'>North Lane</h4>", unsafe_allow_html=True)
        n_cars = st.slider("Cars (N)", 0, 150, 45, key="n_cars")
        n_bikes = st.slider("Bikes (N)", 0, 80, 12, key="n_bikes")
        n_buses = st.slider("Buses (N)", 0, 30, 4, key="n_buses")
        n_trucks = st.slider("Trucks (N)", 0, 20, 2, key="n_trucks")
        
    with col_s:
        st.markdown("<h4 style='color:#10b981;'>South Lane</h4>", unsafe_allow_html=True)
        s_cars = st.slider("Cars (S)", 0, 150, 22, key="s_cars")
        s_bikes = st.slider("Bikes (S)", 0, 80, 8, key="s_bikes")
        s_buses = st.slider("Buses (S)", 0, 30, 1, key="s_buses")
        s_trucks = st.slider("Trucks (S)", 0, 20, 0, key="s_trucks")
        
    with col_e:
        st.markdown("<h4 style='color:#f59e0b;'>East Lane</h4>", unsafe_allow_html=True)
        e_cars = st.slider("Cars (E)", 0, 150, 85, key="e_cars")
        e_bikes = st.slider("Bikes (E)", 0, 80, 24, key="e_bikes")
        e_buses = st.slider("Buses (E)", 0, 30, 8, key="e_buses")
        e_trucks = st.slider("Trucks (E)", 0, 20, 5, key="e_trucks")
        
    with col_w:
        st.markdown("<h4 style='color:#ef4444;'>West Lane</h4>", unsafe_allow_html=True)
        w_cars = st.slider("Cars (W)", 0, 150, 35, key="w_cars")
        w_bikes = st.slider("Bikes (W)", 0, 80, 10, key="w_bikes")
        w_buses = st.slider("Buses (W)", 0, 30, 2, key="w_buses")
        w_trucks = st.slider("Trucks (W)", 0, 20, 1, key="w_trucks")
        
    lanes_data = {
        "north": {"car": n_cars, "bike": n_bikes, "bus": n_buses, "truck": n_trucks},
        "south": {"car": s_cars, "bike": s_bikes, "bus": s_buses, "truck": s_trucks},
        "east": {"car": e_cars, "bike": e_bikes, "bus": e_buses, "truck": e_trucks},
        "west": {"car": w_cars, "bike": w_bikes, "bus": w_buses, "truck": w_trucks}
    }
    
    # 2. Optimization trigger
    if st.button("🚦 Compute Optimal Timing", type="primary"):
        st.subheader("💡 Step 2: Timing Schedule Recommendation")
        
        opt_res = optimizer.optimize_signal_timing(lanes_data)
        
        # Display performance metric
        reduction = opt_res["estimated_delay_reduction_pct"]
        st.markdown(
            f"""
            <div class='traffic-card' style='border-left: 5px solid #10b981; background-color: #0f172a;'>
                <span style='font-size:1.1rem; color:#9ca3af;'>ESTIMATED REDUCTION IN COMMUTE WAITING TIME</span>
                <p style='font-size:2.8rem; font-weight:800; color:#10b981; margin:0;'>{reduction}%</p>
                <p style='margin:0; color:#9ca3af;'>Compared to standard static cycles (30 seconds per phase)</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Display green timings as visually distinct columns
        t_col1, t_col2, t_col3, t_col4 = st.columns(4)
        
        sequence_order = opt_res["priority_sequence"]
        green_times = opt_res["recommended_green_times"]
        densities = opt_res["lane_densities"]
        
        # Helper dictionary of column interfaces
        col_map = {"north": t_col1, "south": t_col2, "east": t_col3, "west": t_col4}
        col_names = {"north": "North", "south": "South", "east": "East", "west": "West"}
        col_colors = {"north": "#60a5fa", "south": "#10b981", "east": "#f59e0b", "west": "#ef4444"}
        
        for idx, lane in enumerate(sequence_order):
            target_col = col_map[lane]
            with target_col:
                is_priority = "🏆 Priority" if idx == 0 else f"Phase {idx+1}"
                border_color = col_colors[lane]
                st.markdown(
                    f"""
                    <div class='traffic-card' style='border-top: 4px solid {border_color};'>
                        <span style='font-size:0.85rem; color:#9ca3af;'>{is_priority}</span>
                        <h4 style='margin: 0.2rem 0; color:{border_color};'>{col_names[lane]}</h4>
                        <hr style='margin:0.5rem 0; border-color:#334155;' />
                        <p style='margin:0; font-size:1.5rem; font-weight:bold;'>{green_times[lane]} sec</p>
                        <p style='margin:0; font-size:0.85rem; color:#9ca3af;'>PCU Density: {densities[lane]}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
        # Sequence path text
        path_text = " ➡️ ".join([col_names[l] for l in sequence_order])
        st.markdown(f"**Recommended Phase Sequence Plan:** {path_text}")
        
        # Action recommendations
        st.subheader("📋 Step 3: Mitigation Directives")
        for act in opt_res["actions"]:
            st.markdown(f"- {act}")
            
    st.markdown("---")
    
    # 3. DB Log ledger
    st.subheader("📚 Optimizer Historical Activity Log")
    db_logs = db_manager.get_recent_signal_recommendations(limit=8)
    if db_logs:
        log_records = []
        for row in db_logs:
            lane_str = ", ".join([f"{l}: {d}" for l, d in row["recommended_green_times"].items()])
            log_records.append({
                "Timestamp": row["timestamp"],
                "Lanes PCU (N, S, E, W)": ", ".join([f"{l.capitalize()}:{sum(counts.values())}" for l, counts in row["lane_data"].items()]),
                "Allocated Green (sec)": lane_str,
                "Estimated Delay Reduction": f"{row['estimated_delay_reduction']}%"
            })
        st.table(pd.DataFrame(log_records))
    else:
        st.info("No logs present. Run calculations to seed history.")
        
    # Standard manual override warning
    st.warning("Manual override capability: Police and Emergency vehicle preemption triggers will bypass optimal sequences.")
