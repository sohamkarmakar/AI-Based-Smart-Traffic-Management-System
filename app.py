import streamlit as st
from src.dashboard.utils import inject_custom_css, get_db_manager
from src.dashboard.pages.overview import render_overview
from src.dashboard.pages.vehicle_analytics import render_vehicle_analytics
from src.dashboard.pages.density_analysis import render_density_analysis
from src.dashboard.pages.prediction import render_prediction
from src.dashboard.pages.signal_optimization import render_signal_optimization
from src.dashboard.pages.historical_trends import render_historical_trends

# 1. Page Configuration
st.set_page_config(
    page_title="AI Smart Traffic Management",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Styling CSS injection
inject_custom_css()

# 3. Connect DB
db_manager = get_db_manager()

# 4. Sidebar Navigation
st.sidebar.markdown(
    """
    <div style='text-align: center; padding: 1rem 0;'>
        <h2 style='color: #60a5fa; margin: 0;'>🚦 Smart City</h2>
        <span style='color: #9ca3af; font-size: 0.95rem;'>Traffic Management Core</span>
    </div>
    <hr style='margin: 0.5rem 0; border-color: #334155;' />
    """, 
    unsafe_allow_html=True
)

st.sidebar.subheader("Navigation")
page_selection = st.sidebar.radio(
    "Select System Module",
    [
        "Traffic Overview",
        "Vehicle Analytics",
        "Traffic Density Analysis",
        "Congestion Prediction",
        "Signal Optimization",
        "Historical Trends"
    ]
)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info(
    "📊 **System Status:** Online\n"
    "🤖 **AI Model:** XGBoost Multi-Class Classifier\n"
    "🔗 **Junction Controller:** Dynamic (PCU loop feedback)\n"
    "💾 **DB Engine:** SQLite Local"
)

# 5. Routing
if page_selection == "Traffic Overview":
    render_overview(db_manager)
elif page_selection == "Vehicle Analytics":
    render_vehicle_analytics(db_manager)
elif page_selection == "Traffic Density Analysis":
    render_density_analysis(db_manager)
elif page_selection == "Congestion Prediction":
    render_prediction(db_manager)
elif page_selection == "Signal Optimization":
    render_signal_optimization(db_manager)
elif page_selection == "Historical Trends":
    render_historical_trends(db_manager)
