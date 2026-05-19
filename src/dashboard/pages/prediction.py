import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from src.dashboard.utils import get_predictor, get_db_manager, format_plotly_fig, THEME_COLORS

def render_prediction(db_manager):
    st.title("🔮 AI Congestion & Volume Prediction")
    st.markdown("Use machine learning models to estimate road congestion levels and forecast volume counts.")
    
    predictor = get_predictor()
    
    # Load training metrics
    report_path = Path("outputs/reports/model_training_report.json")
    metrics = None
    if report_path.exists():
        with open(report_path, "r") as f:
            metrics = json.load(f)

    # 1. Prediction Inputs Form
    st.subheader("🛠️ Step 1: Input Real-Time Road Sensor Telemetry")
    col1, col2 = st.columns(2)
    
    with col1:
        car_count = st.slider("Car Count", min_value=0, max_value=250, value=75, step=1)
        bike_count = st.slider("Bike Count", min_value=0, max_value=100, value=15, step=1)
        bus_count = st.slider("BusCount", min_value=0, max_value=60, value=5, step=1)
        truck_count = st.slider("TruckCount", min_value=0, max_value=50, value=4, step=1)
        
    with col2:
        day_name = st.selectbox("Day of the week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        hour = st.slider("Time of day (Hour)", min_value=0, max_value=23, value=8, step=1)
        minute = st.selectbox("Minute interval", [0, 15, 30, 45], index=0)
        
        clf_model = st.selectbox("Select Classifier", ["xgboost", "random_forest", "gradient_boosting", "logistic_regression"], index=0)
        reg_model = st.selectbox("Select Regressor", ["xgboost", "random_forest", "ridge"], index=0)

    # Pre-calculate features
    # PCU calculations
    weights = config = {
        "car": 1.0,
        "bike": 0.2,
        "bus": 2.5,
        "truck": 3.0
    }
    density_score = (
        car_count * weights["car"] +
        bike_count * weights["bike"] +
        bus_count * weights["bus"] +
        truck_count * weights["truck"]
    )
    
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    day_num = day_mapping[day_name]
    is_weekend = 1 if day_num in [5, 6] else 0
    time_val = hour + (minute / 60.0)
    is_peak = 1 if ((7.0 <= time_val <= 9.5) or (16.5 <= time_val <= 19.5)) else 0
    
    input_features = {
        "CarCount": car_count,
        "BikeCount": bike_count,
        "BusCount": bus_count,
        "TruckCount": truck_count,
        "Hour": hour,
        "DayOfWeek": day_num,
        "IsWeekend": is_weekend,
        "IsPeakHour": is_peak,
        "DensityScore": density_score
    }
    
    st.markdown(f"Computed PCU Occupancy / Density Score: **{density_score:.2f} equivalents**")
    
    # 2. Run button
    if st.button("🔮 Run Traffic Prediction", type="primary"):
        st.subheader("🎯 Step 2: Prediction Results")
        
        try:
            res = predictor.run_inference_and_log(input_features, clf_model=clf_model, reg_model=reg_model)
            
            res_col1, res_col2 = st.columns(2)
            
            # Situation result card
            situation_upper = res["situation"].upper()
            sit_colors = {
                "LOW": THEME_COLORS["success"],
                "NORMAL": THEME_COLORS["primary"],
                "HIGH": THEME_COLORS["warning"],
                "HEAVY": THEME_COLORS["danger"]
            }
            sit_color = sit_colors.get(situation_upper, "#ffffff")
            
            with res_col1:
                st.markdown(
                    f"""
                    <div class='traffic-card' style='border-left: 5px solid {sit_color};'>
                        <div class='traffic-header'>🤖 Congestion Level Classifier</div>
                        <p style='font-size:1.8rem; margin:0; color:{sit_color}; font-weight:bold;'>{situation_upper}</p>
                        <p>Confidence Probability: <b>{res['confidence'] * 100:.1f}%</b></p>
                        <p>Model: <i>{clf_model}</i></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
            with res_col2:
                # Regressed volume result card
                st.markdown(
                    f"""
                    <div class='traffic-card' style='border-left: 5px solid {THEME_COLORS["accent"]};'>
                        <div class='traffic-header'>📉 Volume Forecaster</div>
                        <p style='font-size:1.8rem; margin:0; color:{THEME_COLORS["accent"]}; font-weight:bold;'>{int(res['volume'])} Vehicles</p>
                        <p>Forecasted 15-min Passenger Load</p>
                        <p>Model: <i>{reg_model}</i></p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            st.success("Predictions successfully processed and saved to database audit ledger.")
            
        except Exception as e:
            st.error(f"Inference pipeline failed: {e}. Check if models are trained.")

    st.markdown("---")
    
    # 3. Model performance reporting
    st.subheader("📊 Model Performance & Leaderboard")
    if metrics:
        tab_clf, tab_reg, tab_importance = st.tabs(["Congestion Classifier", "Volume Regressor", "Feature Importance"])
        
        with tab_clf:
            st.markdown("#### Classifier Model Scores (Validation Dataset)")
            clf_df = pd.DataFrame(metrics["classification"]).T[[
                "accuracy_test", "accuracy_validation", "precision_weighted", "recall_weighted", "f1_weighted"
            ]]
            clf_df.columns = ["Test Acc", "Val Acc", "Weighted Prec", "Weighted Rec", "Weighted F1"]
            st.table(clf_df.style.highlight_max(color="#1e3a8a", axis=0))
            st.caption("Training models mapped to 4 classes (low, normal, high, heavy) on TrafficTwoMonth.csv.")
            
        with tab_reg:
            st.markdown("#### Regressor Model Scores (Validation Dataset)")
            reg_df = pd.DataFrame(metrics["regression"]).T[[
                "rmse_test", "rmse_validation", "mae_test", "r2_test"
            ]]
            reg_df.columns = ["Test RMSE", "Val RMSE", "Test MAE", "Test R2"]
            st.table(reg_df.style.highlight_min(color="#1e3a8a", subset=["Test RMSE", "Val RMSE", "Test MAE"], axis=0))
            st.caption("Target variable: Total traffic count.")
            
        with tab_importance:
            st.markdown("#### Relative Feature Importances (XGBoost Classifier)")
            imp_dict = metrics["classification"]["xgboost"]["feature_importance"]
            if imp_dict:
                imp_df = pd.DataFrame(list(imp_dict.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
                
                fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[THEME_COLORS["accent"]])
                fig.update_layout(title="Which signals impact predictions the most?")
                format_plotly_fig(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No feature importances saved.")
    else:
        st.info("Train metrics not loaded. Run train_model.py to see leaderboard performance.")
