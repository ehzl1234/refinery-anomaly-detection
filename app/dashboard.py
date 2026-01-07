"""
Streamlit Dashboard for Real-Time Refinery Anomaly Monitoring.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Page config
st.set_page_config(
    page_title="Refinery Anomaly Monitor",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #e74c3c 0%, #f39c12 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-critical { background: #ffebee; border-left: 4px solid #e74c3c; }
    .alert-warning { background: #fff3e0; border-left: 4px solid #f39c12; }
    .alert-normal { background: #e8f5e9; border-left: 4px solid #2ecc71; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load sensor data."""
    data_path = Path(__file__).parent.parent / "data" / "sensor_data.csv"
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    return df


@st.cache_resource
def load_model():
    """Load trained anomaly detector."""
    try:
        from src.anomaly_detector import AnomalyDetector
        model_path = Path(__file__).parent.parent / "models" / "anomaly_detector.pkl"
        return AnomalyDetector.load(str(model_path))
    except:
        return None


def create_sensor_gauge(value, title, min_val, max_val, warning_threshold, critical_threshold):
    """Create a gauge chart for sensor reading."""
    if value > critical_threshold:
        color = "#e74c3c"
    elif value > warning_threshold:
        color = "#f39c12"
    else:
        color = "#2ecc71"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': color},
            'steps': [
                {'range': [min_val, warning_threshold], 'color': "#e8f5e9"},
                {'range': [warning_threshold, critical_threshold], 'color': "#fff3e0"},
                {'range': [critical_threshold, max_val], 'color': "#ffebee"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 2},
                'thickness': 0.75,
                'value': critical_threshold
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def main():
    st.markdown('<h1 class="main-header">🏭 Refinery Asset Anomaly Monitor</h1>', unsafe_allow_html=True)
    st.markdown("Real-time monitoring of refinery equipment using machine learning anomaly detection.")
    st.markdown("---")
    
    # Load data
    df = load_data()
    detector = load_model()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Data"]
    )
    
    # Anomaly threshold
    sensitivity = st.sidebar.slider("Detection Sensitivity", 0.01, 0.10, 0.05)
    
    # Filter data by time
    if time_range == "Last 24 Hours":
        cutoff = df['timestamp'].max() - timedelta(hours=24)
    elif time_range == "Last 7 Days":
        cutoff = df['timestamp'].max() - timedelta(days=7)
    elif time_range == "Last 30 Days":
        cutoff = df['timestamp'].max() - timedelta(days=30)
    else:
        cutoff = df['timestamp'].min()
    
    filtered_df = df[df['timestamp'] >= cutoff].copy()
    
    # Get latest readings (simulate real-time)
    latest = filtered_df.iloc[-1]
    
    # Current Status Header
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp_status = "🔴" if latest['is_anomaly'] and latest['anomaly_type'] == 'temp_spike' else "🟢"
        st.metric(
            f"{temp_status} Temperature",
            f"{latest['temperature_c']:.1f} °C",
            f"{latest['temperature_c'] - 200:.1f} from baseline"
        )
    
    with col2:
        press_status = "🔴" if latest['is_anomaly'] and latest['anomaly_type'] == 'pressure_drop' else "🟢"
        st.metric(
            f"{press_status} Pressure",
            f"{latest['pressure_bar']:.1f} bar",
            f"{latest['pressure_bar'] - 30:.1f} from baseline"
        )
    
    with col3:
        flow_status = "🔴" if latest['is_anomaly'] and latest['anomaly_type'] == 'flow_blockage' else "🟢"
        st.metric(
            f"{flow_status} Flow Rate",
            f"{latest['flow_rate_m3h']:.0f} m³/h",
            f"{latest['flow_rate_m3h'] - 300:.0f} from baseline"
        )
    
    with col4:
        vib_status = "🔴" if latest['is_anomaly'] and latest['anomaly_type'] == 'vibration_surge' else "🟢"
        st.metric(
            f"{vib_status} Vibration",
            f"{latest['vibration_mms']:.2f} mm/s",
            f"{latest['vibration_mms'] - 2:.2f} from baseline"
        )
    
    st.markdown("---")
    
    # Main content - two columns
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.subheader("📈 Sensor Trends")
        
        # Create multi-line chart with anomaly markers
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=('Temperature (°C)', 'Pressure (bar)', 'Flow Rate (m³/h)', 'Vibration (mm/s)'),
            vertical_spacing=0.08
        )
        
        # Temperature
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['temperature_c'],
                                 mode='lines', name='Temperature', line=dict(color='#e74c3c')), row=1, col=1)
        
        # Pressure
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['pressure_bar'],
                                 mode='lines', name='Pressure', line=dict(color='#3498db')), row=2, col=1)
        
        # Flow Rate
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['flow_rate_m3h'],
                                 mode='lines', name='Flow Rate', line=dict(color='#2ecc71')), row=3, col=1)
        
        # Vibration
        fig.add_trace(go.Scatter(x=filtered_df['timestamp'], y=filtered_df['vibration_mms'],
                                 mode='lines', name='Vibration', line=dict(color='#9b59b6')), row=4, col=1)
        
        # Add anomaly markers
        anomalies = filtered_df[filtered_df['is_anomaly'] == 1]
        for row_idx, (col, color) in enumerate(zip(
            ['temperature_c', 'pressure_bar', 'flow_rate_m3h', 'vibration_mms'],
            ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
        ), 1):
            fig.add_trace(go.Scatter(
                x=anomalies['timestamp'], y=anomalies[col],
                mode='markers', name='Anomaly',
                marker=dict(color='red', size=10, symbol='x'),
                showlegend=(row_idx == 1)
            ), row=row_idx, col=1)
        
        fig.update_layout(height=600, showlegend=False, margin=dict(l=60, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with right_col:
        st.subheader("⚠️ Recent Alerts")
        
        recent_anomalies = filtered_df[filtered_df['is_anomaly'] == 1].tail(10).sort_values('timestamp', ascending=False)
        
        if len(recent_anomalies) > 0:
            for _, row in recent_anomalies.iterrows():
                alert_type = row['anomaly_type'].replace('_', ' ').title()
                time_str = row['timestamp'].strftime('%Y-%m-%d %H:%M')
                
                if 'spike' in row['anomaly_type'] or 'surge' in row['anomaly_type']:
                    st.markdown(f"""
                    <div class="alert-box alert-critical">
                        <strong>🔴 {alert_type}</strong><br>
                        <small>{time_str}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="alert-box alert-warning">
                        <strong>🟡 {alert_type}</strong><br>
                        <small>{time_str}</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-normal">
                <strong>🟢 All Systems Normal</strong><br>
                <small>No anomalies detected</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Stats
        st.subheader("📊 Statistics")
        total = len(filtered_df)
        anomaly_count = filtered_df['is_anomaly'].sum()
        
        st.metric("Total Readings", f"{total:,}")
        st.metric("Anomalies Detected", f"{anomaly_count:,}")
        st.metric("Anomaly Rate", f"{anomaly_count/total:.2%}")
        
        # Anomaly type breakdown
        st.markdown("**Anomaly Breakdown:**")
        type_counts = filtered_df[filtered_df['is_anomaly'] == 1]['anomaly_type'].value_counts()
        for atype, count in type_counts.items():
            st.write(f"• {atype.replace('_', ' ').title()}: {count}")
    
    # Footer
    st.markdown("---")
    st.caption("🏭 Refinery Anomaly Detection System | Built with Python, Scikit-Learn & Streamlit")


if __name__ == "__main__":
    main()
