"""
Generate synthetic refinery sensor data for anomaly detection.
Simulates temperature, pressure, flow rate, and vibration sensors.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sensor_data(n_samples: int = 10000, anomaly_rate: float = 0.05) -> pd.DataFrame:
    """
    Generate synthetic refinery sensor data with injected anomalies.
    
    Sensors simulated:
    - Temperature (°C): Normal range 150-250
    - Pressure (bar): Normal range 10-50
    - Flow Rate (m³/h): Normal range 100-500
    - Vibration (mm/s): Normal range 0.5-5.0
    """
    np.random.seed(42)
    
    # Generate timestamps (every 5 minutes)
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(minutes=5*i) for i in range(n_samples)]
    
    # Generate normal sensor readings
    temperature = np.random.normal(200, 15, n_samples)  # Mean 200°C, std 15
    pressure = np.random.normal(30, 5, n_samples)  # Mean 30 bar, std 5
    flow_rate = np.random.normal(300, 50, n_samples)  # Mean 300 m³/h, std 50
    vibration = np.random.exponential(2, n_samples)  # Exponential with mean 2
    
    # Add time-based patterns (daily cycles)
    hours = np.array([t.hour for t in timestamps])
    temperature += 10 * np.sin(2 * np.pi * hours / 24)  # Daily temp variation
    flow_rate += 30 * np.sin(2 * np.pi * hours / 24)  # Daily flow variation
    
    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Create anomaly labels
    is_anomaly = np.zeros(n_samples, dtype=int)
    anomaly_type = ['normal'] * n_samples
    
    for i, idx in enumerate(anomaly_indices):
        anomaly_kind = np.random.choice(['temp_spike', 'pressure_drop', 'vibration_surge', 'flow_blockage'])
        is_anomaly[idx] = 1
        anomaly_type[idx] = anomaly_kind
        
        if anomaly_kind == 'temp_spike':
            temperature[idx] += np.random.uniform(50, 100)  # Sudden temp spike
        elif anomaly_kind == 'pressure_drop':
            pressure[idx] -= np.random.uniform(15, 25)  # Pressure drop
        elif anomaly_kind == 'vibration_surge':
            vibration[idx] += np.random.uniform(10, 20)  # Vibration surge
        elif anomaly_kind == 'flow_blockage':
            flow_rate[idx] *= np.random.uniform(0.2, 0.5)  # Flow reduction
    
    # Clip to realistic ranges
    temperature = np.clip(temperature, 100, 350)
    pressure = np.clip(pressure, 1, 60)
    flow_rate = np.clip(flow_rate, 50, 600)
    vibration = np.clip(vibration, 0.1, 25)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature_c': np.round(temperature, 2),
        'pressure_bar': np.round(pressure, 2),
        'flow_rate_m3h': np.round(flow_rate, 2),
        'vibration_mms': np.round(vibration, 2),
        'is_anomaly': is_anomaly,
        'anomaly_type': anomaly_type
    })
    
    # Add derived features
    df['temp_pressure_ratio'] = np.round(df['temperature_c'] / df['pressure_bar'], 3)
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df


def main():
    print("Generating synthetic refinery sensor data...")
    
    # Generate training data
    df = generate_sensor_data(n_samples=10000, anomaly_rate=0.05)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sensor_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Generated {len(df)} samples")
    print(f"Anomaly rate: {df['is_anomaly'].mean():.2%}")
    print(f"Saved to: {output_path}")
    
    # Summary stats
    print("\nSensor Statistics:")
    print(df[['temperature_c', 'pressure_bar', 'flow_rate_m3h', 'vibration_mms']].describe())


if __name__ == "__main__":
    main()
