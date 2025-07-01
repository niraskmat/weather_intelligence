import json
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def generate_environmental_data(start_date='2025-07-24', seed=42, climate_type='temperate'):
    """
    Generate synthetic environmental sensor data with realistic patterns.
    
    Parameters:
    - start_date: Starting date for data generation (YYYY-MM-DD format)
    - seed: Random seed for reproducibility
    - climate_type: Type of climate to simulate ('temperate', 'tropical', 'arid', 'continental')
    
    Returns:
    - list of dictionaries with timestamp, temperature, humidity, and pressure data
    """
    np.random.seed(seed)
    random.seed(seed)
    
    # Generate hourly timestamps for one year:
    start = datetime.strptime(start_date, '%Y-%m-%d')
    timestamps = [start + timedelta(hours=i) for i in range(8760)]
    
    # Climate-specific parameters:
    if climate_type == 'temperate':
        # Moderate seasonal temperature variation, higher humidity in winter:
        temp_base = 12.0
        temp_seasonal_amplitude = 15.0
        humidity_base = 65
        humidity_seasonal_amplitude = 15
        # Peak humidity in winter:
        humidity_seasonal_phase = -260
    elif climate_type == 'tropical':
        # Small temperature variation, higher humidity in summer (wet season):
        temp_base = 26.0
        temp_seasonal_amplitude = 5.0
        humidity_base = 75
        humidity_seasonal_amplitude = 20
        # Peak humidity in summer:
        humidity_seasonal_phase = -80
    elif climate_type == 'arid':
        # Large temperature variation, lower overall humidity, less seasonal humidity variation:
        temp_base = 18.0
        temp_seasonal_amplitude = 20.0
        humidity_base = 35
        humidity_seasonal_amplitude = 10
        # Slightly higher humidity in winter:
        humidity_seasonal_phase = -260
    elif climate_type == 'continental':
        # Large temperature variation, moderate humidity with summer peak:
        temp_base = 8.0
        temp_seasonal_amplitude = 25.0
        humidity_base = 60
        humidity_seasonal_amplitude = 15
        # Peak humidity in summer:
        humidity_seasonal_phase = -80
    else:
        raise ValueError(f"Unknown climate_type: {climate_type}. Use 'temperate', 'tropical', 'arid', or 'continental'.")
    
    # Initialize data arrays:
    data = []
    
    for i, ts in enumerate(timestamps):
        # Day of year (1-365) for seasonal calculations:
        day_of_year = ts.timetuple().tm_yday
        hour_of_day = ts.hour
        
        # === TEMPERATURE ===

        # Seasonal component: sine wave with peak in summer (day 180):
        seasonal_temp = temp_seasonal_amplitude * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Daily variation: cooler at night, warmer during day:
        daily_variation = 8 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Random noise to simulate sensor uncertainty:
        noise = np.random.normal(0, 2.5)
        
        # Additional random variation for weather variability:
        weather_variability = np.random.normal(0, 1.5)

        # Take all factors into account:
        temperature = temp_base + seasonal_temp + daily_variation + noise + weather_variability
        
        # === HUMIDITY ===

        # Base humidity with climate-specific seasonal variation:
        base_humidity = humidity_base + humidity_seasonal_amplitude * np.sin(2 * np.pi * (day_of_year + humidity_seasonal_phase) / 365)
        
        # Correlation with temperature (inverse relationship, climate-dependent):
        correlation_strength = 0.8 if climate_type in ['temperate', 'continental'] else 0.4
        temp_humidity_correlation = -correlation_strength * (temperature - temp_base) / 30 * 20
        
        # Daily pattern: we simplify and assume that humidity peaks at dawn (6 AM),
        # minimum in mid-afternoon (3 PM):
        daily_humidity_variation = 15 * np.sin(2 * np.pi * (hour_of_day - 6) / 24)
        
        # Random noise:
        humidity_noise = np.random.normal(0, 5)
        
        # Take all factors into account:
        humidity = base_humidity + temp_humidity_correlation + daily_humidity_variation + humidity_noise
        
        # Ensure more realistic bounds for humidity based on climate:
        min_humidity = 5 if climate_type == 'arid' else 15
        max_humidity = 98 if climate_type == 'tropical' else 95
        humidity = np.clip(humidity, min_humidity, max_humidity)
        
        # === AIR PRESSURE===

        # Base atmospheric pressure (hPa, standard atmospheric pressure):
        base_pressure = 1013.25
        
        # Seasonal variation (slightly higher in winter):
        seasonal_pressure = 8 * np.sin(2 * np.pi * (day_of_year - 260) / 365)
        
        # This simulates the multi-day pressure variations that occur as different weather
        # systems move through an area, though real weather patterns are much more irregular
        # and complex than this simple weekly cycle:
        weather_cycle = 12 * np.sin(2 * np.pi * i / (24 * 7))
        
        # Short-term pressure variations (2-day cycles):
        short_term = 5 * np.sin(2 * np.pi * i / (24 * 2))
        
        # Random noise:
        pressure_noise = np.random.normal(0, 3)
        
        # Take all factors into account:
        pressure = base_pressure + seasonal_pressure + weather_cycle + short_term + pressure_noise
        
        # Create data record:
        record = {
            'timestamp': ts.isoformat(),
            'temperature_c': round(float(temperature), 2),
            'humidity_percent': round(float(humidity), 1),
            'air_pressure_hpa': round(float(pressure), 2)
        }
        
        data.append(record)

    # === ANOMALIES ===
    
    # Inject realistic anomalies for statistical detection (0.5% of data):
    anomaly_rate = 0.005
    n_anomalies = int(len(data) * anomaly_rate)
    
    # Temperature anomalies (sensor spikes, equipment malfunction):
    temp_anomaly_idx = np.random.choice(len(data), size=n_anomalies//3, replace=False)
    temp_anomaly_idx.sort()
    print("temperature_anomalies: ", temp_anomaly_idx)
    for idx in temp_anomaly_idx:
        if np.random.random() < 0.5:
            # Hot spike (e.g. equipment malfunction):
            data[idx]['temperature_c'] = round(float(data[idx]['temperature_c'] + np.random.uniform(20, 40)), 2)
        else:
            # Cold spike (e.g. sensor error):
            data[idx]['temperature_c'] = round(float(data[idx]['temperature_c'] - np.random.uniform(15, 30)), 2)
    
    # Humidity anomalies (impossible readings):
    humidity_anomaly_idx = np.random.choice(len(data), size=n_anomalies//3, replace=False)
    humidity_anomaly_idx.sort()
    print("humidity anomalies: ", humidity_anomaly_idx)

    for idx in humidity_anomaly_idx:
        if np.random.random() < 0.7:
            # Humidity spike (over 100%):
            data[idx]['humidity_percent'] = round(float(np.random.uniform(101, 110)), 1)
        else:
            # Humidity drop (near 0%):
            data[idx]['humidity_percent'] = round(float(np.random.uniform(0, 3)), 1)
    
    # Pressure anomalies (extreme readings):
    pressure_anomaly_idx = np.random.choice(len(data), size=n_anomalies//3, replace=False)
    pressure_anomaly_idx.sort()
    print("pressure anomalies: ", pressure_anomaly_idx)

    for idx in pressure_anomaly_idx:
        base_val = data[idx]['air_pressure_hpa']
        if np.random.random() < 0.5:
            # Pressure spike:
            data[idx]['air_pressure_hpa'] = round(float(base_val + np.random.uniform(50, 100)), 2)
        else:
            # Pressure drop:
            data[idx]['air_pressure_hpa'] = round(float(base_val - np.random.uniform(40, 80)), 2)
    
    # === MISSING VALUES ===

    # Simulate sensor malfunctions (2% missing data):
    missing_rate = 0.02
    n_missing = int(len(data) * missing_rate)
    
    # Random missing values for each sensor:
    temp_missing_idx = np.random.choice(len(data), size=n_missing//3, replace=False)
    temp_missing_idx.sort()
    print("temp missing: ", temp_missing_idx)
    humidity_missing_idx = np.random.choice(len(data), size=n_missing//3, replace=False)
    humidity_missing_idx.sort()
    print("humidity missing: ", humidity_missing_idx)
    pressure_missing_idx = np.random.choice(len(data), size=n_missing//3, replace=False)
    pressure_missing_idx.sort()
    print("pressure missing: ", pressure_missing_idx)

    for idx in temp_missing_idx:
        data[idx]['temperature_c'] = None

    for idx in humidity_missing_idx:
        data[idx]['humidity_percent'] = None

    for idx in pressure_missing_idx:
        data[idx]['air_pressure_hpa'] = None
    
    # Occasionally simulate complete sensor outages (all sensors down simultaneously):
    outage_periods = 5

    for _ in range(outage_periods):
        outage_start = np.random.randint(0, len(data) - 24)
        outage_duration = np.random.randint(2, 12)
        outage_end = min(outage_start + outage_duration, len(data))
        print("outage: ", outage_start, outage_end)
        for i in range(outage_start, outage_end):
            data[i]['temperature_c'] = None
            data[i]['humidity_percent'] = None
            data[i]['air_pressure_hpa'] = None
    
    return data

def save_data(data, filename='environmental_sensor_data.json'):
    """Save the generated data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nData saved to {filename}")
    print(f"Total records: {len(data)}")
    print(f"Date range: {data[0]['timestamp']} to {data[-1]['timestamp']}")
    
    # Count missing values:
    missing_temp = sum(1 for record in data if record['temperature_c'] is None)
    missing_humidity = sum(1 for record in data if record['humidity_percent'] is None)
    missing_pressure = sum(1 for record in data if record['air_pressure_hpa'] is None)
    
    print("\nMissing values:")
    print(f"- temperature_c: {missing_temp}")
    print(f"- humidity_percent: {missing_humidity}")
    print(f"- air_pressure_hpa: {missing_pressure}")
    
    total_missing = missing_temp + missing_humidity + missing_pressure
    total_possible = len(data) * 3
    completeness = (1 - total_missing / total_possible) * 100
    print(f"\nData quality: {completeness:.1f}% complete")

def display_sample_stats(data):
    """Display basic statistics of the generated data."""
    print(f"Dataset size: {len(data)} records")
    print(f"Time period: {data[0]['timestamp']} to {data[-1]['timestamp']}")
    
    # Calculate statistics (filtering out None values):
    temperatures = [r['temperature_c'] for r in data if r['temperature_c'] is not None]
    humidities = [r['humidity_percent'] for r in data if r['humidity_percent'] is not None]
    pressures = [r['air_pressure_hpa'] for r in data if r['air_pressure_hpa'] is not None]
    
    print("\n--- Temperature Statistics ---")
    temp_mean = np.mean(temperatures)
    temp_std = np.std(temperatures)
    print(f"Count: {len(temperatures)} (of {len(data)})")
    print(f"Mean: {temp_mean:.1f}°C")
    print(f"Range: {min(temperatures):.1f}°C to {max(temperatures):.1f}°C")
    print(f"Std deviation: {temp_std:.1f}°C")
    
    print("\n--- Humidity Statistics ---")
    hum_mean = np.mean(humidities)
    hum_std = np.std(humidities)
    print(f"Count: {len(humidities)} (of {len(data)})")
    print(f"Mean: {hum_mean:.1f}%")
    print(f"Range: {min(humidities):.1f}% to {max(humidities):.1f}%")
    print(f"Std deviation: {hum_std:.1f}%")
    
    print("\n--- Air Pressure Statistics ---")
    press_mean = np.mean(pressures)
    press_std = np.std(pressures)
    print(f"Count: {len(pressures)} (of {len(data)})")
    print(f"Mean: {press_mean:.1f} hPa")
    print(f"Range: {min(pressures):.1f} hPa to {max(pressures):.1f} hPa")
    print(f"Std deviation: {press_std:.1f} hPa")

if __name__ == "__main__":
    # Generate the environmental data:
    print("Generating environmental sensor data...")
    data = generate_environmental_data(start_date='2024-01-01', seed=42, climate_type='temperate')
    
    # Display statistics:
    display_sample_stats(data)
    
    # Save to JSON:
    save_data(data, '../environmental_sensor_data.json')
    
    # Create a sample plot of the first week:
    week_data = data[:168]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    timestamps = [datetime.fromisoformat(r['timestamp']) for r in week_data]
    temperatures = [r['temperature_c'] for r in week_data if r['temperature_c'] is not None]
    humidities = [r['humidity_percent'] for r in week_data if r['humidity_percent'] is not None]
    pressures = [r['air_pressure_hpa'] for r in week_data if r['air_pressure_hpa'] is not None]
    
    # Filter timestamps to match non-None values:
    temp_times = [timestamps[i] for i, r in enumerate(week_data) if r['temperature_c'] is not None]
    hum_times = [timestamps[i] for i, r in enumerate(week_data) if r['humidity_percent'] is not None]
    press_times = [timestamps[i] for i, r in enumerate(week_data) if r['air_pressure_hpa'] is not None]
    
    axes[0].plot(temp_times, temperatures, 'r-', alpha=0.7)
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].set_title('First Week - Environmental Sensor Data')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(hum_times, humidities, 'b-', alpha=0.7)
    axes[1].set_ylabel('Humidity (%)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(press_times, pressures, 'g-', alpha=0.7)
    axes[2].set_ylabel('Air Pressure (hPa)')
    axes[2].set_xlabel('Time')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_environmental_data.png', dpi=150, bbox_inches='tight')
    print("Sample visualization saved as 'sample_environmental_data.png'")