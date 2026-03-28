# -*- coding: utf-8 -*-
"""Qdrant.ipynb

# Main Task

## Sensory Data Generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from qdrant_edge import *
import random


class SensorSimulator:
    def __init__(
        self,
        duration_minutes: int = 60, # Total simulation time in minutes
        sampling_rate_hz: int = 1, # Data points per second
        amplitude: float = 10.0, # Base amplitude of the signal
        frequency: float = 0.01, # Frequency of baseline periodic behavior
        noise_level: float = 0.5, # Magnitude of random noise
        spike_magnitude: float = 5.0, # Magnitude of sudden spikes
        spike_frequency_minutes: int = 10, # How often spikes occur in minutes
        anomaly_magnitude: float = 15.0, # Magnitude of repeated anomaly signatures
        anomaly_duration_seconds: int = 30, # Duration of each anomaly in seconds
        anomaly_frequency_minutes: int = 17, # How often anomalies occur in minutes
        random_seed: int = 42, # Seed for reproducibility
        start_datetime_str: str = None # Custom start datetime string
    ):
        self.duration_minutes = duration_minutes
        self.sampling_rate_hz = sampling_rate_hz
        self.amplitude = amplitude
        self.frequency = frequency
        self.noise_level = noise_level
        self.spike_magnitude = spike_magnitude
        self.spike_frequency_minutes = spike_frequency_minutes
        self.anomaly_magnitude = anomaly_magnitude
        self.anomaly_duration_seconds = anomaly_duration_seconds
        self.anomaly_frequency_minutes = anomaly_frequency_minutes
        self.random_seed = random_seed
        self.start_datetime_str = start_datetime_str

        np.random.seed(self.random_seed)

        self.total_samples = self.duration_minutes * 60 * self.sampling_rate_hz
        self.time = np.arange(self.total_samples) / self.sampling_rate_hz

    def _add_baseline(self, offset: float = 0):
        # Simulates a periodic baseline behavior (e.g., daily cycle)
        return self.amplitude * np.sin(2 * np.pi * self.frequency * self.time / 60) + offset

    def _add_noise(self):
        # Adds random fluctuations to the data
        return np.random.normal(0, self.noise_level, self.total_samples)

    def _add_spikes(self):
        # Adds sudden, short-lived increases or decreases
        spikes = np.zeros(self.total_samples)
        spike_interval_samples = self.spike_frequency_minutes * 60 * self.sampling_rate_hz
        if spike_interval_samples > 0:
            num_spikes = self.total_samples // spike_interval_samples
            for i in range(1, num_spikes + 1):
                spike_idx = i * spike_interval_samples + np.random.randint(-self.sampling_rate_hz * 5, self.sampling_rate_hz * 5)
                if 0 <= spike_idx < self.total_samples:
                    spikes[spike_idx] += self.spike_magnitude * np.random.choice([-1, 1])
        return spikes

    def _add_anomalies(self):
        anomalies = np.zeros(self.total_samples)
        anomaly_interval_samples = self.anomaly_frequency_minutes * 60 * self.sampling_rate_hz
        anomaly_duration_samples = self.anomaly_duration_seconds * self.sampling_rate_hz

        if anomaly_interval_samples > 0 and anomaly_duration_samples > 0:
            num_anomalies = self.total_samples // anomaly_interval_samples
            for i in range(1, num_anomalies + 1):
                start_idx = i * anomaly_interval_samples
                end_idx = min(start_idx + anomaly_duration_samples, self.total_samples)
                if start_idx < self.total_samples:
                    anomalies[start_idx:end_idx] += self.anomaly_magnitude * np.random.choice([-1, 1])
        return anomalies

    def generate_data(self, sensor_type: str = 'Temperature') -> pd.DataFrame:
        # Generate individual components
        baseline_offset = {'Temperature': 20, 'Humidity': 60, 'Vibration': 0, 'Air-quality': 50}.get(sensor_type, 0)
        baseline = self._add_baseline(offset=baseline_offset)
        noise = self._add_noise()
        spikes = self._add_spikes()
        anomalies = self._add_anomalies()

        simulated_data = baseline + noise + spikes + anomalies

        if self.start_datetime_str:
            start_timestamp = pd.to_datetime(self.start_datetime_str)
            time_deltas = pd.to_timedelta(self.time, unit='s')
            timestamps = start_timestamp + time_deltas
        else:
            timestamps = pd.to_datetime(self.time, unit='s')

        df = pd.DataFrame({
            'timestamp': timestamps,
            'sensor_type': sensor_type,
            'value': simulated_data
        })
        return df

print("SensorSimulator class defined successfully.\n")

"""## Running the Script

"""

simulator = SensorSimulator(duration_minutes=7200, sampling_rate_hz=1, start_datetime_str='2026-03-01 00:00:00')

sensor_types = ['Temperature', 'Humidity', 'Vibration', 'Air-quality']
all_sensor_data = []

for sensor_type in sensor_types:
    df_sensor = simulator.generate_data(sensor_type)
    all_sensor_data.append(df_sensor)

df_all_sensors = pd.concat(all_sensor_data, ignore_index=True)

print("Combined DataFrame created successfully with custom start datetime. First 5 rows:\n")
print(df_all_sensors.head(), "\n")
print(f"Shape of the combined DataFrame: {df_all_sensors.shape}\n")

"""## Plotting"""

"""
sns.set_theme(style="whitegrid")

unique_sensor_types = df_all_sensors['sensor_type'].unique()


num_sensors = len(unique_sensor_types)
rows = (num_sensors + 1) // 2
cols = 2

fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, sensor_type in enumerate(unique_sensor_types):
    df_sensor_type = df_all_sensors[df_all_sensors['sensor_type'] == sensor_type]

    ax = axes[i]
    sns.lineplot(data=df_sensor_type, x='timestamp', y='value', ax=ax, label=sensor_type)
    ax.set_title(f'Simulated {sensor_type} Data')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.tick_params(axis='x', rotation=45)
    ax.legend()


for j in range(num_sensors, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
"""
"""## Windowing the features"""

window_size = '1min'
df_all_sensors['timestamp'] = pd.to_datetime(df_all_sensors['timestamp'])
windowed_data = []

for sensor_type in df_all_sensors['sensor_type'].unique():
    df_sensor_type = df_all_sensors[df_all_sensors['sensor_type'] == sensor_type].set_index('timestamp')
    df_window = df_sensor_type['value'].resample(window_size).apply(list).reset_index()
    df_window['sensor_type'] = sensor_type
    df_window.rename(columns={'value': 'window_values'}, inplace=True)
    windowed_data.append(df_window)

df_windowed = pd.concat(windowed_data, ignore_index=True)

print(f"Windowed data created successfully with a window size of {window_size}. First 5 rows:\n")
print(df_windowed.head(), "\n")
print(f"Shape of the windowed DataFrame: {df_windowed.shape}\n")

"""## extracting the features"""


def extract_features(window_values, sampling_rate_hz=1):
    if not window_values:
        return pd.Series({
            'mean_value': np.nan,
            'std_dev': np.nan,
            'min_value': np.nan,
            'max_value': np.nan,
            'median_value': np.nan,
            'iqr_value': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'rms_value': np.nan,
            'peak_to_peak': np.nan,
        })

    values_arr = np.array(window_values)
    num_samples = len(values_arr)

    q1 = np.percentile(values_arr, 25)
    q3 = np.percentile(values_arr, 75)

    # Basic statistical features
    mean_val = np.mean(values_arr)
    std_dev_val = np.std(values_arr)
    min_val = np.min(values_arr)
    max_val = np.max(values_arr)
    median_val = np.median(values_arr)
    iqr_val = q3 - q1
    skewness_val = skew(values_arr)
    kurtosis_val = kurtosis(values_arr)
    rms_val = np.sqrt(np.mean(np.square(values_arr)))
    peak_to_peak_val = max_val - min_val

    return pd.Series({
        'mean_value': mean_val,
        'std_dev': std_dev_val,
        'min_value': min_val,
        'max_value': max_val,
        'median_value': median_val,
        'iqr_value': iqr_val,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val,
        'rms_value': rms_val,
        'peak_to_peak': peak_to_peak_val,
    })

df_features = df_windowed['window_values'].apply(lambda x: extract_features(x, sampling_rate_hz=1))

columns_to_drop = [
    'mean_value', 'std_dev', 'min_value', 'max_value', 'median_value',
    'iqr_value', 'skewness', 'kurtosis', 'feature_vector', 'rms_value', 'peak_to_peak'
]

df_windowed = df_windowed.drop(columns=columns_to_drop, errors='ignore')
df_windowed = pd.concat([df_windowed, df_features], axis=1)

print("Features extracted and added to df_windowed. First 5 rows with new features:\n")
print(df_windowed.head(), "\n")
print(f"Shape of the DataFrame with features: {df_windowed.shape}\n")

"""## normalising the features and adding it to the dataframe with vector representation



"""


feature_columns = ['mean_value', 'std_dev', 'min_value', 'max_value', 'median_value', 'iqr_value', 'skewness', 'kurtosis', 'rms_value', 'peak_to_peak']
scaler = StandardScaler()
df_windowed[feature_columns] = scaler.fit_transform(df_windowed[feature_columns])
df_windowed['feature_vector'] = df_windowed[feature_columns].values.tolist()

print("Feature vectors created and added to df_windowed. First 5 rows with new 'feature_vector' column:\n")
print(df_windowed.head(), "\n")

df_windowed.to_csv('df_windowed.csv', index=False)

"""#Qdrant Edge Implementation"""


SHARD_DIRECTORY = "./qdrant-edge-1"
Path(SHARD_DIRECTORY).mkdir(parents=True, exist_ok=True)

VECTOR_DIMENSION = len(df_windowed['feature_vector'].iloc[0])

config = EdgeConfig(
    vectors=EdgeVectorParams(size=VECTOR_DIMENSION, distance=Distance.Cosine),
    )
edge_shard = EdgeShard.create(SHARD_DIRECTORY, config)

"""## adding the data points with metadata into the qdrant data"""

points = []

sensor_clocks = {s: 0 for s in df_windowed['sensor_type'].unique()}
for index, row in df_windowed.iterrows():
    s_type = row['sensor_type']

    local_minute = sensor_clocks[s_type]

    if local_minute > 0 and local_minute % 17 == 0:
        label = 'Bearing Wear'
    elif local_minute > 0 and local_minute % 10 == 0:
        label = 'Sudden Spike'
    else:
        label = 'Normal'

    sensor_clocks[s_type] += 1

    metadata = {
        'sensor_type': row['sensor_type'],
        'timestamp': row['timestamp'].isoformat(),
        'source_id': random.randint(100, 999),
        'location': random.choice(['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D']),
        'simulation_mode': row.get('simulation_mode', 'standard'),
        'anomaly_label': label
    }

    point = Point(
        id=index,
        vector=row['feature_vector'],
        payload=metadata
    )
    points.append(point)

print(f"Successfully created {len(points)} Point objects.\n")
print("Sample metadata from the first point:\n")
print(points[73].payload, "\n")

edge_shard.update(
    UpdateOperation.upsert_points(
        points
    )
)

print(f"Successfully upserted {len(points)} enriched points to collection '{SHARD_DIRECTORY}'.\n")

count = edge_shard.count(CountRequest(exact=True))
print(f"Total points count: {count}\n")
edge_shard.update(
    UpdateOperation.create_field_index("timestamp", PayloadSchemaType.Datetime)
)

"""## Testing with it"""

def features(live_window_raw):
    print("Raw live window data:\n", live_window_raw, "\n")

    return np.array([
    np.mean(live_window_raw),
    np.std(live_window_raw),
    live_window_raw.min(),
    live_window_raw.max(),
    np.median(live_window_raw),
    np.percentile(live_window_raw, 75) - np.percentile(live_window_raw, 25),
    skew(live_window_raw),
    kurtosis(live_window_raw),
    np.sqrt(np.mean(np.square(live_window_raw))),
    live_window_raw.max() - live_window_raw.min()

]).reshape(1, -1)


def save_search_results(filename, test_name, live_window_raw, live_features, query_vector, results):
    """Save search results to file and print to console"""
    feature_names = ['mean_value', 'std_dev', 'min_value', 'max_value', 'median_value',
                     'iqr_value', 'skewness', 'kurtosis', 'rms_value', 'peak_to_peak']

    # Prepare output text
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"TEST: {test_name}")
    output_lines.append("=" * 80)
    output_lines.append("")

    # Raw window data
    output_lines.append("RAW LIVE WINDOW DATA:")
    output_lines.append(str(live_window_raw))
    output_lines.append("")

    # Features extracted
    output_lines.append("EXTRACTED FEATURES:")
    for i, name in enumerate(feature_names):
        output_lines.append(f"  {name}: {live_features[0][i]:.6f}")
    output_lines.append("")

    # Query vector (scaled)
    output_lines.append("QUERY VECTOR (Scaled):")
    output_lines.append(str(query_vector))
    output_lines.append("")

    # Search results
    output_lines.append("SEARCH RESULTS (Top 10 Matches):")
    output_lines.append("-" * 80)

    for idx, hit in enumerate(results, 1):
        output_lines.append(f"Result #{idx}:")
        output_lines.append(f"  Similarity Score: {hit.score:.6f}")
        output_lines.append(f"  Identified Pattern: {hit.payload['anomaly_label']}")
        output_lines.append(f"  Historical Match Time: {hit.payload['timestamp']}")
        output_lines.append(f"  Sensor Type: {hit.payload['sensor_type']}")
        if 'location' in hit.payload:
            output_lines.append(f"  Location: {hit.payload['location']}")
        output_lines.append(f"  Vector: {hit.vector}")
        output_lines.append("-" * 80)

    output_lines.append("\n")

    # Print to console
    output_text = "\n".join(output_lines)
    print(output_text)

    # Append to file
    with open(filename, 'a') as f:
        f.write(output_text)

"""### Bearing Wear Temperature"""

live_window_raw = np.random.normal(20, 0.5 , 60)
live_window_raw[10:35] += 15
live_features = features(live_window_raw)
query_vector = scaler.transform(live_features)[0].tolist()

result = edge_shard.query(
    QueryRequest(
        query=Query.Nearest(query_vector),
        limit=10,
        with_vector=True,
        with_payload=True,
    )
)

save_search_results('search_results.txt', 'Bearing Wear Temperature',
                   live_window_raw, live_features, query_vector, result)

"""### Humidity - Air-Quality"""

live_window_raw = np.random.normal(60, 0.5 , 60)
live_features = features(live_window_raw)
query_vector = scaler.transform(live_features)[0].tolist()

result = edge_shard.query(
    QueryRequest(
        query=Query.Nearest(query_vector),
        limit=10,
        with_vector=True,
        with_payload=True,
    )
)

save_search_results('search_results.txt', 'Humidity - Air-Quality',
                   live_window_raw, live_features, query_vector, result)

"""###  For the above issues we use filter based approach"""

live_window_raw = np.random.normal(60, 0.5 , 60)
live_features = features(live_window_raw)
query_vector = scaler.transform(live_features)[0].tolist()

search_filter = Filter(
    must=[
        FieldCondition(
            key="sensor_type",
            match=MatchTextAny(text_any="Humidity"),
        ),
    ]
)

result = edge_shard.query(
    QueryRequest(
        query=Query.Nearest(query_vector),
        filter = search_filter,
        limit=10,
        with_vector=True,
        with_payload=True,
    )
)

save_search_results('search_results.txt', 'Filter-based Search (Humidity)',
                   live_window_raw, live_features, query_vector, result)

"""### adding some more filters"""

live_window_raw = np.random.normal(60, 0.5 , 60)
live_window_raw[10:35] += 15
live_features = features(live_window_raw)
query_vector = scaler.transform(live_features)[0].tolist()

#########   Filters   #########
#  Sensor-type = humidity
#  location = Zone_B
#  DateTime in range (00:00:00 to 04:00:00) same date

search_filter = Filter(
    must=[
        FieldCondition(
            key="sensor_type",
            match=MatchTextAny(text_any="Humidity"),
        ),
        FieldCondition(
            key="location",
            match=MatchTextAny(text_any="Zone_B"),
        ),
        FieldCondition(
            key="timestamp",
            range=RangeDateTime(
              gte="2026-03-01T00:00:00Z",
              lte="2026-03-01T04:00:00Z"
    ),
)

    ]
)

result = edge_shard.query(
    QueryRequest(
        query=Query.Nearest(query_vector),
        filter = search_filter,
        limit=10,
        with_vector=True,
        with_payload=True,
    )
)

save_search_results('search_results.txt', 'Advanced Filter (Humidity + Zone_B + DateTime Range)',
                   live_window_raw, live_features, query_vector, result)

