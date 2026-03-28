This `README.md` provides a comprehensive guide to setting up an Ubuntu development environment on VirtualBox and running the RAG-based sensor retrieval scripts using Qdrant Edge.

-----

# RAG for Edge Devices: Sensor Retrieval with Qdrant Edge

This project demonstrates a "simulation-first" approach to building a local sensor-pattern search engine. It uses an Ubuntu environment to simulate high-frequency sensor data, featurize it, and perform low-latency vector searches using **Qdrant Edge**.

## Table of Contents

1.  [Prerequisites](https://www.google.com/search?q=%231-prerequisites)
2.  [Step 1: Ubuntu Setup on VirtualBox](https://www.google.com/search?q=%23step-1-ubuntu-setup-on-virtualbox)
3.  [Step 2: Python Environment Setup](https://www.google.com/search?q=%23step-2-python-environment-setup)
4.  [Step 3: Running the Scripts](https://www.google.com/search?q=%23step-3-running-the-scripts)
5.  [Project Architecture](https://www.google.com/search?q=%23project-architecture)
6.  [Results & Analysis](https://www.google.com/search?q=%23results--analysis)

-----

## 1\. Prerequisites

  - **Host Machine**: Windows, macOS, or Linux.
  - **VirtualBox**: [Download and install VirtualBox](https://www.virtualbox.org/wiki/Downloads).
  - **Ubuntu ISO**: [Download Ubuntu Desktop 22.04 or 24.04 LTS](https://www.google.com/search?q=https://ubuntu.com/download/desktop).

-----

## 2\. Step 1: Ubuntu Setup on VirtualBox

To ensure a stable environment, we use Ubuntu as our edge-simulation substrate.

### Creating the Virtual Machine

1.  **New VM**: Open VirtualBox and click **New**.
2.  **Resources**: Assign at least **4GB RAM** (8GB recommended) and **2-4 CPUs**.
3.  **Storage**: Create a virtual hard disk of at least **25GB**.
4.  **ISO Mounting**: In **Settings \> Storage**, click the empty optical drive and select your downloaded Ubuntu ISO.
5.  **Installation**: Start the VM and follow the on-screen Ubuntu installation prompts.

### Post-Installation Optimization

1.  **Guest Additions**: Go to **Devices \> Insert Guest Additions CD image** to enable full-screen resolution and shared clipboard.
2.  **System Update**: Open a terminal (`Ctrl+Alt+T`) and run:
    ```bash
    sudo apt update && sudo apt upgrade -y
    ```

-----

## 3\. Step 2: Python Environment Setup

We recommend using a virtual environment to manage the specific dependencies required for sensor simulation and vector search.

### Install System Dependencies

```bash
sudo apt install python3-pip python3-venv build-essential -y
```

### Initialize Workspace

```bash
git clone https://github.com/Pruthil-2910/RAG_for_Edge_devices.git
cd RAG_for_Edge_devices

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
```

### Install Project Libraries

```bash
pip install -r requirements.txt
```

-----

## 4\. Step 3: Running the Scripts

The core logic is contained in `qdrant.py`. This script handles the full pipeline from data generation to local retrieval.

### Execution

1.  **Run the Main Script**:
    ```bash
    python qdrant.py
    ```
2.  **What happens next**:
      - The `SensorSimulator` generates 7,200 minutes of synthetic data for Temperature, Humidity, Vibration, and Air-quality.
      - Raw data is windowed into 1-minute blocks and converted into **10-dimensional feature vectors** (Mean, Std Dev, Skewness, Kurtosis, etc.).
      - An embedded **Qdrant Edge Shard** is created locally in the `./qdrant-edge-1` directory.
      - Historical data is indexed with metadata (location, sensor type, anomaly labels).

-----

## 5\. Project Architecture

The system mimics a production IoT deployment by separating the "intelligence" of search from physical hardware friction.

  - **SensorSimulator**: Injects realistic patterns like "Bearing Wear" (sustained shifts) and "Sudden Spikes".
  - **EdgeShard**: A self-contained, in-process storage unit for vector and payload data—think of it as "SQLite for vector search".
  - **Metadata Filtering**: Combines vector similarity with boolean logic (e.g., "Find patterns similar to this, but only in Zone\_B").

-----

## 6\. Results & Analysis

The script outputs search results to `search_results.txt`.

### Key Findings

  - **High Precision**: The top 5 matches for a simulated "Bearing Wear" signal typically show similarity scores above **0.99**, correctly retrieving historical anomalies.
  - **Constraint Power**: Using filtered search (e.g., `sensor_type: Humidity`) ensures that retrieval is contextually accurate even in noisy environments.

-----

**References**

  - [Qdrant Edge Documentation](https://qdrant.tech/documentation/edge/)
  - [Ubuntu Developer Setup](https://documentation.ubuntu.com/ubuntu-for-developers/howto/python-setup/)
