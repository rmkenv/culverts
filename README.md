# Culvert Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive geospatial framework for analyzing culvert infrastructure, integrating multi-source data from ArcGIS REST services, USGS NWIS, and NOAA weather alerts. The system performs proximity analysis, hydrologic risk assessment, transportation impact analysis, and machine learning-based failure prediction.

## 🚀 Features

### Data Integration
- **Multi-Source Collection**: Seamlessly integrates culvert data from ArcGIS, stream gauge data from USGS NWIS, and flood event data from NOAA
- **Location-Based Analysis**: Uses Nominatim (OpenStreetMap) to establish analysis boundaries based on county and state
- **Robust Data Parsing**: Handles diverse data formats with automated standardization and error handling

### Spatial Analysis
- **Proximity Analysis**: Identifies stream gauges within configurable buffer distances of culverts using CRS-safe buffering
- **Coordinate Reference System Management**: Automatically selects appropriate UTM zones for accurate distance calculations
- **Geospatial Processing**: Advanced spatial joins and geometric operations

### Risk Assessment
- **Hydrologic Risk Evaluation**: Compares culvert capacity against design floods and assesses sediment risk factors
- **Transportation Impact Analysis**: Estimates economic impact and criticality scores based on traffic volume and road classifications
- **Multi-Criteria Assessment**: Integrates hydraulic, environmental, and transportation factors

### Machine Learning
- **Failure Prediction Model**: Trains Random Forest classifiers to predict culvert failure probability
- **Feature Engineering**: Constructs comprehensive feature vectors from hydraulic and transportation data
- **Model Validation**: Includes performance metrics and cross-validation techniques

### Visualization & Reporting
- **Interactive Mapping**: Creates detailed HTML maps using Folium with toggleable layers and legends
- **Synthetic Flood Scenarios**: Generates and tests culvert performance under various hypothetical flood conditions
- **Summary Reporting**: Produces comprehensive analysis reports with key findings and statistics
- **Data Export**: Saves processed data to GeoJSON format for further analysis

## 📁 Project Structure

```
culvert-analysis/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── culvert_analysis.py      # Main analysis system
├── setup.py                 # Package configuration (optional)
├── .gitignore              # Git ignore patterns
├── LICENSE                 # Project license
└── output/                 # Analysis outputs (git ignored)
    ├── culverts.geojson    # Processed culvert data
    ├── gauges.geojson      # Stream gauge data
    └── flood_events.geojson # Flood event data
```

## 🔧 Setup and Installation

### Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/culvert-analysis.git
   cd culvert-analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import geopandas, requests, folium; print('Installation successful!')"
   ```

## 🎯 Quick Start

### Basic Usage

Run the analysis with default settings:

```bash
python culvert_analysis.py
```

### Custom Analysis

Modify the analysis parameters by editing the main execution block:

```python
if __name__ == "__main__":
    cas = CulvertAnalysisSystem()
    
    # Set analysis location
    cas.set_location_by_county_state("Orange", "California")
    
    # Collect data from various sources
    cas.collect_culvert_data()
    cas.collect_stream_gauge_data()
    cas.collect_flood_event_data()
    
    # Perform spatial and risk analyses
    cas.proximity_analysis(buffer_distance=5000)  # 5km buffer
    cas.hydrologic_risk_assessment()
    cas.transportation_impact_analysis()
    
    # Machine learning and scenario modeling
    cas.train_failure_prediction_model()
    cas.generate_synthetic_flood_scenarios(n_scenarios=10)
    
    # Generate outputs
    interactive_map = cas.create_interactive_map()
    if interactive_map:
        interactive_map.save("culvert_analysis_map.html")
    
    cas.generate_report()
    cas.save_data("output")
    print("Analysis complete!")
```

### Programmatic Usage

```python
from culvert_analysis import CulvertAnalysisSystem

# Initialize the system
analyzer = CulvertAnalysisSystem()

# Set location and collect data
analyzer.set_location_by_county_state("Jefferson", "Colorado")
culverts = analyzer.collect_culvert_data()
gauges = analyzer.collect_stream_gauge_data()

# Perform risk assessment
risk_results = analyzer.hydrologic_risk_assessment()
print(f"Found {len(culverts)} culverts with {risk_results['high_risk_count']} high-risk assets")
```

## 📊 Output Files

The system generates several output files:

| File | Description |
|------|-------------|
| `culvert_analysis_map.html` | Interactive web map with all analysis layers |
| `output/culverts.geojson` | Processed culvert data with risk assessments |
| `output/gauges.geojson` | Stream gauge locations and real-time data |
| `output/flood_events.geojson` | Current flood warnings and advisories |

## 🛠 Configuration Options

### Analysis Parameters

- **Buffer Distance**: Adjust proximity analysis radius (default: 5000m)
- **Risk Thresholds**: Customize capacity ratio thresholds for risk classification
- **ML Model**: Configure Random Forest parameters (n_estimators, random_state)
- **Flood Scenarios**: Set number and characteristics of synthetic flood events

### Data Sources

The system connects to several external APIs:
- **ArcGIS REST Services**: Infrastructure data
- **USGS NWIS**: Real-time stream gauge data
- **NOAA Weather API**: Flood alerts and warnings
- **OpenStreetMap Nominatim**: Geocoding services

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `geopandas` | ≥0.13.0 | Geospatial data manipulation |
| `pandas` | ≥1.5.0 | Data analysis and manipulation |
| `numpy` | ≥1.21.0 | Numerical computing |
| `requests` | ≥2.28.0 | HTTP requests for API calls |
| `folium` | ≥0.14.0 | Interactive map generation |
| `scikit-learn` | ≥1.1.0 | Machine learning algorithms |
| `shapely` | ≥2.0.0 | Geometric operations |

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure compatibility with Python 3.8+

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this software in your research, please cite:

```bibtex
@software{culvert_analysis_system,
  title={Culvert Analysis System: A Comprehensive Geospatial Framework},
  author={Ryan Kmetz},
  year={2025},
  url={https://github.com/rmkenv/culverts},
  version={1.0.0}
}
```


## 🔄 Version History

- **v1.0.0** (2025-08-19): Initial release with core functionality
  - Multi-source data integration
  - Spatial analysis and risk assessment
  - Machine learning failure prediction
  - Interactive visualization

---

**Made with ❤️ for infrastructure resilience and flood risk management**
