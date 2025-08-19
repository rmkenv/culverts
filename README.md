Culvert Analysis System

This project provides a comprehensive system for analyzing culverts, integrating data from various sources like ArcGIS REST services, USGS NWIS, and NOAA weather alerts. It performs proximity analysis, hydrologic risk assessment, transportation impact analysis, and can even train a machine learning model for failure prediction.

Features

Data Collection: Gathers culvert data from ArcGIS, stream gauge data from USGS NWIS, and flood event data from NOAA.

Location-Based Analysis: Uses Nominatim (OpenStreetMap) to set analysis boundaries based on county and state.

Proximity Analysis: Identifies stream gauges within a specified buffer distance of culverts.

Hydrologic Risk Assessment: Evaluates culvert capacity against design floods and assesses sediment risk.

Transportation Impact Analysis: Estimates economic impact and criticality scores based on traffic volume and road type.

Failure Prediction Model: Trains a Random Forest classifier to predict culvert failure probability (requires sufficient data).

Synthetic Flood Scenarios: Generates and tests culvert performance under various hypothetical flood conditions.

Interactive Mapping: Creates an interactive HTML map using Folium to visualize culverts, gauges, and flood events.

Summary Reporting: Generates a concise report summarizing key analysis findings.

Data Export: Saves collected and analyzed data to GeoJSON files.

Project Structure
culvert-analysis/
│
├── README.md
├── requirements.txt
├── culvert_analysis.py
├── setup.py (optional, for packaging)
├── .gitignore
└── output/ (ignored by git)

Setup and Installation
Clone the repository:
git clone https://github.com/your-username/culvert-analysis.git
cd culvert-analysis

Create a virtual environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`

Install dependencies:
pip install -r requirements.txt

Usage

To run the analysis, execute the culvert_analysis.py script. You can modify the if __name__ == "__main__": block at the bottom of the script to specify your desired county and state, or to customize the analysis steps.

python culvert_analysis.py


The script will:

Set the analysis location.

Collect culvert, gauge, and flood event data.

Perform various analyses (proximity, risk, transportation impact).

Optionally train a failure prediction model and generate synthetic flood scenarios.

Create an interactive map (culvert_analysis_map.html) in the project root.

Generate a summary report to the console.

Save processed data to the output/ directory.

Example Configuration in culvert_analysis.py:
if __name__ == "__main__":
    cas = CulvertAnalysisSystem()
    # Example: change county/state as needed
    cas.set_location_by_county_state("Orange", "California")
    cas.collect_culvert_data()
    cas.collect_stream_gauge_data()
    cas.collect_flood_event_data()
    cas.proximity_analysis(buffer_distance=5000) # 5000 meters
    cas.hydrologic_risk_assessment()
    cas.transportation_impact_analysis()
    cas.train_failure_prediction_model()
    cas.generate_synthetic_flood_scenarios(n_scenarios=5)
    m = cas.create_interactive_map()
    if m:
        m.save("culvert_analysis_map.html")
    cas.generate_report()
    cas.save_data("output")
    print("Analysis complete.")

Dependencies

The project relies on the following Python libraries, listed in requirements.txt:

geopandas

pandas

numpy

requests

folium

scikit-learn

shapely

Contributing

Feel free to fork the repository, open issues, or submit pull requests.

License

Specifyyourlicensehere,e.g.,MITLicense
