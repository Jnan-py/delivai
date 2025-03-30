# DelivAI: AI-Powered Road Delivery and Trip Optimization

DelivAI is an AI-powered Streamlit application that optimizes Road Trips and Deliveries by combining geospatial analysis, weather prediction, and delay estimation. The project leverages historical taxi trip data and weather datasets to forecast delivery delays, enabling dynamic route planning and resource allocation.

## Overview

DelivAI integrates multiple data sources and machine learning techniques to predict delivery delays and optimize routes. The key components include:

- **Weather Prediction:**  
  Neural network models are trained (using LSTM networks) to forecast weather parameters such as snow (SNOW), average temperature (TAVG), and precipitation (PRCP) based on historical weather data.
- **Delay Prediction:**  
  A delay prediction model (built using an SVR-based or neural network approach) is trained on an enhanced taxi dataset that includes weather predictions. This model estimates the trip duration (i.e., delay) based on geospatial, temporal, and weather features.

- **Data Visualization:**  
  Interactive maps (using Folium) and Plotly visualizations provide insights into route planning and historical trip performance.

## Workflow

The overall workflow of DelivAI is as follows:

1. **Data Preparation:**
   - A normal NYC taxi dataset (e.g., _nyc_taxi_sample.csv_) is collected.
   - A separate weather dataset is obtained from historical records. (e.g., _NYC_200s.csv_)
2. **Weather Model Training:**
   - Using the notebook `train_weather.ipynb` (located in the `notebooks/` folder), neural network models (LSTM-based) are trained to predict weather parameters (SNOW, TAVG, PRCP) using historical weather data.
   - The resulting models are saved as:
     - `models/nn_model_SNOW.h5`
     - `models/nn_model_TAVG.h5`
     - `models/nn_model_PRCP.h5`
3. **Dataset Enhancement:**

   - The notebook `prepare_weather_ds.ipynb` (in `notebooks/`) uses the weather prediction models to augment the original taxi dataset with predicted weather values.
   - A new enhanced dataset, `Datasets/fin_ds.csv`, is generated containing additional columns for weather (e.g., `SNOW_pred`, `TAVG_pred`, `PRCP_pred`) along with other relevant features.

4. **Delay Model Training:**

   - The notebook `train_delay.ipynb` (in `notebooks/`) trains a delay prediction model on the enhanced dataset (`fin_ds.csv`).
   - The trained delay model is saved as `models/trip_duration_model.h5` along with a preprocessing pipeline (`preprocessors/preprocessing_pipeline.pkl`).

5. **Deployment and Prediction:**
   - The `main.py` file (this repository’s main application) integrates the trained weather and delay models.
   - The app allows users to:
     - Compute optimal routes using geospatial data.
     - Predict trip duration (delay) based on input features such as trip distance, pickup/dropoff coordinates, pickup time, and predicted weather conditions.
     - View interactive maps and exploratory data visualizations.

## Project Structure

```
delivai/
├── Datasets/
│   ├── fin_ds.csv
│   └── nyc_taxi_sample.csv
│   └── other datasets..
├── models/
│   ├── trip_duration_model.h5
│   ├── nn_model_SNOW.h5
│   ├── nn_model_TAVG.h5
│   └── nn_model_PRCP.h5
├── notebooks/
│   ├── prepare_weather_ds.ipynb
│   ├── train_delay.ipynb
│   └── train_weather.ipynb
├── preprocessors/
│   └── preprocessing_pipeline.pkl
├── main.py
├── .env
├── requirements.txt
└── README.md
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Jnan-py/delivai.git
   cd delivai
   ```

2. **Create and activate a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables:**

   ```bash
   GOOGLE_MAPS_API_KEY= your-google-map-api
   ```

## Usage

### Running the Application

To run the main application, use:

```bash
streamlit run main.py
```

The Streamlit app will open in your browser, where you can:

- View route planning maps and traffic heatmaps.
- Input trip details to predict delivery/trip time.
- Explore interactive data visualizations.

### Model Training

Train the models using the provided Jupyter notebooks in the `notebooks/` folder:

- **Weather Prediction Model:**  
  Run `notebooks/train_weather.ipynb` to train and save weather models (`nn_model_SNOW.h5`, `nn_model_TAVG.h5`, `nn_model_PRCP.h5`).

- **Dataset Preparation:**  
  Run `notebooks/prepare_weather_ds.ipynb` to generate the enhanced dataset (`fin_ds.csv`) that combines taxi data with weather predictions.

- **Delay Prediction Model:**  
  Run `notebooks/train_delay.ipynb` to train the delay model on `fin_ds.csv` and save the model (`trip_duration_model.h5`) and preprocessing pipeline (`preprocessing_pipeline.pkl`).

## Data Sources

- **NYC Taxi Data:**  
  Historical taxi trip records (e.g., NYC TLC data, Kaggle datasets).

- **Weather Data:**  
  Historical weather data collected from past records. (e.g., NYC_2000s dataset).

## Contributing

Contributions, issues, and feature requests are welcome!  
Please check the [issues page](https://github.com/Jnan-py/delivai/issues) for known issues before submitting a new one.
