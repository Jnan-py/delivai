import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import pandas as pd
import joblib
import logging
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
import requests
import warnings
from dotenv import load_dotenv
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.pipeline import Pipeline

load_dotenv()

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def load_network(place_name: str):
    logger.info(f"Fetching road network for: {place_name}")
    try:
        G = ox.graph_from_place(place_name, network_type='drive')
        return G
    except Exception as e:
        st.error(f"Error loading network graph: {e}")
        return None

@st.cache_data(show_spinner=False)
def geocode_address(address: str):
    from geopy.exc import GeocoderUnavailable, GeocoderTimedOut
    geolocator = Nominatim(user_agent="delivai_app_production", timeout=10)
    try:
        location = geolocator.geocode(address)
    except (GeocoderUnavailable, GeocoderTimedOut) as e:
        st.error(f"Geocoding error for address '{address}': {e}")
        return None
    if location is None:
        st.error(f"Address '{address}' could not be geocoded.")
        return None
    return (location.latitude, location.longitude)

def compute_route(G, origin: tuple, destination: tuple):
    try:
        origin_node = ox.distance.nearest_nodes(G, origin[1], origin[0])
        destination_node = ox.distance.nearest_nodes(G, destination[1], destination[0])
        route = nx.shortest_path(G, origin_node, destination_node, weight='length')
        return route
    except Exception as e:
        st.error(f"Error computing route: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_delay_model():
    try:
        model = load_model(r"models/trip_duration_model.h5", compile=False)
        logger.info("Delay prediction model loaded successfully.")
        return model
    except Exception as e:
        st.error("Delay prediction model not found. Please run train_delay.ipynb to create trip_duration_model.pkl.")
        logger.error(e)
        return None

@st.cache_resource(show_spinner=False)
def load_preprocessor():
    try:
        pipe = joblib.load('preprocessors/preprocessing_pipeline.pkl')
        return pipe
    except Exception as e:
        st.error("Error loading the preprocessor")
        logger.error(e)
        return None

def compute_geodesic_distance(origin: tuple, destination: tuple):
    return geodesic(origin, destination).miles

@st.cache_resource(show_spinner=False)
def load_nn_weather_model(model_name: str):
    try:
        model = load_model(f"models/nn_model_{model_name}.h5")
        return model
    except Exception as e:
        st.error(f"NN weather model for {model_name} not available. {e}")
        return None

def predict_weather_nn(model, pickup_dt, lat, lon):
    date_ordinal = pickup_dt.toordinal()
    X = np.array([[date_ordinal, lat, lon]])
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    pred = model.predict(X)
    return float(pred[0][0])

def predict_delay(features, delay_model, pipe):    
    df = pd.DataFrame([features])
    
    X_processed = pipe.transform(df)
    
    predicted_duration = delay_model.predict(X_processed)
    
    return predicted_duration[0]

@st.cache_data(show_spinner=False)
def load_eda_data(file_path: str = r"Datasets\fin_ds.csv"):
    try:
        df = pd.read_csv(file_path)
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%Y-%m-%d %H:%M:%S')
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%Y-%m-%d %H:%M:%S')
        df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds()
        return df
    except Exception as e:
        st.error(f"Error loading EDA data: {e}")
        return None

def compute_eda_visualisations_plotly(df: pd.DataFrame):
    import plotly.express as px
    figs = {}
    threshold = df['trip_distance'].quantile(0.99)
    filtered = df[df['trip_distance'] <= threshold]
    fig1 = px.box(filtered, x="trip_distance", title=f"Trip Distance Boxplot (Cap: {threshold:.2f} miles)")
    figs['Trip Distance Boxplot'] = fig1
    
    fig2 = px.histogram(df, x="trip_duration", nbins=50, title="Trip Duration Distribution")
    figs['Trip Duration Histogram'] = fig2
    
    if "passenger_count" in df.columns:
        df_bar = df['passenger_count'].value_counts().reset_index()
        df_bar.columns = ['passenger_count', 'count']
        fig3 = px.bar(df_bar,
                      x='passenger_count', y='count',
                      labels={'passenger_count': 'Passenger Count', 'count': 'Frequency'},
                      title="Passenger Count Distribution")
        figs['Passenger Count Distribution'] = fig3
    
    if "VendorID" in df.columns:
        vendor_counts = df["VendorID"].value_counts().reset_index()
        vendor_counts.columns = ["VendorID", "Count"]
        fig4 = px.pie(vendor_counts, names="VendorID", values="Count", title="Vendor Distribution")
        figs['Vendor Distribution'] = fig4
    
    fig5 = px.scatter(df, x="trip_distance", y="trip_duration", trendline="ols",
                      title="Trip Distance vs. Trip Duration")
    figs['Distance vs Duration'] = fig5
    
    return figs

def main():
    st.set_page_config(page_title="DelivAI: Delivery and Trip Optimisation", layout="wide")
    st.title("DelivAI: Delivery and Trip Optimization")
    st.markdown("**DelivAI** leverages AI and geospatial data to optimize Road Trips, across the New York area.")
    
    st.sidebar.header("Configuration")
    place_name = st.sidebar.text_input("Area for Road Network", "Manhattan, New York, USA")
    
    if "network_graph" not in st.session_state:
        with st.spinner("Loading road network..."):
            network_graph = load_network(place_name)
            if network_graph is not None:
                st.session_state.network_graph = network_graph
                st.sidebar.success("Network loaded!")
            else:
                st.sidebar.error("Failed to load network.")
    else:
        st.sidebar.success("Network already loaded.")
    
    tabs = st.tabs(["Route Planning", "Trip Duration Prediction", "Data Visualisations", "Data Sources and Work Flow"])
    
    with tabs[0]:
        st.header("Route Planning")
        origin_address = st.text_input("Origin Address", "Times Square, New York, NY", key="origin")
        destination_address = st.text_input("Destination Address", "Central Park, New York, NY", key="destination")
        
        if st.button("Display Maps"):            
            with st.spinner("Computing Route Map..."):
                origin_coords = geocode_address(origin_address)
                destination_coords = geocode_address(destination_address)
                if origin_coords and destination_coords and "network_graph" in st.session_state:
                    route = compute_route(st.session_state.network_graph, origin_coords, destination_coords)
                    if route:
                        route_coords = [
                            (st.session_state.network_graph.nodes[node]['y'],
                            st.session_state.network_graph.nodes[node]['x'])
                            for node in route
                        ]
                        route_map = folium.Map(location=route_coords[0], zoom_start=13)
                        folium.PolyLine(route_coords, color="blue", weight=5, opacity=0.8).add_to(route_map)
                        st.session_state.route_map = route_map
                    else:
                        st.error("Could not compute route.")
                else:
                    st.error("Error in geocoding addresses or network not loaded.")
                
        if st.session_state.get("route_map") is not None:
            st.subheader("Maps Display")
            map_choice = st.radio("Select Map to Display", ("Optimal Route Map", "Traffic Heatmap"))
            if map_choice == "Optimal Route Map":
                st.markdown("### Optimal Route Map")
                st_folium(st.session_state.route_map, width=700, height=500, key="route_map_display")
            else:
                st.markdown("### Traffic Heatmap")                           
                df_eda = load_eda_data()
                heat_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)                
                heat_data = [
                    [row['pickup_latitude'], row['pickup_longitude']]
                    for _, row in df_eda.iterrows()
                ]
                HeatMap(heat_data, radius=10, blur=15).add_to(heat_map)
                st_folium(heat_map, width=700, height=500, key="traffic_map_display")
        else:
            st.info("Click 'Display Maps' to compute and show maps.")
    
    with tabs[1]:
        st.header("Trip Duration Prediction")
        st.markdown("Enter trip details to predict the expected trip duration (in minutes).")
        
        with st.spinner("Geocoding addresses for feature extraction..."):
            origin_coords = geocode_address(st.session_state.get("origin", "Times Square, New York, NY"))
            destination_coords = geocode_address(st.session_state.get("destination", "Central Park, New York, NY"))

        if origin_coords and destination_coords:
            trip_distance = compute_geodesic_distance(origin_coords, destination_coords)
            st.info(f"Computed Trip Distance (miles): {trip_distance:.2f}")
                        
            pickup_date = st.date_input("Pickup Date", key="pickup_date")
            pickup_time = st.time_input("Pickup Time", key="pickup_time")
            if pickup_date and pickup_time:
                pickup_datetime = datetime.combine(pickup_date, pickup_time)
                pickup_hour = pickup_datetime.hour
                pickup_day_of_week = pickup_datetime.weekday()
                st.write(f"Selected Pickup Time: {pickup_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
                
                pickup_longitude = origin_coords[1]
                pickup_latitude = origin_coords[0]
                dropoff_longitude = destination_coords[1]
                dropoff_latitude = destination_coords[0]
                
                date_ordinal = pickup_datetime.toordinal()
                
                model_snow = load_nn_weather_model("SNOW")
                model_temp = load_nn_weather_model("TAVG")
                model_prcp = load_nn_weather_model("PRCP")

                if model_snow and model_temp and model_prcp:
                    SNOW_pred = predict_weather_nn(model_snow, pickup_datetime, pickup_latitude, pickup_longitude)
                    TAVG_pred = predict_weather_nn(model_temp, pickup_datetime, pickup_latitude, pickup_longitude)
                    PRCP_pred = predict_weather_nn(model_prcp, pickup_datetime, pickup_latitude, pickup_longitude)
                else:
                    st.error("One or more weather models are not available.")
                    SNOW_pred, TAVG_pred, PRCP_pred = 0, 0, 0
                
                st.info(f"Weather Predicted Successfully for **{pickup_datetime.strftime('%Y-%m-%d')}**")
                                
                features = {
                    "trip_distance": trip_distance,
                    "pickup_longitude": pickup_longitude,
                    "pickup_latitude": pickup_latitude,
                    "dropoff_longitude": dropoff_longitude,
                    "dropoff_latitude": dropoff_latitude,
                    "ordinal_datetime": date_ordinal,
                    "SNOW_pred": SNOW_pred,
                    "TAVG_pred": TAVG_pred,
                    "PRCP_pred": PRCP_pred
                }               

                pipe = load_preprocessor()             

                if st.button("Predict Trip Duration"):
                    with st.spinner("Predicting Time of Travel..."):
                        delay_model = load_delay_model()
                        if delay_model:
                            predicted_duration = predict_delay(features, delay_model, pipe)
                            st.subheader("Predicted Trip Duration")
                            if trip_distance > 2:
                                st.success(f"Estimated Trip Duration: **{predicted_duration[0]:.2f} minutes**")

                            else:
                                st.success(f"Estimated Trip Duration: **{predicted_duration[0] - 10:.2f} minutes**")   
                        else:
                            st.error("Delay prediction model not available.")
        else:
            st.error("Unable to geocode addresses for feature extraction.")
        
    with tabs[2]:
        st.header("Exploratory Data Analysis (EDA)")
        eda_option = st.radio("Select EDA Option", options=["Data Analysis", "Data Visualisation"])
        df_eda = load_eda_data()
        if df_eda is not None:
            if eda_option == "Data Analysis":
                st.subheader("Basic Statistics")
                dif = df_eda['tpep_dropoff_datetime'] - df_eda['tpep_pickup_datetime']
                dif_hours = dif.dt.total_seconds() / 3600
                df_eda['speed_mph'] = df_eda['trip_distance'] / dif_hours.replace(0, pd.NA)
                analysis_results = {
                    "Mean Trip Distance (miles)": df_eda["trip_distance"].mean(),
                    "Median Trip Distance (miles)": df_eda["trip_distance"].median(),
                    "Mean Trip Duration": dif.mean(),
                    "Median Trip Duration": dif.median(),
                    "Mean Speed (mph)": df_eda["speed_mph"].mean(),
                    "Median Speed (mph)": df_eda["speed_mph"].median()
                }
                for k, v in analysis_results.items():
                    st.write(f"**{k}**: {v}")
            else:
                with st.spinner("Getting Visualisations.."):
                    st.subheader("Data Visualisations")
                    if "eda_figs" not in st.session_state:
                        figs = compute_eda_visualisations_plotly(df_eda)
                        st.session_state.eda_figs = figs
                    for title, fig in st.session_state.eda_figs.items():
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Unable to load EDA dataset. Please ensure r'Datasets\\nyc_taxi_sample.csv' is available.")
        
    with tabs[3]:
        st.header("Data Sources and Work Flow")
        st.markdown("""
- **Historical Delivery Data:**  
  - Integrated historical delivery logs (or taxi trip records) capturing pickup/drop-off times and trip distances.  
  - This data is used to train models for estimating delays and optimizing routes.

- **Traffic Data:**  
  - Used real-time traffic data from source [Google Maps Directions API](https://developers.google.com/maps/documentation/directions/start) to capture current congestion levels.  
  - Historical traffic data can further improve model accuracy by identifying typical delays during peak periods.

- **Geospatial Data:**  
  - Obtain road network information via [OpenStreetMap](https://www.openstreetmap.org) using OSMNX, providing a basis for dynamic route planning.

- **Traffic Heatmaps & Analysis:**  
  - Analyze average speeds and identify slow zones by grouping data based on pickup coordinates.  
  - Visualize slow zones via interactive heatmaps to inform route optimization decisions.
        """)

if __name__ == '__main__':
    main()
