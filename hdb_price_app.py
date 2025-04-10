import streamlit as st
import numpy as np
# import joblib
import pandas as pd
# from xgboost import XGBRegressor
from geopy.distance import geodesic
import pydeck as pdk  
from PIL import Image
from models import load_models
from utils import get_lat_long, cal_lease_remaining_years, get_address
from config import town_coordinates, towns, flat_types, flat_models, expected_columns

st.set_page_config(layout="wide")

st.markdown("""
    <style>
        /* Gradient background for the entire app */
        .stApp {
            background: linear-gradient(to right, #ade8f4, #48cae4, #00b4d8, #0096c7);
        }
        /* Style for labels (headings, paragraphs) */
        .stMarkdownContainer h1, h2, h3, h4, h5, h6, p {
            color: #1b3344;  
        }
        /* Style for buttons */
        .stButton > button {
            background-color: #e3f2fd; 
            color: #1b3344; 
            border: none;
            border-radius: 5px;
            font-size: 16px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #90e0ef;
        }
        /* Style for amenity labels */
        .amenity-label {
            font-weight: bold;
            font-size: 16px;
            color: #1b3344; 
        }
    </style>
""", unsafe_allow_html=True)

rf, gb, xgb, dt, meta_model, scaler = load_models()  

# Check if models are loaded correctly
if rf is not None and gb is not None and xgb is not None and dt is not None and meta_model is not None:
    # Streamlit app title
    st.markdown("<h1 style='text-align: center; '>🏠 HDB Resale Price Prediction</h1>", unsafe_allow_html=True)

    st.markdown("<h2>Input Features</h2>", unsafe_allow_html=True)

    st.markdown("<h5>Enter the features of the HDB flat to estimate it's resale price</h5>", unsafe_allow_html=True)

    # Create two columns
    col1, col2, col3 = st.columns([1,1,2])

    # Using dropdowns for categorical inputs in the first column
    with col1:
        selected_town = st.selectbox("**Town**", towns, index=towns.index("JURONG WEST"))
        selected_flat_type = st.selectbox("**Flat Type**", flat_types, index=flat_types.index("5 ROOM"))
        selected_flat_model = st.selectbox("**Flat Model**", flat_models, index=flat_models.index("Improved"))

        block = st.text_input("**HDB Block**", value="909")
        street_name = st.text_input("**Street Name**", value="JURONG WEST ST 91")


    with col2:
        year_of_sale = st.number_input("**Year of Sale**", min_value=2000, max_value=2100, step=1, value=2019)
        storey_median = st.number_input("**Storey Median**", min_value=1, step=1, value=8)
        lease_commence_date = st.number_input("**Lease Commence Date**", min_value=1950, max_value=2023, step=1, value=1989)
        floor_area_sqm = st.number_input("**Floor Area (sqm)**", min_value=0.0, step=1.0, value=122.00)


    # Construct address
    address = get_address(block, street_name)

    latitude, longitude = get_lat_long(address)

    if latitude is None or longitude is None:
        st.error("Could not retrieve coordinates for the provided address.")
    else:
        origin = (latitude, longitude)

    # Load the location data for amenities
    mrt_location = pd.read_csv('data/mrt_address.csv')  
    school_location = pd.read_csv('data/schools_address.csv')  
    supermarket_location = pd.read_csv('data/shops_address.csv')  
    hawker_location = pd.read_csv('data/hawkers_address.csv')  

    # Calculate distances to amenities
    nearest_mrt_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in mrt_location.iterrows()])
    nearest_supermarket_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in supermarket_location.iterrows()])
    nearest_school_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in school_location.iterrows()])
    nearest_hawkers_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in hawker_location.iterrows()])

    # Calculate distance from CBD
    cbd_distance = geodesic(origin, (1.287953, 103.851784)).meters  # CBD coordinates

    # Calculate lease remaining years using lease_commence_date
    calculated_remaining_lease = cal_lease_remaining_years(lease_commence_date, year_of_sale)


    # Create a DataFrame for the input data
    input_data = {
        'nearest_hawkers_distance': [nearest_hawkers_distance],
        'floor_area_sqm': [floor_area_sqm],
        'nearest_mrt_distance': [nearest_mrt_distance],
        'nearest_school_distance': [nearest_school_distance],
        'cbd_distance': [cbd_distance],
        'year_of_sale': [year_of_sale],
        'storey_median': [storey_median],
        'calculated_remaining_lease': [calculated_remaining_lease],
        'nearest_supermarket_distance': [nearest_supermarket_distance]
    }
    
    input_df = pd.DataFrame(input_data)

    for town in towns:
        input_df[f'town_{town}'] = 1 if selected_town == town else 0

    for flat_type in flat_types:
        input_df[f'flat_type_{flat_type}'] = 1 if selected_flat_type == flat_type else 0

    for flat_model in flat_models:
        input_df[f'flat_model_{flat_model}'] = 1 if selected_flat_model == flat_model else 0

    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    # st.write(input_df)

    input_scaled = scaler.transform(input_df)  # Scale validation data
    # st.write(input_scaled)

with col1:
    if st.button("Predict"):
        try:
            # Make predictions using the base models (excluding decision tree)
            rf_pred_new = rf.predict(input_scaled)
            gb_pred_new = gb.predict(input_scaled)
            xgb_pred_new = xgb.predict(input_scaled)
            dt_pred_new = dt.predict(input_scaled)

            # Combine the predictions into a single feature matrix for the meta-model
            X_new_meta = np.column_stack((rf_pred_new, gb_pred_new, dt_pred_new, xgb_pred_new))

            # Make the final prediction using the meta-model
            y_new_pred = meta_model.predict(X_new_meta)
 
            st.markdown("<h4>Predicted Resale Price:  ${:.2f}</h4>".format(y_new_pred[0]), unsafe_allow_html=True)

            # Display the minimum distances to each amenity
            st.markdown("<h4> Distance to Nearby Amenities: </h4>",unsafe_allow_html=True)
            st.markdown(f"<p class='amenity-label'> Nearest MRT Station: {nearest_mrt_distance:.2f} meters</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='amenity-label'> Nearest Supermarket: {nearest_supermarket_distance:.2f} meters</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='amenity-label'> Nearest School: {nearest_school_distance:.2f} meters</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='amenity-label'> Nearest Hawker Center: {nearest_hawkers_distance:.2f} meters</p>", unsafe_allow_html=True)
            st.markdown(f"<p class='amenity-label'> Distance from CBD: {cbd_distance:.2f} meters</p>", unsafe_allow_html=True)

            # Prepare data for Pydeck chart
            town_df_base = input_df.copy().drop(columns=[                
                'nearest_supermarket_distance',
                'nearest_mrt_distance',
                'nearest_school_distance',
                'nearest_hawkers_distance',
                'cbd_distance'
            ])

            # Create a DataFrame for all towns
            town_predictions = []
            for town in towns:
                # Copy the base DataFrame
                town_df = town_df_base.copy()
                
                for current_town in towns:
                    town_df[f'town_{current_town}'] = 1 if current_town == town else 0

                # Get latitude and longitude for the current town from town_coordinates
                if town in town_coordinates:
                    latitude, longitude = town_coordinates[town]
                else:
                    st.error(f"Could not find coordinates for town: {town}.")
                    continue            

                if latitude is None or longitude is None:
                    st.error(f"Could not retrieve coordinates for town: {town}.")
                    continue
                
                # Calculate distances to amenities
                origin = (latitude, longitude)
                nearest_mrt_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in mrt_location.iterrows()])
                nearest_supermarket_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in supermarket_location.iterrows()])
                nearest_school_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in school_location.iterrows()])
                nearest_hawkers_distance = min([geodesic(origin, (row['latitude'], row['longitude'])).meters for _, row in hawker_location.iterrows()])

                # Calculate distance from CBD
                cbd_distance = geodesic(origin, (1.287953, 103.851784)).meters  # CBD coordinates

                # Add the distances to the town_df
                town_df['nearest_mrt_distance'] = nearest_mrt_distance
                town_df['nearest_supermarket_distance'] = nearest_supermarket_distance
                town_df['nearest_school_distance'] = nearest_school_distance
                town_df['nearest_hawkers_distance'] = nearest_hawkers_distance
                town_df['cbd_distance'] = cbd_distance

                town_df = town_df.reindex(columns=expected_columns, fill_value=0)
                # st.write(town_df)

                # Scale the input town_df
                town_scaled = scaler.transform(town_df)

                # Make predictions for the town
                town_rf_pred = rf.predict(town_scaled)
                town_gb_pred = gb.predict(town_scaled)
                town_xgb_pred = xgb.predict(town_scaled)
                town_dt_pred = dt.predict(town_scaled)

                # Combine predictions for meta-model
                town_X_meta = np.column_stack((town_rf_pred, town_gb_pred, town_dt_pred, town_xgb_pred))
                town_y_pred = meta_model.predict(town_X_meta)

                # Store predictions along with town coordinates
                town_predictions.append({
                    'town': town,
                    'latitude': latitude,
                    'longitude': longitude,
                    'predicted_price': town_y_pred[0],
                    'formatted_price': "${:.2f}".format(town_y_pred[0])
                })

            # Convert to DataFrame
            predictions_df = pd.DataFrame(town_predictions)
            predictions_df['normalized_price'] = (predictions_df['predicted_price'] - predictions_df['predicted_price'].min()) / (predictions_df['predicted_price'].max() - predictions_df['predicted_price'].min())

            # Function to map normalized prices to a color gradient from bright yellow to dark green
            def price_to_color(normalized_price):
                # Bright yellow color (255, 255, 0)
                start_color = [255, 255, 0]
                
                # Dark green color (0, 128, 0)
                end_color = [0, 64, 0]

                # Interpolate between bright yellow and dark green
                r = int(start_color[0] + (end_color[0] - start_color[0]) * normalized_price)
                g = int(start_color[1] + (end_color[1] - start_color[1]) * normalized_price)
                b = int(start_color[2] + (end_color[2] - start_color[2]) * normalized_price)

                return [r, g, b]

        
            predictions_df['color'] = predictions_df['normalized_price'].apply(price_to_color)

            # Create a Pydeck chart for all towns
            deck = pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=1.3521,  # Singapore Latitude
                    longitude=103.8198,  # Singapore Longitude
                    zoom=10.5,
                    pitch=0,
                    # height=100,
                    # width=100,
                    # map_style="mapbox://styles/mapbox/light-v11"  # Light mode map style

                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=predictions_df,
                        get_position="[longitude, latitude]",  # Ensure this matches your DataFrame structure
                        get_radius=700,
                        get_color="color",  
                        pickable=True,
                        opacity=0.6,

                    )
                ],
                tooltip={
                    "html": "<b>Town:</b>{town}<br/><b>Predicted Price:</b> {formatted_price}",
                    "style": {"color": "white"},  
                },
            )

            with col3:
                st.markdown("<h3 style='text-align: center;'>🔍 Predicted Resale Prices by Town</h3>", unsafe_allow_html=True)
                st.pydeck_chart(deck)

                # Generate a gradient image
                def create_gradient_image(width=400, height=10):
                    gradient = np.zeros((height, width, 3), dtype=np.uint8)
                    for x in range(width):
                        normalized_price = x / width  # Normalized price for color mapping
                        color = price_to_color(normalized_price)
                        gradient[:, x] = color
                    return Image.fromarray(gradient)

                # Create and display the gradient image
                gradient_image = create_gradient_image()
                st.image(gradient_image, use_column_width=True)

                # Add labels for the gradient bar with improved alignment
                st.markdown("""
                            <div style='display: flex; justify-content: space-between;'>
                                <strong style='color: #1b3344'>Lowest Price</strong>
                                <strong style='color: #1b3344'>Highest Price</strong>
                            </div>
                        """, unsafe_allow_html=True)

        except Exception as prediction_error:
            st.error(f"Error making predictions: {prediction_error}")
