import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for plots
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Set page title and layout
st.set_page_config(
    page_title="Roasting Machine Quality Predictor",
    layout="wide"
)

# Application title and description
st.title("Roasting Machine Product Quality Predictor")
st.markdown("""
This application predicts the quality of products from a roasting machine based on temperature and humidity sensor readings.
Upload your sensor data or manually input values to generate predictions.
""")

@st.cache_resource
def load_model():
    """Load the pre-trained model"""
    try:
        with open('attached_assets/model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' is in the correct location.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Load a sample of the training data to display"""
    try:
        # Try to read the Excel file instead of CSV
        data = pd.read_excel('attached_assets/new_files.xlsx')
        return data  # Return all rows for better visualization
    except FileNotFoundError:
        st.warning("Sample data file not found. Sample data display is not available.")
        return None
    except Exception as e:
        st.warning(f"Error loading sample data: {e}")
        return None

# Load the model
model = load_model()

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Make Prediction", "View Sample Data", "About"])

if page == "Make Prediction":
    st.header("Input Sensor Values")
    
    st.markdown("""
    ### Temperature and Humidity Sensors
    
    Enter values for all temperature sensors (T_data) and humidity readings (H_data, AH_data) to predict product quality.
    
    **Temperature Sensors:** Values typically range from 200-500 units
    **Humidity Sensors:** Values typically range from 100-200 units for H_data and 7-10 units for AH_data
    """)
    
    # Create columns for more organized layout
    col1, col2, col3 = st.columns(3)
    
    # Dictionary to store all inputs
    input_data = {}
    
    # Generate input fields for temperature sensors
    with col1:
        st.subheader("Temperature Group 1")
        input_data["T_data_1_1"] = st.number_input("T_data_1_1 (Sensor 1-1)", min_value=0, max_value=1000, value=250)
        input_data["T_data_1_2"] = st.number_input("T_data_1_2 (Sensor 1-2)", min_value=0, max_value=1000, value=250)
        input_data["T_data_1_3"] = st.number_input("T_data_1_3 (Sensor 1-3)", min_value=0, max_value=1000, value=250)
        
        st.subheader("Temperature Group 2")
        input_data["T_data_2_1"] = st.number_input("T_data_2_1 (Sensor 2-1)", min_value=0, max_value=1000, value=350)
        input_data["T_data_2_2"] = st.number_input("T_data_2_2 (Sensor 2-2)", min_value=0, max_value=1000, value=350)
        input_data["T_data_2_3"] = st.number_input("T_data_2_3 (Sensor 2-3)", min_value=0, max_value=1000, value=350)
    
    with col2:
        st.subheader("Temperature Group 3")
        input_data["T_data_3_1"] = st.number_input("T_data_3_1 (Sensor 3-1)", min_value=0, max_value=1000, value=475)
        input_data["T_data_3_2"] = st.number_input("T_data_3_2 (Sensor 3-2)", min_value=0, max_value=1000, value=475)
        input_data["T_data_3_3"] = st.number_input("T_data_3_3 (Sensor 3-3)", min_value=0, max_value=1000, value=475)
        
        st.subheader("Temperature Group 4")
        input_data["T_data_4_1"] = st.number_input("T_data_4_1 (Sensor 4-1)", min_value=0, max_value=1000, value=350)
        input_data["T_data_4_2"] = st.number_input("T_data_4_2 (Sensor 4-2)", min_value=0, max_value=1000, value=350)
        input_data["T_data_4_3"] = st.number_input("T_data_4_3 (Sensor 4-3)", min_value=0, max_value=1000, value=350)
    
    with col3:
        st.subheader("Temperature Group 5")
        input_data["T_data_5_1"] = st.number_input("T_data_5_1 (Sensor 5-1)", min_value=0, max_value=1000, value=240)
        input_data["T_data_5_2"] = st.number_input("T_data_5_2 (Sensor 5-2)", min_value=0, max_value=1000, value=240)
        input_data["T_data_5_3"] = st.number_input("T_data_5_3 (Sensor 5-3)", min_value=0, max_value=1000, value=240)
        
        st.subheader("Humidity Sensors")
        input_data["H_data"] = st.number_input("H_data (Humidity)", min_value=0.0, max_value=500.0, value=160.0, step=0.01)
        input_data["AH_data"] = st.number_input("AH_data (Absolute Humidity)", min_value=0.0, max_value=20.0, value=9.0, step=0.01)
    
    # Add a timestamp
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.write(f"Timestamp: {current_datetime}")
    
    # Create a button to trigger prediction
    if st.button("Predict Quality"):
        if model:
            try:
                # Create a DataFrame from the input data
                input_df = pd.DataFrame([input_data])
                
                # Add date_time column
                input_df['date_time'] = current_datetime
                
                # Convert date_time to categorical type as required by the model
                input_df['date_time'] = pd.Series([current_datetime]).astype('category')
                
                # Ensure column order matches what the model expects
                expected_columns = [
                    'date_time', 'T_data_1_1', 'T_data_1_2', 'T_data_1_3',
                    'T_data_2_1', 'T_data_2_2', 'T_data_2_3',
                    'T_data_3_1', 'T_data_3_2', 'T_data_3_3',
                    'T_data_4_1', 'T_data_4_2', 'T_data_4_3',
                    'T_data_5_1', 'T_data_5_2', 'T_data_5_3',
                    'H_data', 'AH_data'
                ]
                input_df = input_df[expected_columns]
                
                # Make prediction
                with st.spinner("Generating prediction..."):
                    prediction = model.predict(input_df)
                
                # Display prediction result
                st.success("Prediction Complete!")
                st.subheader("Predicted Quality")
                
                # Format prediction result based on model output type
                if isinstance(prediction, np.ndarray) and len(prediction) > 0:
                    result = prediction[0]
                    st.metric("Quality Score", f"{result:.4f}")
                    
                    # Provide interpretation of the quality score
                    if hasattr(model, 'classes_'):
                        # For classification models
                        st.write(f"Predicted Class: {result}")
                    else:
                        # For regression models 
                        st.write("""
                        **Interpretation:**
                        - Higher values indicate better quality
                        - Values typically range from 100-1000
                        - Scores of 400 and above indicate good quality
                        """)
                else:
                    st.write("Unexpected prediction format. Please check model output.")
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please ensure all input values are in the expected range and format.")
        else:
            st.error("Model not loaded. Please check if model file exists and is valid.")

elif page == "View Sample Data":
    st.header("Sample Training Data")
    
    st.write("""
    Below is a sample of the data used to train the prediction model. 
    This helps you understand the expected data format and typical value ranges.
    """)
    
    # Load and display sample data
    sample_data = load_sample_data()
    if sample_data is not None:
        # Display the raw data table
        st.subheader("Raw Data Sample (First 10 Rows)")
        st.dataframe(sample_data)
        
        # Display descriptions of features
        st.subheader("Feature Descriptions")
        st.markdown("""
        - **date_time**: Timestamp when readings were recorded
        - **T_data_X_Y**: Temperature readings from sensor group X, sensor Y
        - **H_data**: Humidity reading
        - **AH_data**: Absolute humidity reading
        
        Each temperature sensor group may correspond to a different area of the roasting machine, 
        helping monitor the roasting process at various points.
        """)
        
        # Add data visualizations
        st.header("Data Visualizations")
        
        # Create tabs for different visualizations
        viz_tabs = st.tabs(["Temperature Distribution", "Correlation Heatmap", "Time Series"])
        
        with viz_tabs[0]:
            st.subheader("Temperature Sensor Distribution")
            st.write("Distribution of temperature readings across different sensor groups")
            
            # Select a larger sample for plotting
            plot_data = sample_data.copy()
            
            # Create a figure with multiple subplots for temperature groups
            fig_temp = plt.figure(figsize=(10, 15))
            
            # Plot histograms for each temperature group
            for i in range(1, 6):  # 5 temperature groups
                group_cols = [f'T_data_{i}_{j}' for j in range(1, 4)]  # 3 sensors per group
                for j, col in enumerate(group_cols):
                    if col in plot_data.columns:
                        plt.subplot(5, 3, (i-1)*3 + j + 1)
                        plt.hist(plot_data[col], bins=20, alpha=0.7)
                        plt.title(f'Sensor {col}')
                        plt.xlabel('Temperature Value')
                        plt.ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig_temp)
            
            # Add explanation
            st.write("""
            The histograms above show the distribution of temperature values for each sensor. 
            This helps identify the normal operating ranges and potential outliers for each sensor location.
            """)
        
        with viz_tabs[1]:
            st.subheader("Correlation Between Sensors")
            st.write("Heatmap showing correlations between different sensor readings")
            
            # Calculate correlations - exclude non-numeric columns like date_time
            numeric_data = sample_data.select_dtypes(include=['float64', 'int64'])
            corr_matrix = numeric_data.corr()
            
            # Create heatmap
            fig_corr, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', ax=ax)
            plt.title('Correlation Between Sensors')
            st.pyplot(fig_corr)
            
            # Add explanation
            st.write("""
            The correlation heatmap shows relationships between different sensors:
            - Darker red indicates strong positive correlation (sensors move together)
            - Darker blue indicates strong negative correlation (sensors move in opposite directions)
            - White/light colors indicate weak or no correlation
            
            Strong correlations may indicate sensors in similar zones or affected by the same process variables.
            """)
        
        with viz_tabs[2]:
            st.subheader("Sensor Readings Over Time")
            st.write("Time series plot of sensor readings")
            
            # Convert date_time to datetime if needed
            if 'date_time' in sample_data.columns:
                try:
                    # Get more data for time series visualization
                    time_plot_data = sample_data.copy()
                    time_plot_data['date_time'] = pd.to_datetime(time_plot_data['date_time'])
                    time_plot_data = time_plot_data.set_index('date_time')
                    
                    # Create time series plot
                    fig_time = plt.figure(figsize=(12, 8))
                    
                    # Plot average temperature for each group
                    for i in range(1, 6):
                        group_cols = [f'T_data_{i}_{j}' for j in range(1, 4)]
                        group_cols = [col for col in group_cols if col in time_plot_data.columns]
                        if group_cols:
                            # Calculate average for this group
                            time_plot_data[f'T_Group_{i}_Avg'] = time_plot_data[group_cols].mean(axis=1)
                            plt.plot(time_plot_data.index, time_plot_data[f'T_Group_{i}_Avg'], 
                                    label=f'Temperature Group {i} Avg')
                    
                    # Add humidity if available
                    if 'H_data' in time_plot_data.columns:
                        plt.plot(time_plot_data.index, time_plot_data['H_data'], 
                                label='Humidity', linestyle='--')
                    
                    plt.title('Sensor Readings Over Time')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    st.pyplot(fig_time)
                    
                except Exception as e:
                    st.warning(f"Could not create time series plot: {e}")
                    st.write("Time series visualization requires properly formatted datetime data.")
            else:
                st.info("Time series plot requires date_time column.")
            
            # Add explanation
            st.write("""
            The time series plot shows how sensor readings change over time, helping identify:
            - Trends in temperature and humidity
            - Cyclic patterns in the roasting process
            - Anomalies or unusual events
            """)
    else:
        st.info("Sample data is not available for display.")

elif page == "About":
    st.header("About This Application")
    
    st.markdown("""
    ### Roasting Machine Quality Predictor
    
    This application uses a machine learning model trained on historical sensor data to predict the quality 
    of products from a roasting machine. By inputting current sensor readings, operators can predict the 
    expected quality of the roasting process.
    
    ### How It Works
    
    1. Input the temperature and humidity sensor readings from your roasting machine
    2. The application processes these inputs through a pre-trained machine learning model
    3. The model returns a predicted quality score based on patterns learned from historical data
    
    ### Model Information
    
    The prediction model was trained using supervised learning techniques on historical sensor data and 
    corresponding quality measurements. The model identifies patterns between sensor readings and product quality.
    
    ### Sensor Information
    
    - **Temperature Sensors (T_data_X_Y)**: These sensors monitor temperature at different locations in the roasting machine.
      - Sensor groups 1-5 represent different zones
      - Each group has three sensors for comprehensive monitoring
    
    - **Humidity Sensors**:
      - **H_data**: Relative humidity measurement
      - **AH_data**: Absolute humidity measurement
    """)

# Add footer with version information
st.sidebar.markdown("---")
st.sidebar.markdown("v1.0.0")
st.sidebar.markdown("© 2025 Roasting Machine Quality Predictor")
