import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler,LabelEncoder
from sklearn.metrics import  mean_absolute_error, mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="", 
)

print(mydb)
mycursor = mydb.cursor(buffered=True)
mycursor.execute("USE  super_market")

st.set_page_config(page_title= "Super Market Weekly Sales Prediction ", layout="wide")
icons = {
    "Home": "🏠",
    "Weekly Sales Prediction App": "🔄",
    "Data Visualization" :"📈",              
    "ABOUT": "📊",
    
}
SELECT = st.sidebar.selectbox("Choose an option", list(icons.keys()), format_func=lambda option: f'{icons[option]} {option}', key='selectbox')
if SELECT == "Home":
    
        st.markdown("## :green[**Technologies Used :**] Python,Streamlit, Pandas, Numpy, Matplotlib, Seaborn, Plotly Express,Machine Learning,Sklearn,metrics,Decision Tree Regressor,Grid Search CV,train_test_split,StandardScaler,LabelEncoder, Pickle")
        st.markdown("## :green[**Overview :**] This Streamlit web app is designed for predicting weekly sales in a super market. Users can input various parameters such as store, department, temperature, customer price index (CPI), unemployment, size, markdown, fuel price, and more. The app uses a pre-trained machine learning model to make predictions based on these inputs. The predicted weekly sales are then visualized in different ways to provide insights into the data.")
    
   
        st.markdown("## :green[**How to Use :**]")
        st.markdown("1. Navigate to the 'Weekly Sales Prediction App' tab.")
        st.markdown("2. Enter numerical values and select categorical options.")
        st.markdown("3. Click the 'Submit' button to get the predicted weekly sales.")
        st.markdown("4. The predicted sales will be displayed, and visualizations can be seen in the 'Data Visualization' tab.")
    

if SELECT == "Weekly Sales Prediction App":    

    st.header("Weekly Sales Prediction App")
    # Predefined values
    store = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
    dept = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
    Isholiday_dict = {'False': 0, 'True': 1}
    types_dict = {'A': 1, 'B': 2, 'C': 3}
    week_days = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    year = [2010, 2011, 2012]

    with st.form("my_form"):
        col1, col2 = st.columns(2)
        # Text inputs for numerical values
        with col1:

            temp = st.text_input("Enter Temperature (Min:611728 & Max:1722207579)")
            cpi = st.text_input("Enter Customer Price Index (Min:0.18 & Max:400)")
            unemp = st.text_input("Enter Unemployment (Min:1, Max:2990)")
            size = st.text_input("Enter Space Size (Min:12458, Max:30408185)")
            Total_MarkDown = st.text_input("Enter Markdown (Min:12458, Max:30408185)")
            fuel_price = st.text_input("Enter Fuel_Price (Min:12458, Max:30408185)")
            expected_weekly_sale = st.text_input("Enter Expected_Weekly_Sale (Min:12458, Max:30408185)")
    
        with col2:
        # Convert text inputs to appropriate data types
            
            temp = float(temp) if temp else None
            cpi = float(cpi) if cpi else None
            unemp = float(unemp) if unemp else None
            size = float(size) if size else None
            Total_MarkDown = float(Total_MarkDown) if Total_MarkDown else None  
            fuel_price = np.log(float(fuel_price)) if fuel_price else None
            expected_weekly_sale = np.log(float(expected_weekly_sale)) if expected_weekly_sale else None
        
            # User inputs for categorical values
            store = st.selectbox("store", store, key=1)
            dept = st.selectbox("dept", dept, key=2)
            Isholiday = st.selectbox("IsHoliday",list(Isholiday_dict.keys()), key=3)
            store_type = st.selectbox("type", list(types_dict.keys()), key=4)
            week_day = st.selectbox("Week_day", week_days, key=5)
            month = st.selectbox("Month", month, key=6)
            year = st.selectbox("Year", year, key=7)

        # Form submission button
        submit_button = st.form_submit_button("Submit")

    # Load the model and scaler
    with open('C:\\Users\\Admin\\Desktop\\vscode\\decmodel.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    with open('C:\\Users\\Admin\\Desktop\\vscode\\fscaler.pkl', 'rb') as f:
        scaler_loaded = pickle.load(f)

        # Function to make predictions
        def predict_sales(sample):
            sample_scaled = scaler_loaded.transform(sample.reshape(1, -1))
            prediction = loaded_model.predict(sample_scaled)
            return np.exp(prediction)
    #
        # Use the user inputs for prediction
    if submit_button:
        # Create feature array
      
        prediction_result = predict_sales(np.array([[store,dept, Isholiday_dict[Isholiday],temp,cpi,unemp,types_dict[store_type], size,Total_MarkDown,fuel_price,expected_weekly_sale,week_day, month, year]]))
        
        mycursor.execute("USE  super_market")
        query = "INSERT INTO weekly_sales_predictions (store, dept, Isholiday, temp, cpi, unemp, store_type, size, Total_MarkDown, fuel_price, expected_weekly_sale, week_day, month, year, predicted_sales) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        data_to_insert = (store,dept, Isholiday_dict[Isholiday],temp,cpi,unemp,types_dict[store_type], size,Total_MarkDown,fuel_price,expected_weekly_sale,week_day, month, year, prediction_result[0])
        mycursor.execute(query, data_to_insert)
        mydb.commit()

        st.success(f'Predicted Weekly Sales: ${prediction_result[0]:.2f}')

if SELECT == "Data Visualization" :
    st.header("Data Visualization")
    col1,col2,col3,col4,col5 = st.columns([1,1,1,1,1],gap="small")
    
    with col1:
        mycursor.execute("SELECT Isholiday, Total_MarkDown, predicted_sales FROM weekly_sales_predictions  GROUP BY Isholiday ORDER BY predicted_sales DESC LIMIT 10")
        df = pd.DataFrame(mycursor.fetchall(), columns=['Isholiday','Total_MarkDown', 'predicted_sales'])
       
        fig = px.pie(df, values='predicted_sales',
                                names='Isholiday',
                                title='Isholiday wise Sales',
                                color_discrete_sequence=px.colors.sequential.Agsunset,
                                hover_data=['predicted_sales'],
                                labels={'predicted_sales':'predicted_sales'})

        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig,use_container_width=True)

    with col2:
        mycursor.execute("SELECT fuel_price, expected_weekly_sale,predicted_sales FROM weekly_sales_predictions   ORDER BY predicted_sales DESC LIMIT 10")
        df = pd.DataFrame(mycursor.fetchall(), columns=['fuel_price','expected_weekly_sale','predicted_sales'])
        
        fig = px.pie(df, values='predicted_sales',
                                names='fuel_price',
                                title='fuel_price wise Sales',
                                color_discrete_sequence=px.colors.sequential.Agsunset,
                                hover_data=['predicted_sales'],
                                labels={'predicted_sales':'predicted_sales'})

        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig,use_container_width=True)
        
        
        
    with col3:
        mycursor.execute("SELECT store, dept,predicted_sales FROM weekly_sales_predictions   ORDER BY predicted_sales DESC LIMIT 10")
        df = pd.DataFrame(mycursor.fetchall(), columns=['store','dept','predicted_sales'])
        
        fig = px.pie(df, values='predicted_sales',
                                names='dept',
                                title='Department-wise Sales',
                                color_discrete_sequence=px.colors.sequential.Agsunset,
                                hover_data=['predicted_sales'],
                                labels={'predicted_sales':'predicted_sales'})

        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig,use_container_width=True)

    with col4:
        mycursor.execute("SELECT temp,cpi,unemp,predicted_sales FROM weekly_sales_predictions   ORDER BY predicted_sales DESC LIMIT 10")
        df = pd.DataFrame(mycursor.fetchall(), columns=['temp','cpi','unemp','predicted_sales'])
      
        fig = px.pie(df, values='predicted_sales',
                                names='temp',
                                title='temp-wise Sales',
                                color_discrete_sequence=px.colors.sequential.Agsunset,
                                hover_data=['predicted_sales'],
                                labels={'predicted_sales':'predicted_sales'})

        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig,use_container_width=True)

    with col5:
        mycursor.execute("SELECT week_day,month,year,predicted_sales FROM weekly_sales_predictions   ORDER BY predicted_sales DESC LIMIT 10")
        df = pd.DataFrame(mycursor.fetchall(), columns=['week_day','month','year','predicted_sales'])
         
        fig = px.pie(df, values='predicted_sales',
                                names='week_day',
                                title='week_day-wise Sales',
                                color_discrete_sequence=px.colors.sequential.Agsunset,
                                hover_data=['predicted_sales'],
                                labels={'predicted_sales':'predicted_sales'})

        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig,use_container_width=True)

if SELECT == "ABOUT":
        st.header("About Super Market Weekly Sales Prediction App")

        st.markdown("""
        Welcome to the Super Market Weekly Sales Prediction App! This application is designed to provide insights into weekly sales predictions for different super market scenarios. Whether you are a business owner, analyst, or just curious about data, this app aims to make the prediction process easy and interactive.

        **Key Features:**
        - Predict weekly sales based on various input parameters.
        - Visualize predictions through interactive charts and graphs.
        - Explore different aspects of the data, such as holidays, department-wise sales, and more.

        **Technologies Used:**
        - Python
        - Pandas
        - Numpy
        - Matplotlib
        - Seaborn
        - Machine Learning (Decision Tree Regressor)
        - Streamlit

        **How to Use:**
        1. Navigate to the 'Weekly Sales Prediction App' tab.
        2. Input numerical values and select categorical options.
        3. Click the 'Submit' button to get the predicted weekly sales.
        4. Explore the 'Data Visualization' tab for insightful charts.

        **About the Developers:**
        This app was developed by [Manoj Mahalingam ]. We are passionate about data science and aim to provide user-friendly tools for predictive analysis. If you have any questions or feedback, feel free to [contact us](mailto:mkmeetmanoj47@gmail.com).

        **Acknowledgments:**
        We would like to express our gratitude to the open-source community and the developers of the libraries and tools used in building this application.

        Thank you for using the Super Market Weekly Sales Prediction App! We hope it brings valuable insights to your analysis.
        """)

