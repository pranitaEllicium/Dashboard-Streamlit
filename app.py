import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import yaml
from yaml.loader import SafeLoader
from hashlib import sha256

# def generate_hashes():
#     print("Admin hash:", hash_password("admin123"))
#     print("Analyst hash:", hash_password("analyst456")) 
#     print("Viewer hash:", hash_password("viewer789"))

# generate_hashes()

# Page configuration
st.set_page_config(
    page_title="Data Visualization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Password hashing
def hash_password(password):
    return sha256(password.encode()).hexdigest()

# Load config
def load_config():
    with open('config.yaml') as file:
        return yaml.load(file, Loader=SafeLoader)

# Authentication system
def authenticate():
    config = load_config()
    users = config['credentials']['usernames']
    
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.form("login_form"):
            st.write("## Login to Access Dashboard")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                if username in users:
                    hashed_input = hash_password(password)
                    if users[username]['password'] == hashed_input:
                        st.session_state.user = users[username]
                        st.session_state.username = username
                        st.session_state.authenticated = True
                        st.rerun()
                    else:
                        st.error("Incorrect password")
                else:
                    st.error("Username not found")
        st.stop()

# Check authentication
authenticate()

# Main App (only accessible after authentication)
user = st.session_state.user
username = st.session_state.username

# Logout button
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.username = None
    st.rerun()

st.sidebar.write(f"Welcome, {user['name']} ({username})")

# Load Data
@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def show_eda(df):
    st.header("Exploratory Data Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(["Summary Statistics", "Missing Values", "Datatypes", "Correlation"])

    with tab1:
        st.subheader("Summary Statistics")
        st.write(df.describe())

    with tab2:
        st.subheader("Missing Values Analysis")
        missing = df.isna().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.warning(f"Missing values found in {len(missing)} columns")
            st.write(missing)
            fig, ax = plt.subplots()
            sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis', ax=ax)
            st.pyplot(fig)
        else:
            st.success("No missing values found")

    with tab3:
        st.subheader("Data Types")
        st.write(df.dtypes)

    with tab4:
        st.subheader("Correlation Matrix")
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns found for correlation")

def show_visualizations(df):
    st.header("Interactive Visualizations")
    all_columns = df.columns.tolist()
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    viz_type = st.selectbox(
        "Select visualization type",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]
    )

    if viz_type in ["Bar Chart", "Line Chart", "Pie Chart"]:
        x_axis = st.selectbox("X Axis", categorical_columns + numeric_columns)
        y_axis = st.selectbox("Y Axis", numeric_columns if viz_type != "Pie Chart" else [None] + numeric_columns)
        color = st.selectbox("Color (optional)", [None] + categorical_columns)
        
        if viz_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis, color=color, barmode='group')
        elif viz_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis, color=color)
        elif viz_type == "Pie Chart":
            fig = px.pie(df, names=x_axis, values=y_axis, color=color)
            
    elif viz_type == "Scatter Plot":
        x_axis = st.selectbox("X Axis", numeric_columns)
        y_axis = st.selectbox("Y Axis", numeric_columns)
        color = st.selectbox("Color (optional)", [None] + categorical_columns)
        size = st.selectbox("Size (optional)", [None] + numeric_columns)
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color, size=size, hover_data=all_columns)
        
    elif viz_type in ["Histogram", "Box Plot"]:
        column = st.selectbox("Column", numeric_columns)
        color = st.selectbox("Color (optional)", [None] + categorical_columns)
        
        if viz_type == "Histogram":
            bins = st.slider("Number of bins", 5, 100, 20)
            fig = px.histogram(df, x=column, color=color, nbins=bins)
        else:
            fig = px.box(df, y=column, color=color)
    
    if 'fig' in locals():
        st.plotly_chart(fig, use_container_width=True)

def show_data_manipulation(df):
    st.header("Data Filtering and Sorting")
    
    st.subheader("Filter Data")
    filter_cols = st.columns(3)
    filters = {}
    
    for i, col in enumerate(df.columns):
        with filter_cols[i % 3]:
            if df[col].dtype in ['int64', 'float64']:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                val_range = st.slider(
                    f"Range for {col}",
                    min_val, max_val, (min_val, max_val)
                )
                filters[col] = val_range
            else:
                options = st.multiselect(
                    f"Values for {col}",
                    df[col].unique().tolist(),
                    default=df[col].unique().tolist()
                )
                filters[col] = options
    
    filtered_df = df.copy()
    for col, val in filters.items():
        if df[col].dtype in ['int64', 'float64']:
            filtered_df = filtered_df[(filtered_df[col] >= val[0]) & (filtered_df[col] <= val[1])]
        else:
            filtered_df = filtered_df[filtered_df[col].isin(val)]
    
    st.subheader("Sort Data")
    sort_col = st.selectbox("Sort by", df.columns)
    sort_asc = st.checkbox("Ascending", value=True)
    filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)
    
    st.dataframe(filtered_df)
    
    st.download_button(
        label="Download filtered data as CSV",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name='filtered_data.csv',
        mime='text/csv'
    )
    
    return filtered_df

def main():
    st.title("📊 Interactive Data Visualization Dashboard")
    
    if username == "viewer":
        st.sidebar.warning("Viewer access - limited functionality")
    
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None and not df.empty:
            st.success("File uploaded successfully!")
            
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            page = st.sidebar.radio(
                "Navigate",
                ["Data Overview", "Exploratory Analysis", "Visualizations", "Filter & Sort"]
            )
            
            if page == "Data Overview":
                st.subheader("Basic Information")
                cols1, cols2 = st.columns(2)
                
                with cols1:
                    st.write("**Shape of the dataset:**", df.shape)
                
                with cols2:
                    st.write("**Columns:**", list(df.columns))
            
            elif page == "Exploratory Analysis":
                show_eda(df)
            
            elif page == "Visualizations":
                show_visualizations(df)
            
            elif page == "Filter & Sort":
                if username == "viewer":
                    st.warning("You don't have permission to filter and download data")
                else:
                    show_data_manipulation(df)
        else:
            st.warning("The uploaded file is empty or couldn't be processed")
    else:
        st.info("Please upload a CSV file to get started.")

if __name__ == "__main__":
    main()