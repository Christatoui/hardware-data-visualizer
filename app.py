import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Data Cleaning and Processing Functions ---
def clean_data(df):
    """Cleans and preprocesses the hardware request data."""
    if df is None:
        return None
    df.dropna(subset=['Hardware', 'Requester'], inplace=True)
    df['Hardware'] = df['Hardware'].astype(str).str.replace(r'\s*test\s*$', '', regex=True).str.strip()
    df['Hardware'] = df['Hardware'].str.replace(r'\s*\.\s*$', '', regex=True).str.strip()
    df.loc[df['Hardware'].str.contains('^PC', case=False, na=False), 'Hardware'] = 'PC'
    df.loc[df['Hardware'].str.contains('^Windows PC', case=False, na=False), 'Hardware'] = 'PC'
    df.loc[df['Hardware'].str.contains('Other', case=False, na=False), 'Hardware'] = 'Other'

    def parse_dates(date_str):
        for fmt in ('%b %d, %Y, %I:%M:%S %p', '%B %dth, %Y at %I:%M %p UTC', '%B %d, %Y, %I:%M:%S %p', ',', 'Today'):
            try:
                return pd.to_datetime(date_str, format=fmt)
            except (ValueError, TypeError):
                continue
        return pd.NaT

    df['Time'] = df['Time'].apply(parse_dates)
    df.dropna(subset=['Time'], inplace=True)
    return df

# --- Plotting Functions ---

def plot_hardware_counts(df):
    """Plots a bar chart of hardware request counts."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.countplot(y='Hardware', data=df, order=df['Hardware'].value_counts().index, palette='viridis', ax=ax)
    ax.set_title('Hardware Request Counts', fontsize=16)
    ax.set_xlabel('Number of Requests', fontsize=12)
    ax.set_ylabel('Hardware Type', fontsize=12)
    st.pyplot(fig)

def plot_top_requesters(df):
    """Plots a bar chart of the top 10 requesters."""
    fig, ax = plt.subplots(figsize=(12, 8))
    top_10_requesters = df['Requester'].value_counts().nlargest(10)
    sns.barplot(x=top_10_requesters.values, y=top_10_requesters.index, palette='plasma', ax=ax)
    ax.set_title('Top 10 Hardware Requesters', fontsize=16)
    ax.set_xlabel('Number of Requests', fontsize=12)
    ax.set_ylabel('Requester', fontsize=12)
    st.pyplot(fig)

def plot_requests_by_weekday_monthly(df):
    """Plots a grouped bar chart of requests by day of the week for each month."""
    df['Weekday'] = df['Time'].dt.day_name()
    df['Month_Name'] = df['Time'].dt.strftime('%B')
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    monthly_weekday_counts = df.groupby(['Month_Name', 'Weekday']).size().unstack(fill_value=0).reindex(columns=weekday_order)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    monthly_weekday_counts.plot(kind='bar', ax=ax)
    ax.set_title('Requests by Day of the Week for Each Month', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Number of Requests', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Day of the Week')
    st.pyplot(fig)

def plot_requests_by_hour(df):
    """Plots a bar chart showing the percentage of requests by hour of the day."""
    df['Hour'] = df['Time'].dt.hour
    hourly_counts = df['Hour'].value_counts(normalize=True).sort_index() * 100
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='twilight', ax=ax)
    ax.set_title('Percentage of Requests by Hour of the Day', fontsize=16)
    ax.set_xlabel('Hour of the Day', fontsize=12)
    ax.set_ylabel('Percentage of Requests (%)', fontsize=12)
    ax.set_xticks(range(24))
    st.pyplot(fig)

def plot_daily_requests_by_weekday(df):
    """Plots a line chart of daily requests, broken down by day of the week."""
    df['Date'] = df['Time'].dt.date
    df['Weekday'] = df['Time'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    daily_counts = df[df['Weekday'].isin(weekday_order)].groupby(['Date', 'Weekday']).size().unstack(fill_value=0)
    daily_counts.index = pd.to_datetime(daily_counts.index)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.lineplot(data=daily_counts, dashes=False, ax=ax)
    ax.set_title('Daily Requests by Day of the Week (Mon-Fri)', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Requests', fontsize=12)
    ax.legend(title='Day of the Week')
    ax.grid(True)
    st.pyplot(fig)

def plot_hourly_requests_by_month(df):
    """Plots a line chart comparing hourly request patterns across different months."""
    df['Hour'] = df['Time'].dt.hour
    df['Month_Name'] = df['Time'].dt.strftime('%B')
    monthly_hourly_counts = df.groupby(['Month_Name', 'Hour']).size().unstack(fill_value=0)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    sns.lineplot(data=monthly_hourly_counts.T, dashes=False, ax=ax)
    ax.set_title('Hourly Requests by Month', fontsize=16)
    ax.set_xlabel('Hour of the Day', fontsize=12)
    ax.set_ylabel('Number of Requests', fontsize=12)
    ax.set_xticks(range(24))
    ax.legend(title='Month')
    ax.grid(True)
    st.pyplot(fig)

# --- Streamlit App ---

st.set_page_config(layout="wide")

# Set up ngrok tunnel
# This will create a public URL for the Streamlit app
# You may need to provide an auth token if you have a free ngrok account
# You can set the NGROK_AUTHTOKEN environment variable
st.title("Hardware Request Data Visualizer")


uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")
    
    df_cleaned = clean_data(data.copy())

    st.sidebar.header("Graph Options")
    
    graph_options = {
        "Hardware Request Counts": plot_hardware_counts,
        "Top 10 Requesters": plot_top_requesters,
        "Requests by Weekday (Monthly)": plot_requests_by_weekday_monthly,
        "Requests by Hour (%)": plot_requests_by_hour,
        "Daily Requests by Weekday (Mon-Fri)": plot_daily_requests_by_weekday,
        "Hourly Requests by Month": plot_hourly_requests_by_month,
    }

    selected_graph = st.sidebar.selectbox("Choose a graph to display", list(graph_options.keys()))

    # --- Filtering Options ---
    st.sidebar.header("Filter Data")
    
    # Filter by Hardware
    all_hardware = df_cleaned['Hardware'].unique()
    selected_hardware = st.sidebar.multiselect("Filter by Hardware", all_hardware, default=all_hardware)
    
    # Filter by Requester
    all_requesters = df_cleaned['Requester'].unique()
    selected_requesters = st.sidebar.multiselect("Filter by Requester", all_requesters, default=all_requesters)

    # Filter by Date Range
    min_date = df_cleaned['Time'].min().date()
    max_date = df_cleaned['Time'].max().date()
    start_date = st.sidebar.date_input('Start date', min_date)
    end_date = st.sidebar.date_input('End date', max_date)

    # Apply filters
    filtered_df = df_cleaned[
        (df_cleaned['Hardware'].isin(selected_hardware)) &
        (df_cleaned['Requester'].isin(selected_requesters)) &
        (df_cleaned['Time'].dt.date >= start_date) &
        (df_cleaned['Time'].dt.date <= end_date)
    ]

    st.header(selected_graph)
    if not filtered_df.empty:
        graph_function = graph_options[selected_graph]
        graph_function(filtered_df)
    else:
        st.warning("No data available for the selected filters.")

else:
    st.info("Please upload a CSV file to get started.")
