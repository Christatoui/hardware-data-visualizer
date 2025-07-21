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
    df = df[df['Time'].dt.year >= 2024]
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
    """Plots a bar chart of the top 10 requesters and allows drill-down."""
    st.subheader("Top 10 Hardware Requesters")
    fig, ax = plt.subplots(figsize=(12, 8))
    top_10_requesters = df['Requester'].value_counts().nlargest(10)
    sns.barplot(x=top_10_requesters.values, y=top_10_requesters.index, palette='plasma', ax=ax)
    ax.set_title('Top 10 Hardware Requesters', fontsize=16)
    ax.set_xlabel('Number of Requests', fontsize=12)
    ax.set_ylabel('Requester', fontsize=12)
    st.pyplot(fig)

    st.subheader("Drill-down: Top 5 Items for a Requester")
    selected_requester = st.selectbox("Select a requester to see their top 5 items:", top_10_requesters.index)

    if selected_requester:
        requester_df = df[df['Requester'] == selected_requester]
        top_5_items = requester_df['Hardware'].value_counts().nlargest(5)

        if not top_5_items.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x=top_5_items.values, y=top_5_items.index, palette='magma', ax=ax2)
            ax2.set_title(f"Top 5 Hardware Requests for {selected_requester}", fontsize=14)
            ax2.set_xlabel("Number of Requests", fontsize=10)
            ax2.set_ylabel("Hardware Type", fontsize=10)
            st.pyplot(fig2)
        else:
            st.info(f"No hardware request data found for {selected_requester}.")

def plot_requests_by_weekday_monthly(df):
    """Plots a line chart of requests by day of the week, with a separate line for each month."""
    df['Weekday'] = df['Time'].dt.day_name()
    df['Month_Name'] = df['Time'].dt.strftime('%B')
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create a pivot table to get the counts
    monthly_weekday_counts = df.groupby(['Month_Name', 'Weekday']).size().reset_index(name='counts')
    monthly_weekday_counts['Weekday'] = pd.Categorical(monthly_weekday_counts['Weekday'], categories=weekday_order, ordered=True)
    monthly_weekday_counts = monthly_weekday_counts.sort_values('Weekday')

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.lineplot(data=monthly_weekday_counts, x='Weekday', y='counts', hue='Month_Name', ax=ax, marker='o')
    
    ax.set_title('Requests by Day of the Week (Monthly Comparison)', fontsize=16)
    ax.set_xlabel('Day of the Week', fontsize=12)
    ax.set_ylabel('Number of Requests', fontsize=12)
    ax.legend(title='Month')
    ax.grid(True)
    st.pyplot(fig)

def plot_requests_by_hour(df):
    """Plots a bar chart showing the percentage of requests between 7 AM and 6 PM."""
    df['Hour'] = df['Time'].dt.hour
    # Filter for hours between 7 AM (7) and 6 PM (18)
    df_filtered_by_hour = df[(df['Hour'] >= 7) & (df['Hour'] <= 18)]
    
    if df_filtered_by_hour.empty:
        st.warning("No request data available between 7 AM and 6 PM.")
        return

    hourly_counts = df_filtered_by_hour['Hour'].value_counts(normalize=True) * 100
    # Ensure all hours from 7 to 18 are present, filling missing ones with 0
    all_hours = pd.Index(range(7, 19), name="Hour")
    hourly_counts = hourly_counts.reindex(all_hours, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='twilight', ax=ax)
    ax.set_title('Requests by Hour (%) Average (7 AM - 6 PM)', fontsize=16)
    ax.set_xlabel('Hour of the Day', fontsize=12)
    ax.set_ylabel('Percentage of Requests (%)', fontsize=12)
    
    # Add percentage labels on top of each bar
    for bar in bars.patches:
        ax.annotate(f'{bar.get_height():.1f}%',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='center',
                    size=10, xytext=(0, 8),
                    textcoords='offset points')

    # Set the limits of the x-axis to trim empty space
    ax.set_xlim(left=-0.5, right=len(hourly_counts)-0.5)
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

def plot_hardware_specific_analysis(df):
    """Provides an analysis view for a selected hardware type."""
    st.header("Analysis by Hardware Type")

    # Dropdown to select a single hardware type
    hardware_to_analyze = st.selectbox("Select a hardware type to analyze:", df['Hardware'].unique())

    if hardware_to_analyze:
        # Filter the dataframe for the selected hardware
        specific_hardware_df = df[df['Hardware'] == hardware_to_analyze]

        if specific_hardware_df.empty:
            st.warning(f"No data available for hardware: {hardware_to_analyze}")
            return

        # --- Top 10 Requesters for this Hardware ---
        st.subheader(f"Top 10 Requesters for '{hardware_to_analyze}'")
        top_10_requesters = specific_hardware_df['Requester'].value_counts().nlargest(10)

        if top_10_requesters.empty:
            st.info(f"No requesters found for '{hardware_to_analyze}'.")
            return

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=top_10_requesters.values, y=top_10_requesters.index, palette='viridis', ax=ax)
        ax.set_title(f"Top 10 Requesters for '{hardware_to_analyze}'", fontsize=16)
        ax.set_xlabel('Number of Requests', fontsize=12)
        ax.set_ylabel('Requester', fontsize=12)
        st.pyplot(fig)

        # --- Drill-down for a specific requester ---
        st.subheader("Drill-down: Top 5 Items for a Requester")
        selected_requester = st.selectbox("Select a requester to see their top 5 items:", top_10_requesters.index)

        if selected_requester:
            # Filter the original dataframe for the selected requester
            requester_df = df[df['Requester'] == selected_requester]
            top_5_items = requester_df['Hardware'].value_counts().nlargest(5)

            if not top_5_items.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.barplot(x=top_5_items.values, y=top_5_items.index, palette='magma', ax=ax2)
                ax2.set_title(f"Top 5 Hardware Requests for {selected_requester}", fontsize=14)
                ax2.set_xlabel("Number of Requests", fontsize=10)
                ax2.set_ylabel("Hardware Type", fontsize=10)
                st.pyplot(fig2)
            else:
                st.info(f"No hardware request data found for {selected_requester}.")

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
        "Analysis by Hardware Type": plot_hardware_specific_analysis,
        "Requests by Weekday (Monthly)": plot_requests_by_weekday_monthly,
        "Requests by Hour (%) Average": plot_requests_by_hour,
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
    graph_function = graph_options[selected_graph]

    # Special handling for the new analysis mode, which uses the full dataset
    if selected_graph == "Analysis by Hardware Type":
        graph_function(df_cleaned)
    else:
        # All other graphs use the data filtered by the sidebar
        if not filtered_df.empty:
            graph_function(filtered_df)
        else:
            st.warning("No data available for the selected filters.")

else:
    st.info("Please upload a CSV file to get started.")
