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
    """Plots a bar chart of the average number of requests for each weekday (Mon-Fri)."""
    df['Weekday'] = df['Time'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # Filter for Monday to Friday
    df_weekdays = df[df['Weekday'].isin(weekday_order)]
    
    if df_weekdays.empty:
        st.warning("No data available for weekdays (Mon-Fri) in the selected range.")
        return
        
    # Calculate the number of unique weeks in the dataset to get a proper average
    num_weeks = (df_weekdays['Time'].max() - df_weekdays['Time'].min()).days / 7
    if num_weeks < 1:
        num_weeks = 1 # Avoid division by zero if the range is less than a week

    # Get total counts per weekday and calculate the average
    weekday_counts = df_weekdays['Weekday'].value_counts()
    average_counts = (weekday_counts / num_weeks).reindex(weekday_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=average_counts.index, y=average_counts.values, palette='crest', ax=ax)
    ax.set_title('Average Daily Requests by Weekday (Mon-Fri)', fontsize=16)
    ax.set_xlabel('Day of the Week', fontsize=12)
    ax.set_ylabel('Average Number of Requests', fontsize=12)
    
    # Add average number labels on top of each bar
    for bar in bars.patches:
        ax.annotate(f'{bar.get_height():.1f}',
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='center',
                    size=10, xytext=(0, 8),
                    textcoords='offset points')
                    
    st.pyplot(fig)

def plot_total_requests_per_month(df):
    """Plots a bar chart of the total number of requests per month."""
    df['Month'] = df['Time'].dt.to_period('M')
    monthly_counts = df['Month'].value_counts().sort_index()
    
    # Convert Period to string for plotting
    monthly_counts.index = monthly_counts.index.strftime('%B %Y')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette='cubehelix', ax=ax)
    ax.set_title('Total Requests per Month', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Number of Requests', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
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
        "Total Requests per Month": plot_total_requests_per_month,
        "Requests by Hour (%) Average": plot_requests_by_hour,
        "Daily Requests by Weekday (Mon-Fri)": plot_daily_requests_by_weekday,
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

    # --- Conditional Monthly Breakdown ---
    # Only show the breakdown option if the correct graph is selected
    if selected_graph == "Requests by Hour (%) Average":
        show_monthly_breakdown = st.sidebar.checkbox("Show Monthly Breakdown")
        
        if show_monthly_breakdown and not filtered_df.empty:
            st.header("Monthly Breakdown: Requests by Hour (%) Average")
            unique_months = sorted(filtered_df['Time'].dt.to_period('M').unique())

            for month in unique_months:
                monthly_df = filtered_df[filtered_df['Time'].dt.to_period('M') == month]
                if not monthly_df.empty:
                    st.subheader(f"Analysis for {month.strftime('%B %Y')}")
                    
                    monthly_df['Hour'] = monthly_df['Time'].dt.hour
                    df_filtered_by_hour = monthly_df[(monthly_df['Hour'] >= 7) & (monthly_df['Hour'] <= 18)]

                    if df_filtered_by_hour.empty:
                        st.warning(f"No request data available between 7 AM and 6 PM for {month.strftime('%B %Y')}.")
                        continue

                    hourly_counts = df_filtered_by_hour['Hour'].value_counts(normalize=True) * 100
                    all_hours = pd.Index(range(7, 19), name="Hour")
                    hourly_counts = hourly_counts.reindex(all_hours, fill_value=0)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bars = sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='viridis', ax=ax)
                    ax.set_title(f'Requests by Hour (%) for {month.strftime("%B %Y")}', fontsize=16)
                    ax.set_xlabel('Hour of the Day', fontsize=12)
                    ax.set_ylabel('Percentage of Requests (%)', fontsize=12)
                    
                    for bar in bars.patches:
                        ax.annotate(f'{bar.get_height():.1f}%',
                                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    ha='center', va='center',
                                    size=10, xytext=(0, 8),
                                    textcoords='offset points')

                    ax.set_xlim(left=-0.5, right=len(hourly_counts)-0.5)
                    st.pyplot(fig)

    if selected_graph == "Daily Requests by Weekday (Mon-Fri)":
        show_weekday_breakdown = st.sidebar.checkbox("Show Monthly Weekday Average")

        if show_weekday_breakdown and not filtered_df.empty:
            st.header("Monthly Breakdown: Average Daily Requests by Weekday")
            unique_months = sorted(filtered_df['Time'].dt.to_period('M').unique())

            for month in unique_months:
                monthly_df = filtered_df[filtered_df['Time'].dt.to_period('M') == month]
                if not monthly_df.empty:
                    st.subheader(f"Analysis for {month.strftime('%B %Y')}")
                    
                    monthly_df['Weekday'] = monthly_df['Time'].dt.day_name()
                    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
                    df_weekdays = monthly_df[monthly_df['Weekday'].isin(weekday_order)]

                    if df_weekdays.empty:
                        st.warning(f"No weekday data for {month.strftime('%B %Y')}.")
                        continue
                    
                    num_weeks = (df_weekdays['Time'].max() - df_weekdays['Time'].min()).days / 7
                    if num_weeks < 1:
                        num_weeks = 1

                    weekday_counts = df_weekdays['Weekday'].value_counts()
                    average_counts = (weekday_counts / num_weeks).reindex(weekday_order)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = sns.barplot(x=average_counts.index, y=average_counts.values, palette='crest', ax=ax)
                    ax.set_title(f'Average Daily Requests for {month.strftime("%B %Y")}', fontsize=16)
                    ax.set_xlabel('Day of the Week', fontsize=12)
                    ax.set_ylabel('Average Number of Requests', fontsize=12)
                    
                    for bar in bars.patches:
                        ax.annotate(f'{bar.get_height():.1f}',
                                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                    ha='center', va='center',
                                    size=10, xytext=(0, 8),
                                    textcoords='offset points')
                    st.pyplot(fig)

    if selected_graph == "Total Requests per Month":
        show_daily_breakdown = st.sidebar.checkbox("Show Daily Breakdown")

        if show_daily_breakdown and not filtered_df.empty:
            st.header("Daily Breakdown of Total Requests")
            unique_months = sorted(filtered_df['Time'].dt.to_period('M').unique())

            for month in unique_months:
                monthly_df = filtered_df[filtered_df['Time'].dt.to_period('M') == month]
                if not monthly_df.empty:
                    st.subheader(f"Daily Requests for {month.strftime('%B %Y')}")
                    
                    daily_counts = monthly_df['Time'].dt.date.value_counts().sort_index()
                    
                    fig, ax = plt.subplots(figsize=(15, 6))
                    sns.barplot(x=daily_counts.index.strftime('%Y-%m-%d'), y=daily_counts.values, color='skyblue', ax=ax)
                    ax.set_title(f'Daily Requests for {month.strftime("%B %Y")}', fontsize=16)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Number of Requests', fontsize=12)
                    
                    # Create labels for the x-axis, showing only Mondays
                    x_labels = [d.strftime('%Y-%m-%d') if d.weekday() == 0 else '' for d in daily_counts.index]
                    ax.set_xticklabels(x_labels)
                    
                    ax.tick_params(axis='x', rotation=45)
                    ax.grid(axis='y')
                    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to get started.")
