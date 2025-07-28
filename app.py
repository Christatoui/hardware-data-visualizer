import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Matplotlib settings for SVG plots ---
plt.rcParams['figure.facecolor'] = 'white'
# --- Constants ---
ASSIGNMENT_FILE = 'AppRoo/FrontEnd/language_assignments.csv'

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
    
    df['Requester'] = df['Requester'].str.replace(r'\s*\([Vv]\)\s*$', '', regex=True).str.strip()

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

# --- Language Assignment Functions ---
def load_language_assignments():
    """Loads language assignments from the CSV file."""
    if os.path.exists(ASSIGNMENT_FILE):
        return pd.read_csv(ASSIGNMENT_FILE)
    else:
        return pd.DataFrame(columns=['Requester', 'Language Code'])

def save_language_assignments(df):
    """Saves language assignments to the CSV file."""
    os.makedirs(os.path.dirname(ASSIGNMENT_FILE), exist_ok=True)
    df.to_csv(ASSIGNMENT_FILE, index=False)

# --- Plotting Functions ---

def plot_hardware_counts(df):
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = sns.countplot(y='Hardware', data=df, order=df['Hardware'].value_counts().index, palette='viridis', ax=ax)
    ax.set_title('Hardware Request Counts', fontsize=16)
    ax.set_xlabel('Number of Requests', fontsize=12)
    ax.set_ylabel('Hardware Type', fontsize=12)
    for bar in bars.patches:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', size=10)
    st.pyplot(fig, use_container_width=True)

def plot_top_requesters(df):
    st.subheader("Top 10 Hardware Requesters")
    fig, ax = plt.subplots(figsize=(12, 8))
    top_10_requesters = df['Requester'].value_counts().nlargest(10)
    bars1 = sns.barplot(x=top_10_requesters.values, y=top_10_requesters.index, palette='plasma', ax=ax)
    ax.set_title('Top 10 Hardware Requesters', fontsize=16)
    ax.set_xlabel('Number of Requests', fontsize=12)
    ax.set_ylabel('Requester', fontsize=12)
    for bar in bars1.patches:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', size=10)
    st.pyplot(fig, use_container_width=True)

    st.subheader("Drill-down: Top 5 Items for a Requester")
    selected_requester = st.selectbox("Select a requester to see their top 5 items:", top_10_requesters.index)
    if selected_requester:
        requester_df = df[df['Requester'] == selected_requester]
        top_5_items = requester_df['Hardware'].value_counts().nlargest(5)
        if not top_5_items.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars2 = sns.barplot(x=top_5_items.values, y=top_5_items.index, palette='magma', ax=ax2)
            ax2.set_title(f"Top 5 Hardware Requests for {selected_requester}", fontsize=14)
            ax2.set_xlabel("Number of Requests", fontsize=10)
            ax2.set_ylabel("Hardware Type", fontsize=10)
            for bar in bars2.patches:
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', size=10)
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info(f"No hardware request data found for {selected_requester}.")

def plot_requests_by_hour(df):
    df['Hour'] = df['Time'].dt.hour
    df_filtered_by_hour = df[(df['Hour'] >= 7) & (df['Hour'] <= 18)]
    if df_filtered_by_hour.empty:
        st.warning("No request data available between 7 AM and 6 PM.")
        return
    hourly_counts = df_filtered_by_hour['Hour'].value_counts(normalize=True) * 100
    all_hours = pd.Index(range(7, 19), name="Hour")
    hourly_counts = hourly_counts.reindex(all_hours, fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x=hourly_counts.index, y=hourly_counts.values, palette='twilight', ax=ax)
    ax.set_title('Requests by Hour (%) Average (7 AM - 6 PM)', fontsize=16)
    ax.set_xlabel('Hour of the Day', fontsize=12)
    ax.set_ylabel('Percentage of Requests (%)', fontsize=12)
    for bar in bars.patches:
        ax.annotate(f'{bar.get_height():.1f}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=10, xytext=(0, 8), textcoords='offset points')
    ax.set_xlim(left=-0.5, right=len(hourly_counts)-0.5)
    st.pyplot(fig, use_container_width=True)

def plot_daily_requests_by_weekday(df):
    df['Weekday'] = df['Time'].dt.day_name()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    df_weekdays = df[df['Weekday'].isin(weekday_order)]
    if df_weekdays.empty:
        st.warning("No data available for weekdays (Mon-Fri) in the selected range.")
        return
    num_weeks = (df_weekdays['Time'].max() - df_weekdays['Time'].min()).days / 7
    if num_weeks < 1: num_weeks = 1
    weekday_counts = df_weekdays['Weekday'].value_counts()
    average_counts = (weekday_counts / num_weeks).reindex(weekday_order)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = sns.barplot(x=average_counts.index, y=average_counts.values, palette='crest', ax=ax)
    ax.set_title('Average Daily Requests by Weekday (Mon-Fri)', fontsize=16)
    ax.set_xlabel('Day of the Week', fontsize=12)
    ax.set_ylabel('Average Number of Requests', fontsize=12)
    for bar in bars.patches:
        ax.annotate(f'{bar.get_height():.1f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=10, xytext=(0, 8), textcoords='offset points')
    st.pyplot(fig, use_container_width=True)

def plot_total_requests_per_month(df):
    df['Month'] = df['Time'].dt.to_period('M')
    monthly_counts = df['Month'].value_counts().sort_index()
    monthly_counts.index = monthly_counts.index.strftime('%B %Y')
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette='cubehelix', ax=ax)
    ax.set_title('Total Requests per Month', fontsize=16)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Total Number of Requests', fontsize=12)
    for bar in bars.patches:
        ax.annotate(f'{int(bar.get_height())}', (bar.get_x() + bar.get_width() / 2, bar.get_height()), ha='center', va='center', size=10, xytext=(0, 8), textcoords='offset points')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig, use_container_width=True)

def plot_hardware_analysis(df):
    plot_hardware_counts(df)
    st.header("Drill-down Analysis")
    hardware_to_analyze = st.selectbox("Select a hardware type to analyze its top requesters:", df['Hardware'].unique())
    if hardware_to_analyze:
        specific_hardware_df = df[df['Hardware'] == hardware_to_analyze]
        if specific_hardware_df.empty:
            st.warning(f"No data available for hardware: {hardware_to_analyze}")
            return
        st.subheader(f"Top 10 Requesters for '{hardware_to_analyze}'")
        top_10_requesters = specific_hardware_df['Requester'].value_counts().nlargest(10)
        if top_10_requesters.empty:
            st.info(f"No requesters found for '{hardware_to_analyze}'.")
            return
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = sns.barplot(x=top_10_requesters.values, y=top_10_requesters.index, palette='viridis', ax=ax)
        ax.set_title(f"Top 10 Requesters for '{hardware_to_analyze}'", fontsize=16)
        ax.set_xlabel('Number of Requests', fontsize=12)
        ax.set_ylabel('Requester', fontsize=12)
        for bar in bars.patches:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', size=10)
        st.pyplot(fig, use_container_width=True)

def plot_engineer_specific_analysis(df):
    all_engineers = sorted(df['Requester'].unique())
    search_term = st.text_input("Search for an engineer:")
    if search_term:
        filtered_engineers = [eng for eng in all_engineers if search_term.lower() in eng.lower()]
    else:
        filtered_engineers = all_engineers
    if not filtered_engineers:
        st.warning("No engineers found matching your search.")
        return
    engineer_to_analyze = st.selectbox("Select an engineer to analyze:", filtered_engineers)
    if engineer_to_analyze:
        engineer_df = df[df['Requester'] == engineer_to_analyze]
        if engineer_df.empty:
            st.warning(f"No data available for engineer: {engineer_to_analyze}")
            return
        hardware_counts = engineer_df['Hardware'].value_counts()
        total_requests = hardware_counts.sum()
        hardware_percentages = (hardware_counts / total_requests) * 100
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = sns.barplot(x=hardware_counts.values, y=hardware_counts.index, palette='coolwarm', ax=ax)
        ax.set_title(f"Hardware Requests for {engineer_to_analyze}", fontsize=16)
        ax.set_xlabel('Number of Requests', fontsize=12)
        ax.set_ylabel('Hardware Type', fontsize=12)
        for i, bar in enumerate(bars.patches):
            ax.annotate(f'{hardware_percentages.iloc[i]:.1f}%', (bar.get_width(), bar.get_y() + bar.get_height() / 2), ha='center', va='center', size=10, xytext=(20, 0), textcoords='offset points')
            ax.annotate(f'{int(bar.get_width())}', (bar.get_width(), bar.get_y() + bar.get_height() / 2), ha='center', va='center', size=10, xytext=(-20, 0), textcoords='offset points', color='white')
        ax.set_xlim(right=ax.get_xlim()[1] + 1)
        st.pyplot(fig, use_container_width=True)

def plot_top_languages(df, language_df):
    st.header("Top 10 Languages by Total Requests")
    if language_df.empty or 'Language Code' not in language_df.columns or language_df['Language Code'].dropna().empty:
        st.warning("No language codes have been assigned in the 'Data Sheet' tab. Please assign them to see this graph.")
        return
    
    valid_assigned_languages = language_df[language_df['Language Code'].isin(LANGUAGE_CODES.keys())].copy()
    if valid_assigned_languages.empty:
        st.warning("No valid language codes found in assignments. Please check the 'Data Sheet' tab.")
        return

    df_with_languages = pd.merge(df, valid_assigned_languages, on='Requester', how='inner')
    if df_with_languages.empty:
        st.warning("No requests found for the requesters with assigned languages.")
        return

    language_counts = df_with_languages['Language Code'].value_counts().nlargest(10)
    language_names_map = language_counts.index.map(LANGUAGE_CODES).dropna()
    
    if language_names_map.empty:
        st.warning("Could not map any assigned language codes to known languages.")
        return

    valid_codes = [code for code, name in LANGUAGE_CODES.items() if name in language_names_map.values]
    language_counts = language_counts[language_counts.index.isin(valid_codes)]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = sns.barplot(x=language_counts.values, y=language_names_map, palette='rocket', ax=ax)
    ax.set_title('Top 10 Languages by Total Requests', fontsize=16)
    ax.set_xlabel('Total Number of Requests', fontsize=12)
    ax.set_ylabel('Language', fontsize=12)
    for bar in bars.patches:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', size=10)
    st.pyplot(fig, use_container_width=True)

    st.header("Drill-down: Top 5 Devices by Language")
    code_to_name = {v: k for k, v in LANGUAGE_CODES.items()}
    selected_language_name = st.selectbox("Select a language to see its top 5 devices:", language_names_map)
    if selected_language_name:
        selected_language_code = code_to_name[selected_language_name]
        requesters_in_language = valid_assigned_languages[valid_assigned_languages['Language Code'] == selected_language_code]['Requester'].tolist()
        language_specific_df = df[df['Requester'].isin(requesters_in_language)]
        if language_specific_df.empty:
            st.info(f"No request data found for the language: {selected_language_name}")
            return
        top_5_devices = language_specific_df['Hardware'].value_counts().nlargest(5)
        if not top_5_devices.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            bars2 = sns.barplot(x=top_5_devices.values, y=top_5_devices.index, palette='mako', ax=ax2)
            ax2.set_title(f"Top 5 Hardware Requests in {selected_language_name}", fontsize=14)
            ax2.set_xlabel("Number of Requests", fontsize=10)
            ax2.set_ylabel("Hardware Type", fontsize=10)
            for bar in bars2.patches:
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, f'{int(bar.get_width())}', va='center', ha='left', size=10)
            st.pyplot(fig2, use_container_width=True)
        else:
            st.info(f"No hardware request data found for requesters in {selected_language_name}.")

# --- Streamlit App ---
st.set_page_config(layout="wide")
st.title("Hardware Request Data Visualizer")

LANGUAGE_CODES = {
    "AB": "Arabic", "B": "English - UK", "BG": "Bulgarian", "BR": "Portuguese - Brazil",
    "C": "French Canadian", "CA": "Catalan", "CR": "Croatian", "CZ": "Czech",
    "D": "German", "DK": "Danish", "E": "Spanish", "FU": "French", "GR": "Greek",
    "H": "Norwegian", "HB": "Hebrew", "HI": "Hindi", "K": "Finnish", "KZ": "Kazakh",
    "LA": "Spanish LATAM", "LT": "Lithuanian", "MG": "Hungarian", "N": "Dutch",
    "PL": "Polish", "PO": "Portuguese", "RO": "Romanian", "RS": "Russian",
    "S": "Swedish", "SV": "Slovenian", "SK": "Slovakian", "T": "Italian",
    "TU": "Turkish", "UA": "Ukrainian", "X": "English - Australian"
}

if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'language_data' not in st.session_state:
    st.session_state.language_data = load_language_assignments()

st.sidebar.title("File Upload")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.session_state.df_cleaned = clean_data(data.copy())
    st.sidebar.success("File processed!")

tab1, tab2 = st.tabs(["Graphs", "Data Sheet"])

with tab1:
    if st.session_state.df_cleaned is not None:
        df_cleaned = st.session_state.df_cleaned
        st.sidebar.header("Graph Options and Filters")
        
        graph_options = {
            "Top 10 by Language": plot_top_languages,
            "Top 10 Requesters": plot_top_requesters,
            "Analysis by Engineer": plot_engineer_specific_analysis,
            "Total Requests per Month": plot_total_requests_per_month,
            "Requests by Hour (%) Average": plot_requests_by_hour,
            "Daily Requests by Weekday (Mon-Fri)": plot_daily_requests_by_weekday,
            "Hardware Analysis": plot_hardware_analysis,
        }
        selected_graph = st.sidebar.selectbox("Choose a graph to display", list(graph_options.keys()))

        show_monthly_breakdown = False
        show_weekday_breakdown = False
        show_daily_breakdown = False
        if selected_graph == "Requests by Hour (%) Average":
            show_monthly_breakdown = st.sidebar.checkbox("Show Monthly Breakdown", key="monthly_breakdown_cb")
        if selected_graph == "Daily Requests by Weekday (Mon-Fri)":
            show_weekday_breakdown = st.sidebar.checkbox("Show Monthly Weekday Average", key="weekday_breakdown_cb")
        if selected_graph == "Total Requests per Month":
            show_daily_breakdown = st.sidebar.checkbox("Show Daily Breakdown", key="daily_breakdown_cb")

        st.sidebar.header("Data Filters")
        min_date = df_cleaned['Time'].min().date()
        max_date = df_cleaned['Time'].max().date()
        start_date, end_date = st.sidebar.date_input('Filter by Date Range', [min_date, max_date])

        with st.sidebar.expander("Filter by Hardware"):
            all_hardware = sorted(df_cleaned['Hardware'].unique())
            select_all_hardware = st.checkbox("Select/Deselect All", value=True, key="select_all_hardware")
            selected_hardware = [hw for hw in all_hardware if st.checkbox(hw, value=select_all_hardware, key=f"hardware_{hw}")]
        
        requester_filter_option = st.sidebar.selectbox("Filter by Requester", ["Select All", "Deselect All", "Custom"], index=0)
        all_requesters = sorted(df_cleaned['Requester'].unique())
        if requester_filter_option == "Select All":
            selected_requesters = all_requesters
        elif requester_filter_option == "Deselect All":
            selected_requesters = []
        else:
            st.sidebar.subheader("Custom Requester Selection")
            select_all_custom = st.sidebar.checkbox("Select/Deselect All", value=True, key="custom_select_all")
            requester_search = st.sidebar.text_input("Search Requesters", key="custom_requester_search")
            if requester_search:
                filtered_requesters = [r for r in all_requesters if requester_search.lower() in r.lower()]
            else:
                filtered_requesters = all_requesters
            requester_container = st.sidebar.container(height=200)
            selected_requesters = [req for req in filtered_requesters if requester_container.checkbox(req, value=select_all_custom, key=f"custom_requester_{req}")]

        filtered_df = df_cleaned[(df_cleaned['Hardware'].isin(selected_hardware)) & (df_cleaned['Requester'].isin(selected_requesters)) & (df_cleaned['Time'].dt.date >= start_date) & (df_cleaned['Time'].dt.date <= end_date)]

        st.header(selected_graph)
        graph_function = graph_options[selected_graph]
        if selected_graph == "Top 10 by Language":
            graph_function(filtered_df, st.session_state.language_data)
        elif selected_graph in ["Hardware Analysis", "Analysis by Engineer"]:
            graph_function(df_cleaned)
        else:
            if not filtered_df.empty:
                graph_function(filtered_df)
            else:
                st.warning("No data available for the selected filters.")
    else:
        st.info("Please upload a CSV file using the sidebar to get started.")

with tab2:
    if st.session_state.df_cleaned is not None:
        df_cleaned = st.session_state.df_cleaned
        st.header("Requester Language Codes")
        
        unique_requesters_df = pd.DataFrame(df_cleaned['Requester'].unique(), columns=['Requester'])
        
        current_assignments = st.session_state.language_data
        if not current_assignments.empty:
            merged_df = pd.merge(unique_requesters_df, current_assignments, on='Requester', how='outer').fillna('')
        else:
            merged_df = unique_requesters_df
            merged_df['Language Code'] = ''

        merged_df.drop_duplicates(subset=['Requester'], inplace=True)

        edited_df = st.data_editor(
            merged_df,
            column_config={
                "Language Code": st.column_config.SelectboxColumn("Language Code", help="Select the language code for the requester", options=list(LANGUAGE_CODES.keys()), required=False)
            },
            hide_index=True,
            key="data_editor"
        )
        
        if st.button("Save Assignments"):
            save_language_assignments(edited_df)
            st.session_state.language_data = edited_df.copy()
            st.success("Language assignments explicitly saved to language_assignments.csv!")
    else:
        st.info("Please upload a CSV file using the sidebar to get started.")
