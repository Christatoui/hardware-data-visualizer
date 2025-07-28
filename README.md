# Hardware Request Data Visualizer

This Streamlit application allows you to upload a CSV file of hardware requests and visualize the data in various ways.

## How to Use

### 1. Upload Your Data
-   The **File Upload** control is located in the sidebar on the left.
-   Click the "Browse files" button and select your CSV file. The application expects columns named `Hardware`, `Requester`, and `Time`.
-   Once the file is uploaded and processed, you will see a "File processed!" success message.

### 2. View Graphs
-   The **Graphs** tab is the main view for data visualization.
-   Use the **Graph Options and Filters** section in the sidebar to customize the view:
    -   **Choose a graph to display:** Select from a variety of graphs, such as "Top 10 Requesters" or "Total Requests per Month."
    -   **Breakdown Options:** For certain graphs, additional checkboxes will appear, allowing you to see more detailed monthly or daily breakdowns.
    -   **Data Filters:**
        -   **Filter by Date Range:** Select a start and end date to narrow the data.
        -   **Filter by Hardware:** Use the expandable section to select or deselect specific hardware types.
        -   **Filter by Requester:** Choose "Select All," "Deselect All," or "Custom" to filter by requesters. The custom option provides a searchable list with checkboxes.

### 3. Assign Language Codes
-   The **Data Sheet** tab allows you to assign a language code to each unique requester. This data is used in the "Top 10 by Language" graph.
-   The application uses a persistent storage system for these assignments.
    -   **Load Assignments:** When the application starts, it automatically loads any previously saved assignments.
    -   **Edit Assignments:** Use the dropdown menu in the "Language Code" column to assign a code to each requester.
    -   **Save Assignments:** Click the "Save Assignments" button to save your changes. The assignments are stored in a file named `language_assignments.csv` and will be available the next time you use the app.

### 4. Code Backups
-   This directory contains versioned backups of the application's source code (`app_v1.0.py`, `app_v2.0.py`).
-   If you encounter any major issues, you can revert to a previous version by renaming one of these files to `app.py`.
