![Screenshot from 2025-04-13 23-26-34](https://github.com/user-attachments/assets/77a761eb-521f-48e9-ae4b-fe1325d22e15)
# Simplified Safecast Map

This project is a simplified web map visualizing radiation measurement data from Safecast. It's built using Python (FastAPI), DuckDB, and Leaflet.

## Features

* **Interactive Map:** Displays Safecast data on an interactive map using Leaflet.
* **Data Visualization:** Radiation measurements are visualized as a heatmap.
* **Data Filtering:** The map dynamically loads data based on the map view.
* **Basic Data API:** Provides simple API endpoints for retrieving measurements and statistics.
* **Tile Generation:** Generates map tiles for efficient data loading.
* **Data Import:** Supports importing data from CSV and JSON files.
* **Sample Data:** Includes the ability to generate sample data if no data is present.

## Technical Details

* **Backend:**
    * Python (FastAPI): Handles API requests, data processing, and tile generation.
    * DuckDB: A fast and lightweight analytical database used to store measurement data.
* **Frontend:**
    * Leaflet: A JavaScript library for interactive maps.
    * JavaScript: Used for fetching and displaying data on the map.
* **Data Flow:**
    1.  Data is either imported from CSV/JSON or generated as sample data.
    2.  FastAPI provides API endpoints to access the data.
    3.  Leaflet map in the browser fetches data from the API and displays it.
    4.  Map tiles are generated on the fly by the server.

## Setup Instructions

### Prerequisites

* Python 3.x
* DuckDB (Included, but may need system dependencies)
* pip (Python package manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Safecast/map2025.git](https://github.com/Safecast/map2025.git)
    cd map2025
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Start the FastAPI server:**
    ```bash
    python main_map.py
    ```
2.  **View the map:** Open your browser and go to http://localhost:8000.

### API Endpoints

* `GET /`: Serves the main HTML page.
* `GET /api/measurements`: Retrieves measurement data with optional filtering:
    * `min_lat`, `max_lat`, `min_lng`, `max_lng`: Filter by latitude and longitude.
    * `start_date`, `end_date`: Filter by date.
    * `limit`: Limit the number of results (default: 1000).
* `GET /api/stats`: Retrieves statistics about the data.
* `GET /api/tiles/{z}/{x}/{y}.png`: Retrieves a map tile for the specified zoom level and coordinates.
* `POST /api/import/csv`: Imports data from a CSV file.  Requires a `file_path` parameter in the post.
* `POST /api/import/json`: Imports data from a JSON file. Requires a `file_path` parameter in the post.

### Data Import

The application supports importing data from CSV and JSON files.  The files should contain columns/fields that map to the `measurements` table in the database (id, value, latitude, longitude, captured_at, device_id, unit).  See the `import_csv_data` and `import_json_data` functions in `main_map.py` for details on the expected data format.

Example using curl:

```bash
curl -X POST -F "file_path=./data/mydata.csv" http://localhost:8000/api/import/csv
