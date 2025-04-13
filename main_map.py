import os
import json
import pandas as pd
import numpy as np
import duckdb
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import uvicorn
import time
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import io
import base64
from PIL import Image
import shutil
import requests
from starlette.middleware.cors import CORSMiddleware
import matplotlib.tri as tri 

# Create necessary directories
os.makedirs("static", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("tiles", exist_ok=True)
os.makedirs("cache", exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Simplified Safecast Map")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize templates
templates = Jinja2Templates(directory="static")

# Initialize DuckDB connection
conn = duckdb.connect('data/safecast.duckdb')

# Initialize database tables
def init_database():
    conn.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER,
            value DOUBLE,
            latitude DOUBLE,
            longitude DOUBLE,
            captured_at TIMESTAMP,
            device_id INTEGER,
            unit VARCHAR
        )
    """)
    
    # Create spatial index
    conn.execute("""
        CREATE TABLE IF NOT EXISTS spatial_index (
            tile_z INTEGER,
            tile_x INTEGER,
            tile_y INTEGER,
            measurement_ids INTEGER[]
        )
    """)
    
    # Create devices table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            model VARCHAR
        )
    """)

init_database()

# Helper functions for tile calculations
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = np.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = np.arctan(np.sinh(np.pi * (1 - 2 * ytile / n)))
    lat_deg = np.degrees(lat_rad)
    return (lat_deg, lon_deg)

def get_tile_bounds(z, x, y):
    # Get the bounds of the tile
    nw_lat, nw_lng = num2deg(x, y, z)
    se_lat, se_lng = num2deg(x + 1, y + 1, z)
    return {
        'min_lat': se_lat,
        'max_lat': nw_lat,
        'min_lng': nw_lng,
        'max_lng': se_lng
    }

# Data import functions
def import_csv_data(file_path):
    try:
        # Read CSV into pandas DataFrame
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['value', 'latitude', 'longitude']
        if not all(col in df.columns for col in required_columns):
            print(f"CSV file is missing required columns: {required_columns}")
            return False
        
        # Make sure we have at least these columns with default values if needed
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
        if 'device_id' not in df.columns:
            df['device_id'] = 1
        if 'captured_at' not in df.columns:
            df['captured_at'] = datetime.datetime.now()
        if 'unit' not in df.columns:
            df['unit'] = 'CPM'
        
        # Insert into DuckDB
        conn.execute("INSERT INTO measurements SELECT * FROM df")
        print(f"Imported {len(df)} records from {file_path}")
        return True
    except Exception as e:
        print(f"Error importing CSV: {e}")
        return False

def import_json_data(file_path):
    try:
        # Read JSON
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if it's an array of objects or a single object
        if isinstance(data, dict):
            data = [data]
        
        # Create DataFrame
        df = pd.json_normalize(data)
        
        # Map JSON fields to database schema
        # This depends on the exact structure of your JSON
        required_fields = ['value', 'latitude', 'longitude']
        
        # Check for nested fields
        field_mapping = {}
        for field in required_fields:
            found = False
            if field in df.columns:
                found = True
            else:
                # Check for nested fields
                for col in df.columns:
                    if col.endswith(f".{field}"):
                        field_mapping[field] = col
                        found = True
                        break
            
            if not found:
                print(f"JSON file is missing required field: {field}")
                return False
        
        # Apply field mapping
        for target, source in field_mapping.items():
            df[target] = df[source]
        
        # Set default values for missing fields
        if 'id' not in df.columns:
            df['id'] = range(1, len(df) + 1)
        if 'device_id' not in df.columns:
            df['device_id'] = 1
        if 'captured_at' not in df.columns:
            df['captured_at'] = datetime.datetime.now()
        if 'unit' not in df.columns:
            df['unit'] = 'CPM'
        
        # Insert into DuckDB
        conn.execute("INSERT INTO measurements SELECT * FROM df")
        print(f"Imported {len(df)} records from {file_path}")
        return True
    except Exception as e:
        print(f"Error importing JSON: {e}")
        return False

# [Previous Python code remains the same]
# Make sure 'import matplotlib.pyplot as plt' and 'from matplotlib.colors import LinearSegmentedColormap' are present
import matplotlib.colors as mcolors # Needed for Normalize

# Function to generate heatmap tiles - UPDATED COLORMAP & NORMALIZATION
def generate_tile(z, x, y, width=256, height=256):
    bounds = get_tile_bounds(z, x, y)

    lat_range = bounds['max_lat'] - bounds['min_lat']
    lng_range = bounds['max_lng'] - bounds['min_lng']
    buffer_factor = 0.25
    buffered_bounds = {
        'min_lat': bounds['min_lat'] - lat_range * buffer_factor,
        'max_lat': bounds['max_lat'] + lat_range * buffer_factor,
        'min_lng': bounds['min_lng'] - lng_range * buffer_factor,
        'max_lng': bounds['max_lng'] + lng_range * buffer_factor
    }

    query = f"""
        SELECT latitude, longitude, value
        FROM measurements
        WHERE latitude >= {buffered_bounds['min_lat']}
          AND latitude <= {buffered_bounds['max_lat']}
          AND longitude >= {buffered_bounds['min_lng']}
          AND longitude <= {buffered_bounds['max_lng']}
    """
    result = conn.execute(query).fetchdf()

    if len(result) == 0:
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        img_path = f"tiles/{z}/{x}/{y}.png"
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)
        return img_path

    dpi = 100
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    x_pixels = (result['longitude'] - bounds['min_lng']) / lng_range * width
    y_pixels = (1 - (result['latitude'] - bounds['min_lat']) / lat_range) * height

    # Use logarithmic scaling for radiation values
    values = np.log1p(result['value'].clip(lower=0)) # log1p(x) = log(1+x)

    # *** DEFINE COLORMAP AND NORMALIZATION ***
    # Define the green-yellow-red colormap
    cmap_colors = [(0, 0.6, 0), (0.9, 0.9, 0), (0.9, 0, 0)] # Green, Yellow, Red
    cmap = LinearSegmentedColormap.from_list('safecast_gyr', cmap_colors)

    # Define normalization range based on LOG values (log1p scale)
    # Adjust these based on expected range of CPM values in your data
    # Example: log1p(0)=0, log1p(100)~=4.6, log1p(1000)~=6.9, log1p(5000)~=8.5
    log_vmin = 0.0  # Corresponds to 0 CPM
    log_vmax = 7.0  # Corresponds to approx exp(7)-1 ~= 1095 CPM. Adjust as needed.
    norm = mcolors.Normalize(vmin=log_vmin, vmax=log_vmax)
    # *** END DEFINITION ***

    marker_size = 300
    alpha_value = 0.15

    # Apply the colormap and normalization to scatter
    scatter = ax.scatter(
        x_pixels, y_pixels,
        c=values,        # Color based on log values
        s=marker_size,
        cmap=cmap,       # Apply the G-Y-R colormap
        alpha=alpha_value,
        norm=norm,       # Apply the normalization
        marker='o',
        linewidths=0
    )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, dpi=dpi)
    plt.close(fig)

    try:
        img = Image.open(buf)
    except Exception as e:
        print(f"Error opening image buffer for tile {z}/{x}/{y}: {e}")
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    img_path = f"tiles/{z}/{x}/{y}.png"
    os.makedirs(os.path.dirname(img_path), exist_ok=True)
    try:
        img.save(img_path)
    except Exception as e:
        print(f"Error saving tile image {img_path}: {e}")
        Path(img_path).touch()

    return img_path

# [Rest of the Python code remains the same]

# API Routes
@app.get("/", response_class=HTMLResponse) # Define response class directly
async def root(request: Request):
    """Serve the main HTML page from the default_html variable"""
    # Directly return the content of the variable
    return HTMLResponse(content=default_html)


@app.get("/api/measurements")
async def get_measurements(
    min_lat: Optional[float] = None,
    max_lat: Optional[float] = None,
    min_lng: Optional[float] = None,
    max_lng: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000
):
    """Get measurements with optional filtering"""
    query = "SELECT * FROM measurements"
    conditions = []
    
    if min_lat is not None:
        conditions.append(f"latitude >= {min_lat}")
    if max_lat is not None:
        conditions.append(f"latitude <= {max_lat}")
    if min_lng is not None:
        conditions.append(f"longitude >= {min_lng}")
    if max_lng is not None:
        conditions.append(f"longitude <= {max_lng}")
    if start_date is not None:
        conditions.append(f"captured_at >= '{start_date}'")
    if end_date is not None:
        conditions.append(f"captured_at <= '{end_date}'")
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += f" LIMIT {limit}"
    
    result = conn.execute(query).fetchdf()
    
    return JSONResponse(content=json.loads(result.to_json(orient="records")))

@app.get("/api/stats")
async def get_stats():
    """Get general statistics about the data"""
    stats = {}
    
    # Total number of measurements
    stats["total_measurements"] = conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
    
    # Min, max, and average values
    value_stats = conn.execute("""
        SELECT 
            MIN(value) as min_value, 
            MAX(value) as max_value, 
            AVG(value) as avg_value
        FROM measurements
    """).fetchone()
    
    stats["min_value"] = value_stats[0]
    stats["max_value"] = value_stats[1]
    stats["avg_value"] = value_stats[2]
    
    # Date range
    date_range = conn.execute("""
        SELECT 
            MIN(captured_at) as first_date, 
            MAX(captured_at) as last_date
        FROM measurements
    """).fetchone()
    
    stats["first_date"] = str(date_range[0])
    stats["last_date"] = str(date_range[1])
    
    # Number of devices
    stats["device_count"] = conn.execute("SELECT COUNT(DISTINCT device_id) FROM measurements").fetchone()[0]
    
    return stats

@app.get("/api/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    """Get a map tile for the given coordinates"""
    # Check if tile exists in cache
    tile_path = f"tiles/{z}/{x}/{y}.png"
    
    if not os.path.exists(tile_path):
        # Generate the tile
        tile_path = generate_tile(z, x, y)
    
    # Return the tile
    return FileResponse(tile_path, media_type="image/png")

@app.post("/api/import/csv")
async def import_csv(file_path: str):
    """Import data from a CSV file"""
    success = import_csv_data(file_path)
    return {"success": success}

@app.post("/api/import/json")
async def import_json(file_path: str):
    """Import data from a JSON file"""
    success = import_json_data(file_path)
    return {"success": success}


# ---------------------------------------------------------------

# [Existing Python code before default_html remains the same]

# Default HTML content for the main page - CORRECTED DATE PARSING (* 1000)
default_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Simplified Safecast Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
    <style>
        /* Using user-tested styles with !important */
        body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
        #map { width: 100%; height: 100vh; }
        .info { padding: 6px 8px !important; font: 14px/16px Arial, Helvetica, sans-serif !important; box-shadow: 0 0 15px rgba(0,0,0,0.2) !important; border-radius: 5px !important; color: white !important; background-color: black !important; }
        .info h4 { margin: 0 0 5px !important; color: #fff !important; }
        .legend { line-height: 18px !important; color: white !important; }
        .legend i { width: 18px !important; height: 18px !important; float: left !important; margin-right: 8px !important; opacity: 0.7 !important; border: 1px solid #555 !important; }
        .leaflet-tooltip-hover { background-color: rgba(0, 0, 0, 0.85) !important; color: white !important; border: none !important; box-shadow: none !important; padding: 5px !important; font-size: 12px !important; }
        .leaflet-bar a, .leaflet-bar a:hover { background-color: #0f0f0f !important; border-bottom: 1px solid #ccc !important; width: 26px !important; height: 26px !important; line-height: 26px !important; display: block !important; text-align: center !important; text-decoration: none !important; color: #fff !important; }
        .leaflet-bar a:first-child { border-top-left-radius: 4px !important; border-top-right-radius: 4px !important; }
        .leaflet-bar a:last-child { border-bottom-left-radius: 4px !important; border-bottom-right-radius: 4px !important; border-bottom: none !important; }
        .leaflet-control-attribution { background: rgba(15, 15, 15, 0.8) !important; color: #ccc !important; }
        .leaflet-control-attribution a { color: #aaa !important; }
        .leaflet-control-scale-line { border: 2px solid #555 !important; border-top: none !important; color: #eee !important; background: rgba(15, 15, 15, 0.7) !important; padding: 2px 5px !important; box-shadow: none !important; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000 !important; }
        #map.leaflet-container.pointer-cursor { cursor: pointer !important; }
        .leaflet-popup-content-wrapper { background-color: #333; color: #eee; border-radius: 5px;}
        .leaflet-popup-content { margin: 13px 19px; line-height: 1.4; }
        .leaflet-popup-tip { background: #333; }
        .leaflet-popup-close-button { color: #bbb !important; }
    </style>
</head>
<body>
    <div id="map"></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
    <script>
        // --- JavaScript Code ---
        const map = L.map('map').setView([37.5, 140.0], 6);
        const mapContainer = map.getContainer();
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' }).addTo(map);
        const safecastLayer = L.tileLayer('/api/tiles/{z}/{x}/{y}.png', { maxZoom: 18, tms: false, opacity: 0.65, attribution: 'Data: <a href="https://safecast.org">Safecast</a>' }).addTo(map);

        let currentPoints = []; let hoverTooltip = null; let mouseLatLng = null; let hoveredPointData = null; let pinnedPopup = null;

        // --- Functions (fetchPointsForView, findClosestPoint, debounce, hideHoverTooltip - unchanged) ---
        async function fetchPointsForView() { const bounds = map.getBounds(); const url = `/api/measurements?min_lat=${bounds.getSouth()}&max_lat=${bounds.getNorth()}&min_lng=${bounds.getWest()}&max_lng=${bounds.getEast()}&limit=1000`; try { const response = await fetch(url); if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); } currentPoints = await response.json(); } catch (error) { console.error("Error fetching points for hover:", error); currentPoints = []; } }
        function findClosestPoint(latlng) { let closestPoint = null; let minDistanceSq = Infinity; const baseThreshold = 0.0005; const thresholdDistanceSq = baseThreshold * Math.pow(0.85, map.getZoom()); currentPoints.forEach(point => { if (point.latitude && point.longitude) { const dx = latlng.lng - point.longitude; const dy = latlng.lat - point.latitude; const distanceSq = dx * dx + dy * dy; if (distanceSq < thresholdDistanceSq) { if (distanceSq < minDistanceSq) { minDistanceSq = distanceSq; closestPoint = point; } } } }); return closestPoint; }
        function debounce(func, wait) { let timeout; return function executedFunction(...args) { const later = () => { clearTimeout(timeout); func(...args); }; clearTimeout(timeout); timeout = setTimeout(later, wait); }; }
        function hideHoverTooltip() { if (hoverTooltip && map.hasLayer(hoverTooltip)) { map.removeLayer(hoverTooltip); hoverTooltip = null; } }

        // --- updateTooltip function (unchanged) ---
        function updateTooltip(latlng) { if (!latlng) return; const closestPoint = findClosestPoint(latlng); if (closestPoint) { mapContainer.classList.add('pointer-cursor'); hoveredPointData = closestPoint; if (pinnedPopup && pinnedPopup.options.customId === closestPoint.id) { hideHoverTooltip(); return; } const content = `<i>Hover - Click to Pin</i><br><b>Value:</b> ${closestPoint.value.toFixed(2)} ...`; if (!hoverTooltip) { hoverTooltip = L.tooltip({ permanent: false, sticky: true, direction: 'top', offset: L.point(0, -10), className: 'leaflet-tooltip-hover' }).setLatLng(latlng).setContent(content).addTo(map); } else { hoverTooltip.setLatLng(latlng).setContent(content); if (!map.hasLayer(hoverTooltip)) { hoverTooltip.addTo(map); } } } else { mapContainer.classList.remove('pointer-cursor'); hoveredPointData = null; hideHoverTooltip(); } }

        // --- Event Listeners ---
        map.on('load zoomend moveend', debounce(fetchPointsForView, 500));
        map.on('mousemove', debounce((e) => { mouseLatLng = e.latlng; updateTooltip(mouseLatLng); }, 50));
        map.on('mouseout', () => { mapContainer.classList.remove('pointer-cursor'); hideHoverTooltip(); hoveredPointData = null; mouseLatLng = null; });

        // ** MODIFIED map click listener **
        map.on('click', function(e) {
            if (pinnedPopup) { map.removeLayer(pinnedPopup); pinnedPopup = null; }

            if (hoveredPointData) {
                const point = hoveredPointData;
                // *** Convert seconds to milliseconds by multiplying by 1000 ***
                const dateObject = new Date(point.captured_at * 1000);
                // *** Check if date is valid before formatting ***
                const dateString = !isNaN(dateObject) ? dateObject.toISOString() : "Invalid Date";
                // *** End change ***

                const popupContent = `
                    <b>Value:</b> ${point.value.toFixed(2)} ${point.unit || 'CPM'}<br>
                    <b>Date:</b> ${dateString}<br> 
                    <b>Coords:</b> ${point.latitude.toFixed(4)}, ${point.longitude.toFixed(4)}
                    ${point.device_id ? `<br><b>Device:</b> ${point.device_id}` : ''}
                `;

                pinnedPopup = L.popup({ closeButton: true, autoClose: false, closeOnClick: false, keepInView: true, customId: point.id })
                    .setLatLng([point.latitude, point.longitude])
                    .setContent(popupContent)
                    .openOn(map);

                hideHoverTooltip();
                hoveredPointData = null;
            }
        });

        fetchPointsForView(); // Initial fetch

        // --- Controls (unchanged) ---
        const info = L.control({position: 'topright'}); info.onAdd = function (map) { this._div = L.DomUtil.create('div', 'info'); this.update(); return this._div; }; info.update = function (props) { this._div.innerHTML = '<h4>Simplified Safecast Map</h4><p>Hover over data areas for details</p>'; }; info.addTo(map);
        const legend = L.control({position: 'bottomright'}); const legendColors = { green: '#009900', yellow: '#E6E600', red: '#E60000' }; function getLegendColor(cpmValue) { if (cpmValue <= 100) { return legendColors.green; } else if (cpmValue <= 500) { return legendColors.yellow; } else { return legendColors.red; } } legend.onAdd = function (map) { const div = L.DomUtil.create('div', 'info legend leaflet-control'); const grades = [0, 100, 500, 1000]; div.innerHTML += '<b>Radiation (CPM)</b><br>'; for (let i = 0; i < grades.length; i++) { const gradeValue = grades[i]; const color = getLegendColor(gradeValue); div.innerHTML += '<i style="background:' + color + '; opacity: 0.8;"></i> ' + gradeValue + (grades[i + 1] ? '&ndash;' + grades[i + 1] + '<br>' : '+'); } return div; }; legend.addTo(map);
        L.control.scale({imperial: false, position: 'bottomleft'}).addTo(map);
        // --- End JavaScript Code ---
    </script>
</body>
</html>
"""

# [Existing Python code after default_html remains the same]

# ---------------------------------------------------------------

# In main_map.py.txt
import pandas as pd # Ensure pandas is imported
import numpy as np  # Ensure numpy is imported

# Sample data import function - MODIFIED TO ADD RANDOM TIMES
def import_sample_data():
    """Import sample data if no data exists"""
    count = conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
    if count == 0:
        print("No data found, importing sample data with random times...")

        center_lat, center_lng = 35.6895, 139.6917  # Tokyo
        num_points = 500

        np.random.seed(42)

        lats = center_lat + np.random.normal(0, 0.5, num_points)
        lngs = center_lng + np.random.normal(0, 0.5, num_points)

        fukushima_lat, fukushima_lng = 37.4216, 141.0329
        values = []
        for lat, lng in zip(lats, lngs):
            dist = np.sqrt((lat - fukushima_lat)**2 + (lng - fukushima_lng)**2)
            value = np.random.gamma(shape=1.5, scale=30) + 500 * np.exp(-dist * 2)
            values.append(value)

        # *** MODIFIED TIMESTAMP GENERATION ***
        # Create base dates (still at midnight)
        base_dates = pd.date_range(start='2023-01-01', periods=num_points, tz='UTC') # Specify UTC
        # Create random time offsets (in seconds for a full day)
        random_seconds = np.random.randint(0, 86400, num_points)
        # Add random time offsets to base dates
        timestamps = base_dates + pd.to_timedelta(random_seconds, unit='s')
        # *** END MODIFIED TIMESTAMP GENERATION ***

        df = pd.DataFrame({
            'id': range(1, num_points + 1),
            'value': values,
            'latitude': lats,
            'longitude': lngs,
            # Use the new timestamps with time components
            'captured_at': timestamps,
            'device_id': np.random.randint(1, 10, num_points),
            'unit': 'CPM'
        })

        # Optional: Verify timestamp format before insert
        # print("Sample Timestamps (first 5):")
        # print(df['captured_at'].head())

        # Import to database
        # Ensure table exists before inserting
        init_database() # Make sure init_database was called or call again
        try:
            conn.execute("INSERT INTO measurements SELECT * FROM df")
            print(f"Imported {len(df)} sample records")
        except Exception as e:
            print(f"Error inserting sample data: {e}")
            print("DataFrame dtypes:", df.dtypes)


# [Rest of the Python code, including calling import_sample_data(), remains the same]

# Run sample data import
import_sample_data()

# Main function
def main():
    print("Starting Simplified Safecast Map Server...")
    print("To view the map, open http://localhost:8000 in your browser")
    print("API endpoints:")
    print("  - GET /api/measurements - Get measurement data")
    print("  - GET /api/stats - Get data statistics")
    print("  - GET /api/tiles/{z}/{x}/{y}.png - Get map tiles")
    print("  - POST /api/import/csv - Import CSV data")
    print("  - POST /api/import/json - Import JSON data")
    print("  - POST /test") 
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
