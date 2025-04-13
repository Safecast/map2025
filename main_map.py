import os
import json
import pandas as pd
import numpy as np
import duckdb
# Import Form along with Query, Request etc.
from fastapi import FastAPI, HTTPException, Query, Request, Form
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
# Removed StaticFiles and Jinja2Templates as they are not used directly anymore
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
from typing import List, Optional
import uvicorn
import time
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors # Needed for Normalize
# import matplotlib.cm as cm # Not used directly
import io
# import base64 # Not used
from PIL import Image
# import shutil # Not used
# import requests # Not used
from starlette.middleware.cors import CORSMiddleware
import math # Needed for tile calculations
import re # Needed for coordinate parsing

# Create necessary directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("tiles", exist_ok=True)

# Initialize FastAPI
app = FastAPI(title="Simplified Safecast Map")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Moved default_html definition here, BEFORE it's used in the root route
# Default HTML content for the main page - Includes all JS and CSS
# (Using the last working version with dark UI elements and fixed legend)
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
        .legend { line-height: 1.5 !important; color: white !important; }
        .legend i { display: inline-block; vertical-align: middle; width: 18px !important; height: 18px !important; margin-right: 8px !important; opacity: 0.8 !important; border: 1px solid #555 !important; }
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
        // --- Leaflet Map Initialization ---
        const map = L.map('map').setView([37.5, 140.0], 6);
        const mapContainer = map.getContainer();
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' }).addTo(map);
        const safecastLayer = L.tileLayer('/api/tiles/{z}/{x}/{y}.png', { maxZoom: 18, tms: false, opacity: 0.65, attribution: 'Data: <a href="https://safecast.org">Safecast</a>' }).addTo(map);
        let currentPoints = []; let hoverTooltip = null; let mouseLatLng = null; let hoveredPointData = null; let pinnedPopup = null;
        async function fetchPointsForView() { const bounds = map.getBounds(); const url = `/api/measurements?min_lat=${bounds.getSouth()}&max_lat=${bounds.getNorth()}&min_lng=${bounds.getWest()}&max_lng=${bounds.getEast()}&limit=1000`; try { const response = await fetch(url); if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); } currentPoints = await response.json(); } catch (error) { console.error("Error fetching points for hover:", error); currentPoints = []; } }
        function findClosestPoint(latlng) { let closestPoint = null; let minDistanceSq = Infinity; const baseThreshold = 0.0005; const thresholdDistanceSq = baseThreshold * Math.pow(0.85, map.getZoom()); currentPoints.forEach(point => { if (point.latitude && point.longitude) { const dx = latlng.lng - point.longitude; const dy = latlng.lat - point.latitude; const distanceSq = dx * dx + dy * dy; if (distanceSq < thresholdDistanceSq) { if (distanceSq < minDistanceSq) { minDistanceSq = distanceSq; closestPoint = point; } } } }); return closestPoint; }
        function hideHoverTooltip() { if (hoverTooltip && map.hasLayer(hoverTooltip)) { map.removeLayer(hoverTooltip); hoverTooltip = null; } }
        function updateTooltip(latlng) { if (!latlng) return; const closestPoint = findClosestPoint(latlng); if (closestPoint) { mapContainer.classList.add('pointer-cursor'); hoveredPointData = closestPoint; if (pinnedPopup && pinnedPopup.options.customId === closestPoint.id) { hideHoverTooltip(); return; } const content = `<i>Hover - Click to Pin</i><br><b>Value:</b> ${closestPoint.value.toFixed(2)} ...`; if (!hoverTooltip) { hoverTooltip = L.tooltip({ permanent: false, sticky: true, direction: 'top', offset: L.point(0, -10), className: 'leaflet-tooltip-hover' }).setLatLng(latlng).setContent(content).addTo(map); } else { hoverTooltip.setLatLng(latlng).setContent(content); if (!map.hasLayer(hoverTooltip)) { hoverTooltip.addTo(map); } } } else { mapContainer.classList.remove('pointer-cursor'); hoveredPointData = null; hideHoverTooltip(); } }
        function debounce(func, wait) { let timeout; return function executedFunction(...args) { const later = () => { clearTimeout(timeout); func(...args); }; clearTimeout(timeout); timeout = setTimeout(later, wait); }; }
        map.on('load zoomend moveend', debounce(fetchPointsForView, 500));
        map.on('mousemove', debounce((e) => { mouseLatLng = e.latlng; updateTooltip(mouseLatLng); }, 50));
        map.on('mouseout', () => { mapContainer.classList.remove('pointer-cursor'); hideHoverTooltip(); hoveredPointData = null; mouseLatLng = null; });
        map.on('click', function(e) { if (pinnedPopup) { map.removeLayer(pinnedPopup); pinnedPopup = null; } if (hoveredPointData) { const point = hoveredPointData; const dateObject = new Date(point.captured_at * 1000); const dateString = !isNaN(dateObject) ? dateObject.toISOString() : "Invalid Date"; const popupContent = `<b>Value:</b> ${point.value.toFixed(2)} ${point.unit || 'CPM'}<br><b>Date:</b> ${dateString}<br><b>Coords:</b> ${point.latitude.toFixed(4)}, ${point.longitude.toFixed(4)}${point.device_id ? `<br><b>Device:</b> ${point.device_id}` : ''}`; pinnedPopup = L.popup({ closeButton: true, autoClose: false, closeOnClick: false, keepInView: true, customId: point.id }).setLatLng([point.latitude, point.longitude]).setContent(popupContent).openOn(map); hideHoverTooltip(); hoveredPointData = null; } });
        fetchPointsForView();
        const info = L.control({position: 'topright'}); info.onAdd = function (map) { this._div = L.DomUtil.create('div', 'info'); this.update(); return this._div; }; info.update = function (props) { this._div.innerHTML = '<h4>Simplified Safecast Map</h4><p>Hover over data areas for details</p>'; }; info.addTo(map);
        const legend = L.control({position: 'bottomright'}); const legendColors = { green: '#009900', yellow: '#E6E600', red: '#E60000' }; function getLegendColor(cpmValue) { if (cpmValue <= 100) { return legendColors.green; } else if (cpmValue <= 500) { return legendColors.yellow; } else { return legendColors.red; } } legend.onAdd = function (map) { const div = L.DomUtil.create('div', 'info legend leaflet-control'); const grades = [0, 100, 500, 1000]; div.innerHTML += '<b>Radiation (CPM)</b><br>'; for (let i = 0; i < grades.length; i++) { const gradeValue = grades[i]; const color = getLegendColor(gradeValue); const label = gradeValue + (grades[i + 1] ? '&ndash;' + grades[i + 1] : '+'); div.innerHTML += '<i style="background:' + color + ';"></i> ' + label + '<br>'; } return div; }; legend.addTo(map);
        L.control.scale({imperial: false, position: 'bottomleft'}).addTo(map);
    </script>
</body>
</html>
"""

# Initialize DuckDB connection
conn = duckdb.connect('data/safecast.duckdb', read_only=False)

# Initialize database tables
def init_database():
    """Creates database tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY, -- Make ID the primary key
            value DOUBLE,
            latitude DOUBLE,
            longitude DOUBLE,
            captured_at TIMESTAMP,
            device_id INTEGER,
            unit VARCHAR
        )
    """)
    print("Database tables checked/created.")

init_database() # Ensure tables exist

# --- Helper Functions ---

def deg2num(lat_deg, lon_deg, zoom):
    """Converts lat/lon degrees to tile numbers."""
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    """Converts tile numbers to lat/lon degrees (northwest corner)."""
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def get_tile_bounds(z, x, y):
    """Calculates the geographical bounds of a given map tile."""
    nw_lat, nw_lng = num2deg(x, y, z)
    se_lat, se_lng = num2deg(x + 1, y + 1, z)
    return {'min_lat': se_lat, 'max_lat': nw_lat, 'min_lng': nw_lng, 'max_lng': se_lng}

# Helper function to convert DDMM.MMMM or DDDMM.MMMM to Decimal Degrees
def convert_dmm_to_dd(dmm_str, direction):
    """Converts Degree-Decimal-Minute format to Decimal Degrees."""
    dmm_str = str(dmm_str).strip()
    direction = str(direction).strip().upper()
    decimal_point = dmm_str.find('.')

    if decimal_point == -1: return None
    if decimal_point < 2: return None
    minutes_str = dmm_str[decimal_point-2:]
    degrees_str = dmm_str[:decimal_point-2]

    try:
        degrees = float(degrees_str) if degrees_str else 0.0
        minutes = float(minutes_str)
        decimal_degrees = degrees + (minutes / 60.0)
        if direction == 'S' or direction == 'W': decimal_degrees *= -1
        elif direction not in ['N', 'E']: return None
        return decimal_degrees
    except ValueError: return None

# Function to import data from the specific CSV format
def import_csv_data(file_path):
    """Imports data from a CSV file with a specific NMEA-like format ($BNRDD)."""
    if not os.path.exists(file_path): print(f"Error: File not found at {file_path}"); return False
    parsed_rows = [];
    try: start_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM measurements").fetchone()[0]
    except Exception as db_err: print(f"Error getting max ID from database: {db_err}"); return False
    row_counter = start_id
    print(f"Starting import from {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'): continue
                if line.startswith('$BNRDD,'):
                    line = line.split('*')[0]; fields = line.split(',')
                    if len(fields) < 11: print(f"Warning: Skipping malformed $BNRDD line {line_num}: Too few fields ({len(fields)})"); continue
                    try:
                        device_id_str = fields[1]; timestamp_str = fields[2]; value_str = fields[3]
                        lat_str = fields[7]; lat_dir = fields[8]; lon_str = fields[9]; lon_dir = fields[10]
                        device_id = int(device_id_str); timestamp = datetime.datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        value = float(value_str); latitude = convert_dmm_to_dd(lat_str, lat_dir); longitude = convert_dmm_to_dd(lon_str, lon_dir)
                        if latitude is None or longitude is None: print(f"Warning: Skipping line {line_num} due to invalid coordinate format: Lat='{lat_str}{lat_dir}', Lon='{lon_str}{lon_dir}'"); continue
                        parsed_rows.append({'id': row_counter, 'value': value, 'latitude': latitude, 'longitude': longitude, 'captured_at': timestamp, 'device_id': device_id, 'unit': 'CPM' })
                        row_counter += 1
                    except (IndexError, ValueError, TypeError) as e: print(f"Warning: Skipping malformed $BNRDD line {line_num}: {e} | Data: {fields}"); continue
        if not parsed_rows: print("No valid $BNRDD data found in the file."); return False
        df = pd.DataFrame(parsed_rows); print(f"Successfully parsed {len(df)} records.")
        conn.execute("CREATE OR REPLACE TEMP TABLE tmp_import_data AS SELECT * FROM df")
        conn.execute("INSERT INTO measurements SELECT * FROM tmp_import_data ON CONFLICT(id) DO NOTHING")
        conn.execute("DROP TABLE tmp_import_data"); print(f"Successfully inserted {len(df)} records into the database."); return True
    except FileNotFoundError: print(f"Error: File not found at {file_path}"); return False
    except Exception as e: print(f"An unexpected error occurred during CSV import: {e}"); return False

# Function to generate a heatmap tile
def generate_tile(z, x, y, width=256, height=256):
    tile_path = f"tiles/{z}/{x}/{y}.png"; tile_dir = os.path.dirname(tile_path); os.makedirs(tile_dir, exist_ok=True)
    bounds = get_tile_bounds(z, x, y); lat_range = bounds['max_lat'] - bounds['min_lat']; lng_range = bounds['max_lng'] - bounds['min_lng']
    buffer_factor = 0.25; buffered_bounds = { 'min_lat': bounds['min_lat'] - lat_range * buffer_factor, 'max_lat': bounds['max_lat'] + lat_range * buffer_factor, 'min_lng': bounds['min_lng'] - lng_range * buffer_factor, 'max_lng': bounds['max_lng'] + lng_range * buffer_factor }
    query = f"SELECT latitude, longitude, value FROM measurements WHERE latitude >= {buffered_bounds['min_lat']} AND latitude <= {buffered_bounds['max_lat']} AND longitude >= {buffered_bounds['min_lng']} AND longitude <= {buffered_bounds['max_lng']}"
    try: result = conn.execute(query).fetchdf()
    except Exception as e: print(f"Error querying data for tile {z}/{x}/{y}: {e}"); result = pd.DataFrame()
    if result.empty: img = Image.new('RGBA', (width, height), (0, 0, 0, 0)); img.save(tile_path); return tile_path
    dpi = 100; fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi); ax = fig.add_subplot(111)
    x_pixels = (result['longitude'] - bounds['min_lng']) / lng_range * width; y_pixels = (1 - (result['latitude'] - bounds['min_lat']) / lat_range) * height
    values = np.log1p(result['value'].clip(lower=0)); cmap_colors = [(0, 0.6, 0), (0.9, 0.9, 0), (0.9, 0, 0)]; cmap = LinearSegmentedColormap.from_list('safecast_gyr', cmap_colors)
    log_vmin = 0.0; log_vmax = 7.0; norm = mcolors.Normalize(vmin=log_vmin, vmax=log_vmax); marker_size = 300; alpha_value = 0.15
    scatter = ax.scatter( x_pixels, y_pixels, c=values, s=marker_size, cmap=cmap, alpha=alpha_value, norm=norm, marker='o', linewidths=0 )
    ax.set_xlim(0, width); ax.set_ylim(height, 0); ax.axis('off'); fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    buf = io.BytesIO(); fig.savefig(buf, format='png', transparent=True, dpi=dpi); plt.close(fig); buf.seek(0)
    try: img = Image.open(buf); img.save(tile_path)
    except Exception as e: print(f"Error saving tile image {tile_path}: {e}"); Path(tile_path).touch(); return None
    return tile_path

# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page from the default_html variable"""
    return HTMLResponse(content=default_html)

# Corrected get_measurements function
@app.get("/api/measurements")
async def get_measurements( min_lat: Optional[float] = Query(None), max_lat: Optional[float] = Query(None), min_lng: Optional[float] = Query(None), max_lng: Optional[float] = Query(None), start_date: Optional[str] = Query(None), end_date: Optional[str] = Query(None), limit: int = Query(1000) ):
    """Get measurements with optional filtering. Returns epoch seconds for timestamp."""
    query = "SELECT id, value, latitude, longitude, epoch(captured_at) as captured_at, device_id, unit FROM measurements"
    conditions = []; params = {}
    if min_lat is not None: conditions.append("latitude >= $min_lat"); params['min_lat'] = min_lat
    if max_lat is not None: conditions.append("latitude <= $max_lat"); params['max_lat'] = max_lat
    if min_lng is not None: conditions.append("longitude >= $min_lng"); params['min_lng'] = min_lng
    if max_lng is not None: conditions.append("longitude <= $max_lng"); params['max_lng'] = max_lng
    if start_date is not None:
        try: datetime.datetime.strptime(start_date, '%Y-%m-%d'); conditions.append("captured_at >= $start_date"); params['start_date'] = start_date
        except ValueError: raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD.")
    if end_date is not None:
        try: datetime.datetime.strptime(end_date, '%Y-%m-%d'); conditions.append("captured_at <= $end_date"); params['end_date'] = end_date
        except ValueError: raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")
    if conditions: query += " WHERE " + " AND ".join(conditions)
    query += " LIMIT $limit"; params['limit'] = limit
    try: result = conn.execute(query, params).fetchdf(); return JSONResponse(content=result.to_dict(orient="records"))
    except Exception as e: print(f"Error executing measurement query: {e}"); raise HTTPException(status_code=500, detail="Error querying measurements")


# CORRECTED get_stats function
@app.get("/api/stats")
async def get_stats():
    """Get general statistics about the measurement data."""
    stats = {}
    try: # Start try block on new line
        stats["total_measurements"] = conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
        value_stats = conn.execute("SELECT MIN(value), MAX(value), AVG(value) FROM measurements").fetchone()
        stats["min_value"] = value_stats[0] if value_stats else None
        stats["max_value"] = value_stats[1] if value_stats else None
        stats["avg_value"] = value_stats[2] if value_stats else None
        date_range = conn.execute("SELECT MIN(captured_at), MAX(captured_at) FROM measurements").fetchone()
        stats["first_date"] = date_range[0].isoformat() if date_range and date_range[0] else None
        stats["last_date"] = date_range[1].isoformat() if date_range and date_range[1] else None
        stats["device_count"] = conn.execute("SELECT COUNT(DISTINCT device_id) FROM measurements").fetchone()[0]
        return stats
    except Exception as e: # Correctly indented except block
        print(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error calculating statistics")


@app.get("/api/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    """Generates and serves a map tile image."""
    tile_path = f"tiles/{z}/{x}/{y}.png"
    if not os.path.exists(tile_path): print(f"Generating tile: {z}/{x}/{y}"); generated_path = generate_tile(z, x, y);
    if not os.path.exists(tile_path): raise HTTPException(status_code=500, detail="Failed to generate or find tile")
    return FileResponse(tile_path, media_type="image/png", headers={"Cache-Control": "max-age=86400"})

# UPDATED /api/import/csv endpoint to use Form(...)
@app.post("/api/import/csv")
async def import_csv(file_path: str = Form(..., description="Path to the CSV file on the server")):
    """Import data from a specific format CSV file located at file_path on the server"""
    print(f"Received request to import CSV via form: {file_path}")
    # WARNING: Accepting a file path like this is a security risk in production.
    # Consider using FastAPI's UploadFile for actual file uploads.
    if not os.path.isabs(file_path):
         base_dir = os.path.abspath(os.path.dirname(__file__))
         file_path = os.path.abspath(os.path.join(base_dir, file_path))
         print(f"Resolved relative path to: {file_path}")

    success = import_csv_data(file_path)
    if success:
        return {"success": True, "message": f"Data from {file_path} imported successfully."}
    else:
        if not os.path.exists(file_path):
             raise HTTPException(status_code=404, detail=f"File not found by server at resolved path: {file_path}")
        else:
             raise HTTPException(status_code=400, detail=f"Failed to import data from {file_path}. Check server logs for details.")

# Sample data import function (modified to add random times and unique IDs)
def import_sample_data():
    """Imports sample data if the measurements table is empty."""
    try:
        count = conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
        if count == 0:
            print("No data found, importing sample data with random times...")
            center_lat, center_lng = 35.6895, 139.6917; num_points = 500; np.random.seed(42)
            lats = center_lat + np.random.normal(0, 0.5, num_points); lngs = center_lng + np.random.normal(0, 0.5, num_points)
            fukushima_lat, fukushima_lng = 37.4216, 141.0329; values = []
            for lat, lng in zip(lats, lngs): dist = np.sqrt((lat - fukushima_lat)**2 + (lng - fukushima_lng)**2); value = np.random.gamma(shape=1.5, scale=30) + 500 * np.exp(-dist * 2); values.append(max(0, value))
            base_dates = pd.date_range(start='2023-01-01', periods=num_points, tz='UTC'); random_seconds = np.random.randint(0, 86400, num_points); timestamps = base_dates + pd.to_timedelta(random_seconds, unit='s')
            start_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM measurements").fetchone()[0]
            df = pd.DataFrame({'id': range(start_id, start_id + num_points), 'value': values, 'latitude': lats, 'longitude': lngs, 'captured_at': timestamps, 'device_id': np.random.randint(1, 10, num_points), 'unit': 'CPM'})
            conn.execute("CREATE OR REPLACE TEMP TABLE tmp_sample_data AS SELECT * FROM df")
            conn.execute("INSERT INTO measurements SELECT * FROM tmp_sample_data ON CONFLICT(id) DO NOTHING")
            conn.execute("DROP TABLE tmp_sample_data"); print(f"Imported {len(df)} sample records")
    except Exception as e: print(f"Error during sample data import: {e}")

# Run sample data import on startup (optional)
# Comment out if you only want to load manually imported data
# import_sample_data()

# *** ADDED Database Check after import attempt ***
print("--- Database Check ---")
try:
    db_count = conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
    print(f"Total records currently in measurements table: {db_count}")
    if db_count > 0:
        print("Sample row:")
        print(conn.execute("SELECT * FROM measurements LIMIT 1").fetchdf())
        print("Value range:")
        print(conn.execute("SELECT MIN(value), MAX(value) FROM measurements").fetchone())
        print("Coordinate range:")
        print(conn.execute("SELECT MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude) FROM measurements").fetchone())
except Exception as e:
    print(f"Could not perform database check: {e}")
print("--- End Database Check ---")


# Main execution block
def main():
    """Starts the FastAPI server."""
    print("--- Simplified Safecast Map Server ---")
    print(f"Database: data/safecast.duckdb"); print(f"Serving map at: http://localhost:8000"); print("Press Ctrl+C to stop the server.")
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()

