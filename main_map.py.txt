import os
import json
import pandas as pd
import numpy as np
import duckdb
from fastapi import FastAPI, HTTPException, Query, Request
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

# Create necessary directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("tiles", exist_ok=True)
# os.makedirs("static", exist_ok=True) # Not strictly needed anymore unless serving other static files
# os.makedirs("static/js", exist_ok=True)
# os.makedirs("static/css", exist_ok=True)
# os.makedirs("cache", exist_ok=True) # Cache directory not used in current logic

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
default_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Simplified Safecast Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css" />
    <style>
        /* Base Styles & User-Provided CSS Rules with !important */
        body { margin: 0; padding: 0; font-family: Arial, sans-serif; }
        #map { width: 100%; height: 100vh; }

        /* Info box and Legend base */
        .info {
            padding: 6px 8px !important;
            font: 14px/16px Arial, Helvetica, sans-serif !important;
            box-shadow: 0 0 15px rgba(0,0,0,0.2) !important;
            border-radius: 5px !important;
            color: white !important;
            background-color: black !important; /* User specified black */
         }

        /* Heading inside info box */
        .info h4 {
            margin: 0 0 5px !important;
            color: #fff !important; /* User specified white */
        }

        /* Legend specific styles */
        .legend {
            line-height: 1.5 !important; /* Control spacing between lines */
            color: white !important; /* User specified white */
         }
        .legend i { /* Style for the color square */
            display: inline-block;      /* Display as inline block */
            vertical-align: middle;   /* Align vertically with text */
            width: 18px !important;
            height: 18px !important;
            margin-right: 8px !important; /* Keep margin */
            opacity: 0.8 !important;
            border: 1px solid #555 !important;
        }

        /* Tooltip */
        .leaflet-tooltip-hover {
             background-color: rgba(0, 0, 0, 0.85) !important;
             color: white !important; border: none !important; box-shadow: none !important; padding: 5px !important; font-size: 12px !important;
        }

        /* Zoom buttons (using user selector .leaflet-bar a) */
        .leaflet-bar a, .leaflet-bar a:hover {
            background-color: #0f0f0f !important; /* User specified dark background */
            border-bottom: 1px solid #ccc !important; /* User specified border */
            width: 26px !important;
            height: 26px !important;
            line-height: 26px !important;
            display: block !important;
            text-align: center !important;
            text-decoration: none !important;
            color: #fff !important; /* User specified white text */
        }
         /* Adjust border-radius for zoom buttons */
         .leaflet-bar a:first-child { border-top-left-radius: 4px !important; border-top-right-radius: 4px !important; }
         .leaflet-bar a:last-child { border-bottom-left-radius: 4px !important; border-bottom-right-radius: 4px !important; border-bottom: none !important; }

        /* Attribution */
        .leaflet-control-attribution {
            background: rgba(15, 15, 15, 0.8) !important; /* Match user's dark tone */
            color: #ccc !important;
        }
        .leaflet-control-attribution a {
            color: #aaa !important;
        }

         /* Scale line */
         .leaflet-control-scale-line {
            border: 2px solid #555 !important; border-top: none !important;
            color: #eee !important;
            background: rgba(15, 15, 15, 0.7) !important; /* Match user's dark tone */
            padding: 2px 5px !important; box-shadow: none !important;
            text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000 !important;
         }

        /* Cursor style for hover */
        #map.leaflet-container.pointer-cursor { cursor: pointer !important; }

        /* Pinned Popup Style */
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
        const map = L.map('map').setView([37.5, 140.0], 6); // Centered on Japan
        const mapContainer = map.getContainer(); // Get map container element for cursor styling

        // --- Base Map Layer ---
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }).addTo(map);

        // --- Heatmap Tile Layer ---
        const safecastLayer = L.tileLayer('/api/tiles/{z}/{x}/{y}.png', {
            maxZoom: 18, // Max zoom for which tiles are generated/requested
            tms: false,  // Standard XYZ tiling scheme
            opacity: 0.65, // Adjust transparency of the heatmap
            attribution: 'Data: <a href="https://safecast.org">Safecast</a>'
        }).addTo(map);

        // --- State Variables for Interactivity ---
        let currentPoints = [];      // Stores measurement points currently fetched for the view
        let hoverTooltip = null;     // Stores the temporary hover tooltip instance
        let mouseLatLng = null;      // Stores the last known mouse coordinates
        let hoveredPointData = null; // Stores the data object of the currently hovered point
        let pinnedPopup = null;      // Stores the permanent pinned popup instance

        // --- Helper Functions ---

        /**
         * Fetches measurement points for the current map view bounds from the API.
         */
        async function fetchPointsForView() {
            const bounds = map.getBounds();
            // Construct API URL with current map bounds and a limit
            const url = `/api/measurements?min_lat=${bounds.getSouth()}&max_lat=${bounds.getNorth()}&min_lng=${bounds.getWest()}&max_lng=${bounds.getEast()}&limit=1000`; // Limit points for performance
            try {
                const response = await fetch(url);
                if (!response.ok) { throw new Error(`HTTP error! status: ${response.status}`); }
                currentPoints = await response.json(); // Store fetched points
                // console.log(`Fetched ${currentPoints.length} points for hover.`);
            } catch (error) {
                console.error("Error fetching points for hover:", error);
                currentPoints = []; // Clear points on error
            }
        }

        /**
         * Finds the closest measurement point to the given LatLng within a threshold.
         * @param {L.LatLng} latlng - The latitude and longitude of the mouse cursor.
         * @returns {object|null} The data object of the closest point, or null if none found.
         */
        function findClosestPoint(latlng) {
            let closestPoint = null;
            let minDistanceSq = Infinity;
            // Dynamic threshold: smaller distance required at higher zoom levels
            const baseThreshold = 0.0005; // Base sensitivity (degrees squared)
            const thresholdDistanceSq = baseThreshold * Math.pow(0.85, map.getZoom()); // Reduces threshold as zoom increases

            currentPoints.forEach(point => {
                if (point.latitude && point.longitude) {
                    // Calculate squared distance (faster than sqrt)
                    const dx = latlng.lng - point.longitude;
                    const dy = latlng.lat - point.latitude;
                    const distanceSq = dx * dx + dy * dy;
                    // Check if within threshold and closer than previous minimum
                    if (distanceSq < thresholdDistanceSq) {
                         if (distanceSq < minDistanceSq) {
                            minDistanceSq = distanceSq;
                            closestPoint = point;
                         }
                    }
                }
            });
            return closestPoint;
        }

        /**
         * Hides the temporary hover tooltip if it exists.
         */
        function hideHoverTooltip() {
             if (hoverTooltip && map.hasLayer(hoverTooltip)) {
                 map.removeLayer(hoverTooltip);
                 hoverTooltip = null;
             }
        }

        /**
         * Updates the hover tooltip based on the mouse position.
         * @param {L.LatLng} latlng - The current mouse coordinates.
         */
        function updateTooltip(latlng) {
            if (!latlng) return; // Exit if no coordinates

            const closestPoint = findClosestPoint(latlng); // Find nearby point

            if (closestPoint) {
                // Point found: change cursor and store data
                mapContainer.classList.add('pointer-cursor');
                hoveredPointData = closestPoint;

                // Don't show hover tooltip if a popup for this exact point is already pinned
                if (pinnedPopup && pinnedPopup.options.customId === closestPoint.id) {
                     hideHoverTooltip(); // Hide hover tooltip
                     return; // Exit without showing hover tooltip
                }

                // Prepare tooltip content (can be simpler than popup)
                const content = `<i>Hover - Click to Pin</i><br><b>Value:</b> ${closestPoint.value.toFixed(2)} ...`;

                // Create or update the hover tooltip
                if (!hoverTooltip) { // Create if doesn't exist
                    hoverTooltip = L.tooltip({
                        permanent: false, // Show only on hover
                        sticky: true,     // Follow the mouse
                        direction: 'top', // Position above cursor
                        offset: L.point(0, -10), // Offset slightly
                        className: 'leaflet-tooltip-hover' // Apply custom style
                    })
                    .setLatLng(latlng)
                    .setContent(content)
                    .addTo(map);
                } else { // Update existing tooltip
                    hoverTooltip.setLatLng(latlng).setContent(content);
                    if (!map.hasLayer(hoverTooltip)) { // Add back if removed somehow
                        hoverTooltip.addTo(map);
                    }
                }
            } else {
                // No close point found: reset cursor and clear hover data
                mapContainer.classList.remove('pointer-cursor');
                hoveredPointData = null;
                hideHoverTooltip(); // Hide hover tooltip
            }
        }

        /**
         * Debounce function to limit the rate at which a function can fire.
         * @param {Function} func - The function to debounce.
         * @param {number} wait - The debounce interval in milliseconds.
         * @returns {Function} The debounced function.
         */
        function debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        }

        // --- Event Listeners ---

        // Fetch points when map view changes (pan/zoom end)
        map.on('load zoomend moveend', debounce(fetchPointsForView, 500)); // Debounced

        // Update tooltip on mouse move
        map.on('mousemove', debounce((e) => {
             mouseLatLng = e.latlng; // Store current mouse position
             updateTooltip(mouseLatLng); // Update tooltip/cursor
        }, 50)); // Debounced frequently

        // Handle mouse leaving the map container
        map.on('mouseout', () => {
             mapContainer.classList.remove('pointer-cursor'); // Reset cursor
             hideHoverTooltip(); // Hide hover tooltip
             hoveredPointData = null; // Clear hover data
             mouseLatLng = null;
        });

        // Handle clicks on the map for pinning popups
        map.on('click', function(e) {
            // Close any existing pinned popup first
            if (pinnedPopup) {
                map.removeLayer(pinnedPopup);
                pinnedPopup = null;
            }

            // If the click occurred while hovering over a valid point
            if (hoveredPointData) {
                const point = hoveredPointData; // Use the stored data

                // Convert epoch seconds to milliseconds for Date object
                const dateObject = new Date(point.captured_at * 1000);
                // Format date as UTC ISO string, handle invalid dates
                const dateString = !isNaN(dateObject) ? dateObject.toISOString() : "Invalid Date";

                // Prepare popup content
                const popupContent = `
                    <b>Value:</b> ${point.value.toFixed(2)} ${point.unit || 'CPM'}<br>
                    <b>Date:</b> ${dateString}<br>
                    <b>Coords:</b> ${point.latitude.toFixed(4)}, ${point.longitude.toFixed(4)}
                    ${point.device_id ? `<br><b>Device:</b> ${point.device_id}` : ''}
                `;

                // Create and open the persistent popup
                pinnedPopup = L.popup({
                        closeButton: true,   // Show close button
                        autoClose: false,    // Don't close on map click
                        closeOnClick: false, // Don't close on map click
                        keepInView: true,    // Try to keep popup visible
                        customId: point.id   // Store point ID to prevent hover tooltip overlap
                    })
                    .setLatLng([point.latitude, point.longitude]) // Position at the actual point
                    .setContent(popupContent)
                    .openOn(map); // Open the popup on the map

                // Clean up hover state
                hideHoverTooltip();
                hoveredPointData = null;
            }
        });

        // --- Initial Data Fetch ---
        fetchPointsForView();

        // --- UI Controls ---

        // Info Box Control
        const info = L.control({position: 'topright'}); // Position top right
        info.onAdd = function (map) {
            this._div = L.DomUtil.create('div', 'info'); // Create a div with class 'info'
            this.update();
            return this._div;
        };
        info.update = function (props) { // Method that we will use to update the control based on map state
            this._div.innerHTML = '<h4>Simplified Safecast Map</h4><p>Hover over data areas for details</p>';
        };
        info.addTo(map);

        // Legend Control
        const legend = L.control({position: 'bottomright'}); // Position bottom right
        const legendColors = { green: '#009900', yellow: '#E6E600', red: '#E60000' }; // Colors matching backend
        // Function to map CPM value ranges to specific colors for the legend display
        function getLegendColor(cpmValue) {
            if (cpmValue <= 100) { return legendColors.green; }
            else if (cpmValue <= 500) { return legendColors.yellow; }
            else { return legendColors.red; }
        }
        legend.onAdd = function (map) {
            const div = L.DomUtil.create('div', 'info legend leaflet-control'); // Use 'info' class for styling
            const grades = [0, 100, 500, 1000]; // CPM thresholds for legend ranges

            div.innerHTML += '<b>Radiation (CPM)</b><br>'; // Legend title
            // Loop through grades and generate legend items
            for (let i = 0; i < grades.length; i++) {
                 const gradeValue = grades[i];
                 const color = getLegendColor(gradeValue); // Get color for this range
                 const label = gradeValue + (grades[i + 1] ? '&ndash;' + grades[i + 1] : '+'); // Format label text

                 // Add color square (i tag) and label text, followed by line break
                 div.innerHTML +=
                    '<i style="background:' + color + ';"></i> ' + // Color square
                    label + '<br>'; // Label text and line break
            }
            return div;
        };
        legend.addTo(map);

        // Scale Control
        L.control.scale({imperial: false, position: 'bottomleft'}).addTo(map); // Metric scale, bottom left

        // --- End UI Controls ---

    </script>
</body>
</html>
"""

# Initialize DuckDB connection
conn = duckdb.connect('data/safecast.duckdb', read_only=False) # Ensure read_only is False if writing

# Initialize database tables
def init_database():
    """Creates database tables if they don't exist."""
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
    # Removed spatial_index and devices tables as they weren't used in API/logic
    # conn.execute("""
    #     CREATE TABLE IF NOT EXISTS spatial_index ( ... )
    # """)
    # conn.execute("""
    #     CREATE TABLE IF NOT EXISTS devices ( ... )
    # """)
    print("Database tables checked/created.")

init_database() # Ensure tables exist

# Helper functions for tile calculations
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
    nw_lat, nw_lng = num2deg(x, y, z) # Northwest corner
    se_lat, se_lng = num2deg(x + 1, y + 1, z) # Southeast corner (tile below and right)
    return {
        'min_lat': se_lat, # Minimum latitude (bottom edge)
        'max_lat': nw_lat, # Maximum latitude (top edge)
        'min_lng': nw_lng, # Minimum longitude (left edge)
        'max_lng': se_lng  # Maximum longitude (right edge)
    }

# Function to generate a heatmap tile using scatter plot
import matplotlib.colors as mcolors # Ensure import
def generate_tile(z, x, y, width=256, height=256):
    """Generates a heatmap tile image for the given Z/X/Y coordinates."""
    tile_path = f"tiles/{z}/{x}/{y}.png"
    tile_dir = os.path.dirname(tile_path)
    os.makedirs(tile_dir, exist_ok=True) # Ensure directory exists

    # Get the geographical bounds for this tile
    bounds = get_tile_bounds(z, x, y)

    # --- Calculate bounds with overlap for smoother edges ---
    lat_range = bounds['max_lat'] - bounds['min_lat']
    lng_range = bounds['max_lng'] - bounds['min_lng']
    # Add a buffer (e.g., 25% of tile dimension)
    buffer_factor = 0.25
    buffered_bounds = {
        'min_lat': bounds['min_lat'] - lat_range * buffer_factor,
        'max_lat': bounds['max_lat'] + lat_range * buffer_factor,
        'min_lng': bounds['min_lng'] - lng_range * buffer_factor,
        'max_lng': bounds['max_lng'] + lng_range * buffer_factor
    }
    # --- End Calculate bounds with overlap ---

    # Fetch measurements within the *buffered* geographical bounds
    query = f"""
        SELECT latitude, longitude, value
        FROM measurements
        WHERE latitude >= {buffered_bounds['min_lat']}
          AND latitude <= {buffered_bounds['max_lat']}
          AND longitude >= {buffered_bounds['min_lng']}
          AND longitude <= {buffered_bounds['max_lng']}
    """
    try:
        result = conn.execute(query).fetchdf()
    except Exception as e:
        print(f"Error querying data for tile {z}/{x}/{y}: {e}")
        result = pd.DataFrame() # Empty dataframe on error

    # If no data points found in the buffered area, create a transparent tile
    if result.empty:
        img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        img.save(tile_path)
        return tile_path

    # --- Generate Plot ---
    dpi = 100 # Dots per inch for the figure
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    ax = fig.add_subplot(111)

    # Calculate pixel coordinates RELATIVE TO THE ORIGINAL (non-buffered) BOUNDS
    # This makes points from the buffer zone fall outside the 0-width/0-height range initially
    x_pixels = (result['longitude'] - bounds['min_lng']) / lng_range * width
    y_pixels = (1 - (result['latitude'] - bounds['min_lat']) / lat_range) * height # Y is inverted for image coords

    # Use logarithmic scaling for radiation values (log(1+x))
    values = np.log1p(result['value'].clip(lower=0)) # Clip ensures non-negative before log

    # Define the green-yellow-red colormap
    cmap_colors = [(0, 0.6, 0), (0.9, 0.9, 0), (0.9, 0, 0)] # Green, Yellow, Red RGB tuples
    cmap = LinearSegmentedColormap.from_list('safecast_gyr', cmap_colors)

    # Define normalization range based on LOG values (log1p scale)
    # Adjust vmax based on expected max CPM, e.g., log1p(5000) ~= 8.5
    log_vmin = 0.0  # Corresponds to 0 CPM
    log_vmax = 7.0  # Corresponds to approx 1100 CPM. Adjust as needed.
    norm = mcolors.Normalize(vmin=log_vmin, vmax=log_vmax)

    # Parameters for scatter plot (marker size and alpha for blending)
    marker_size = 300 # Size in points^2, controls overlap. Adjust based on zoom/density.
    alpha_value = 0.15 # Low alpha for additive blending effect. Adjust for desired intensity.

    # Plot points using scatter with specified colormap, normalization, size, and alpha
    scatter = ax.scatter(
        x_pixels, y_pixels,
        c=values,        # Color based on log values
        s=marker_size,   # Marker size
        cmap=cmap,       # Apply the G-Y-R colormap
        alpha=alpha_value,# Apply transparency
        norm=norm,       # Apply the normalization
        marker='o',      # Circle markers
        linewidths=0     # No marker outlines
    )

    # Configure plot appearance
    ax.set_xlim(0, width)   # Set X limits strictly to tile width
    ax.set_ylim(height, 0)  # Set Y limits strictly to tile height (inverted)
    ax.axis('off')          # Turn off axes and labels
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # Remove padding around plot

    # Save plot to a memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True, dpi=dpi) # Save as transparent PNG
    plt.close(fig) # Close the figure to free memory
    buf.seek(0) # Rewind buffer

    # Create PIL image from buffer and save
    try:
        img = Image.open(buf)
        img.save(tile_path)
    except Exception as e:
        print(f"Error saving tile image {tile_path}: {e}")
        # Create an empty file to avoid future 404s if saving fails
        Path(tile_path).touch()
        return None # Indicate failure

    return tile_path

# --- API Routes ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main HTML page from the default_html variable"""
    return HTMLResponse(content=default_html)

@app.get("/api/measurements")
async def get_measurements(
    min_lat: Optional[float] = Query(None, description="Minimum latitude"),
    max_lat: Optional[float] = Query(None, description="Maximum latitude"),
    min_lng: Optional[float] = Query(None, description="Minimum longitude"),
    max_lng: Optional[float] = Query(None, description="Maximum longitude"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, description="Maximum number of records to return")
):
    """Get measurements with optional filtering by bounds, date, and limit."""
    query = "SELECT id, value, latitude, longitude, epoch(captured_at) as captured_at, device_id, unit FROM measurements" # Select epoch seconds
    conditions = []
    params = {}

    # Build query conditions safely using parameters
    if min_lat is not None:
        conditions.append("latitude >= $min_lat")
        params['min_lat'] = min_lat
    if max_lat is not None:
        conditions.append("latitude <= $max_lat")
        params['max_lat'] = max_lat
    if min_lng is not None:
        conditions.append("longitude >= $min_lng")
        params['min_lng'] = min_lng
    if max_lng is not None:
        conditions.append("longitude <= $max_lng")
        params['max_lng'] = max_lng
    # Note: Date filtering assumes 'YYYY-MM-DD' string input
    if start_date is not None:
        try:
            datetime.datetime.strptime(start_date, '%Y-%m-%d') # Validate format
            conditions.append("captured_at >= $start_date")
            params['start_date'] = start_date
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD.")
    if end_date is not None:
        try:
            datetime.datetime.strptime(end_date, '%Y-%m-%d') # Validate format
            conditions.append("captured_at <= $end_date")
            params['end_date'] = end_date
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " LIMIT $limit"
    params['limit'] = limit

    try:
        # Execute query with parameters
        result = conn.execute(query, params).fetchdf()
        # Convert dataframe to list of dictionaries for JSON response
        return JSONResponse(content=result.to_dict(orient="records"))
    except Exception as e:
        print(f"Error executing measurement query: {e}")
        raise HTTPException(status_code=500, detail="Error querying measurements")


@app.get("/api/stats")
async def get_stats():
    """Get general statistics about the measurement data."""
    stats = {}
    try:
        stats["total_measurements"] = conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]

        value_stats = conn.execute("SELECT MIN(value), MAX(value), AVG(value) FROM measurements").fetchone()
        stats["min_value"] = value_stats[0]
        stats["max_value"] = value_stats[1]
        stats["avg_value"] = value_stats[2]

        date_range = conn.execute("SELECT MIN(captured_at), MAX(captured_at) FROM measurements").fetchone()
        # Format dates nicely, handle None if table is empty
        stats["first_date"] = date_range[0].isoformat() if date_range and date_range[0] else None
        stats["last_date"] = date_range[1].isoformat() if date_range and date_range[1] else None

        stats["device_count"] = conn.execute("SELECT COUNT(DISTINCT device_id) FROM measurements").fetchone()[0]

        return stats
    except Exception as e:
        print(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Error calculating statistics")

@app.get("/api/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    """Generates and serves a map tile image."""
    tile_path = f"tiles/{z}/{x}/{y}.png"

    # Simple caching: Check if file exists and is reasonably recent (e.g., < 1 day old)
    # More advanced caching could use modification times or a dedicated cache store.
    # For now, just generate if it doesn't exist.
    if not os.path.exists(tile_path):
        print(f"Generating tile: {z}/{x}/{y}")
        generated_path = generate_tile(z, x, y)
        if generated_path is None:
             raise HTTPException(status_code=500, detail="Failed to generate tile")
    # else:
    #     print(f"Serving cached tile: {z}/{x}/{y}")

    # Return the tile file
    return FileResponse(tile_path, media_type="image/png", headers={"Cache-Control": "max-age=86400"}) # Cache for 1 day


# Data Import API Routes (Example implementations)
# Note: These expect a file path on the server. For uploads, use FastAPI's UploadFile.
# @app.post("/api/import/csv")
# async def import_csv(file_path: str):
#     """Import data from a CSV file located at file_path on the server"""
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail=f"CSV file not found: {file_path}")
#     # Add logic here to read the CSV and insert into DuckDB
#     # Example: df = pd.read_csv(file_path); conn.register('df_view', df); conn.execute("INSERT INTO measurements SELECT * FROM df_view")
#     return {"message": f"CSV import initiated for {file_path} (implementation pending)"}

# @app.post("/api/import/json")
# async def import_json(file_path: str):
#     """Import data from a JSON file located at file_path on the server"""
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail=f"JSON file not found: {file_path}")
#     # Add logic here to read the JSON and insert into DuckDB
#     return {"message": f"JSON import initiated for {file_path} (implementation pending)"}


# Sample data import function - MODIFIED TO ADD RANDOM TIMES
def import_sample_data():
    """Imports sample data if the measurements table is empty."""
    try:
        count = conn.execute("SELECT COUNT(*) FROM measurements").fetchone()[0]
        if count == 0:
            print("No data found, importing sample data with random times...")

            center_lat, center_lng = 35.6895, 139.6917  # Tokyo
            num_points = 500
            np.random.seed(42) # For reproducibility

            lats = center_lat + np.random.normal(0, 0.5, num_points)
            lngs = center_lng + np.random.normal(0, 0.5, num_points)

            # Generate radiation values - higher near Fukushima
            fukushima_lat, fukushima_lng = 37.4216, 141.0329
            values = []
            for lat, lng in zip(lats, lngs):
                dist = np.sqrt((lat - fukushima_lat)**2 + (lng - fukushima_lng)**2)
                value = np.random.gamma(shape=1.5, scale=30) + 500 * np.exp(-dist * 2)
                values.append(max(0, value)) # Ensure value is not negative

            # Create base dates (at midnight UTC)
            base_dates = pd.date_range(start='2023-01-01', periods=num_points, tz='UTC')
            # Create random time offsets (in seconds for a full day)
            random_seconds = np.random.randint(0, 86400, num_points)
            # Add random time offsets to base dates
            timestamps = base_dates + pd.to_timedelta(random_seconds, unit='s')

            df = pd.DataFrame({
                'id': range(1, num_points + 1),
                'value': values,
                'latitude': lats,
                'longitude': lngs,
                'captured_at': timestamps, # Use timestamps with time components
                'device_id': np.random.randint(1, 10, num_points),
                'unit': 'CPM'
            })

            # Import to database using parameterized query for safety
            conn.execute("CREATE OR REPLACE TEMP TABLE tmp_sample_data AS SELECT * FROM df")
            conn.execute("INSERT INTO measurements SELECT * FROM tmp_sample_data")
            conn.execute("DROP TABLE tmp_sample_data") # Clean up temp table
            print(f"Imported {len(df)} sample records")

    except Exception as e:
        print(f"Error during sample data import: {e}")

# Run sample data import on startup
import_sample_data()

# Main execution block
def main():
    """Starts the FastAPI server."""
    print("--- Simplified Safecast Map Server ---")
    print(f"Database: data/safecast.duckdb")
    print(f"Serving map at: http://localhost:8000")
    print("Press Ctrl+C to stop the server.")
    # Run Uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    # This block executes when the script is run directly
    main()
