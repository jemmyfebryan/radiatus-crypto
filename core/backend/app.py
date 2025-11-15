import uvicorn
import pandas as pd
import aiofiles
import asyncio
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi_utilities import repeat_every
from typing import List, Optional
from datetime import datetime

from core.backend.utils import update_data, generate_signals
from core.logger import get_logger

logger = get_logger(__name__)

# --- Application State ---
# Global variables to store the latest signals.
# We use a dictionary as a simple in-memory "state"
app_state = {
    "long_signals": pd.DataFrame(),
    "short_signals": pd.DataFrame(),
    "last_updated": None
}
# Use a lock to prevent race conditions during the update
update_lock = asyncio.Lock()

# --- FastAPI App ---
app = FastAPI(
    title="Signal Dashboard",
    description="Backend for displaying long/short trading signals."
)

app.mount("/static", StaticFiles(directory="core/frontend"), name="static")

# --- Background Task ---
@app.on_event("startup")
@repeat_every(seconds=60 * 15, wait_first=False, raise_exceptions=True) # Runs every 15 minutes
async def run_signal_generation():
    """
    This is the background task that updates data and generates signals.
    """
    async with update_lock:
        logger.info(f"[{datetime.utcnow().isoformat()}] Starting scheduled signal generation...")
        
        # 1. Update Data (This is synchronous, but we run it in a thread pool
        #    to avoid blocking the main async loop if it's I/O bound)
        #    If your update_data() is already async, 'await' it directly.
        loop = asyncio.get_event_loop()
        # await loop.run_in_executor(None, update_data)
        
        logger.info(f"Finished updating data, now generating signals...")
        
        # 2. Generate Signals
        #    Same as above, run in executor if it's blocking.
        df_long, df_short = await loop.run_in_executor(None, generate_signals)
        df_long.fillna("None", inplace=True)
        df_short.fillna("None", inplace=True)
        
        # 3. Update global state
        app_state["long_signals"] = df_long
        app_state["short_signals"] = df_short
        app_state["last_updated"] = datetime.utcnow().isoformat()
        
        logger.info(f"[{datetime.utcnow().isoformat()}] Scheduled task complete. Signals updated.")

# --- API Endpoints ---
@app.get("/api/signals")
async def get_signals():
    """
    API endpoint to fetch the latest signals.
    """
    # async with update_lock:
    return {
        "last_updated": app_state["last_updated"],
        "long": app_state["long_signals"].to_dict("records"),
        "short": app_state["short_signals"].to_dict("records"),
    }
  
@app.get("/", response_class=FileResponse)
async def frontend():
    return FileResponse("core/frontend/index.html")
  
# @app.get("/", response_class=HTMLResponse)
# async def get_dashboard(request: Request):
#     """
#     Serves the main HTML dashboard.
#     """
#     # We embed the HTML, CSS (Bootstrap), and JS directly into the response.
#     # This keeps the app as a single file.
#     return HTMLResponse(content=f"""
#     <!doctype html>
#     <html lang="en" data-bs-theme="dark">
#       <head>
#         <meta charset="utf-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1">
#         <title>Radiatus Crypto - Signals</title>
#         <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
#         <style>
#           body {{ font-size: 0.875rem; }}
#           .table {{ vertical-align: middle; }}
#           .metric-bar {{
#             background-color: #333;
#             border-radius: 4px;
#             overflow: hidden;
#             width: 100px;
#             height: 10px;
#           }}
#           .metric-bar-inner {{
#             height: 100%;
#             background-color: #0d6efd;
#             transition: width 0.3s ease-in-out;
#           }}
#           .metric-bar-inner.high {{ background-color: #198754; }} /* green */
#           .metric-bar-inner.medium {{ background-color: #ffc107; }} /* yellow */
#           .metric-bar-inner.low {{ background-color: #dc3545; }} /* red */
#           .bd-true {{ color: #198754; font-weight: bold; }}
#           .bd-false {{ color: #6c757d; }}
#           .loading-spinner {{
#             display: none; 
#             position: fixed; 
#             top: 50%; 
#             left: 50%; 
#             transform: translate(-50%, -50%); 
#             z-index: 1000;
#           }}
#           .table-responsive {{ max-height: 70vh; }}
          
#           /* --- Sticky Header --- */
#           .table-responsive thead th {{
#             position: sticky;
#             top: 0;
#             z-index: 1;
#             background-color: #212529; /* Match table-dark background */
#             box-shadow: inset 0 -2px 0 #32383e; /* Optional border */
#           }}
#         </style>
#       </head>
#       <body>
#         <!-- Header Navbar -->
#         <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
#           <div class="container-fluid">
#             <a class="navbar-brand" href="#"><b> Radiatus Crypto </b></a>
#             <span class="navbar-text" id="last-updated">
#               Loading...
#             </span>
#           </div>
#         </nav>

#         <!-- Main Content -->
#         <div class="container-fluid mt-4">
          
#           <!-- Page Subtitle -->
#           <p class="lead text-muted" style="margin-top: -10px; margin-bottom: 20px;">Automated crypto signals generator</p>

#           <div class="row">
            
#             <!-- Long Signals Table (Full Width) -->
#             <div class="col-12 mb-4">
#               <h3 class="text-success">Long Signals</h3>
#               <div class="table-responsive" id="long-signals-container">
#                 <p>Loading signals...</p>
#               </div>
#             </div>
            
#             <!-- Short Signals Table (Full Width) -->
#             <div class="col-12">
#               <h3 class="text-danger">Short Signals</h3>
#               <div class="table-responsive" id="short-signals-container">
#                 <p>Loading signals...</p>
#               </div>
#             </div>
            
#           </div>
#         </div>
        
#         <!-- Footer -->
#         <footer class="text-center text-muted p-4 mt-5">
#           © 2025 Radiatus Crypto. All rights reserved.
#         </footer>
        
#         <!-- Loading Spinner -->
#         <div class="spinner-border text-primary loading-spinner" id="loading" role="status">
#           <span class="visually-hidden">Loading...</span>
#         </div>

#         <script>
#           // Columns to display and their headers
#           const COLUMNS = [
#             {{ key: 'asset', header: 'Asset' }},
#             {{ key: 'price', header: 'Price' }},
#             {{ key: 'ob_metric', header: 'OB Metric' }},
#             {{ key: 'divergence_metric', header: 'Divergence Metric' }},
#             {{ key: 'iob_top', header: 'OB Top' }},
#             {{ key: 'iob_btm', header: 'OB Bottom' }},
#             {{ key: 'iob_left_utc', header: 'OB Start (UTC)' }},
#             {{ key: 'mfi_bd', header: 'MFI BD' }},
#             {{ key: 'stoch_bd', header: 'Stoch BD' }},
#             {{ key: 'rsi_bd', header: 'RSI BD' }},
#             {{ key: 'williams_r_bd', header: 'W%R BD' }},
#           ];
          
#           function renderBoolArray(arr) {{
#             if (!Array.isArray(arr)) return '<span>-</span>';
#             return arr.map(b => 
#               `<span class="bd-${{b}}">${{b ? 'T' : 'F'}}</span>`
#             ).join(' ');
#           }}

#           function getMetricColorClass(metric) {{
#             if (metric >= 0.7) return 'high';
#             if (metric >= 0.4) return 'medium';
#             return 'low';
#           }}

#           function renderMetric(metric) {{
#             const rounded = (metric * 100).toFixed(1);
#             const colorClass = getMetricColorClass(metric);
#             return `
#               <div class="d-flex align-items-center">
#                 <span class="me-2" style="width: 40px;">${{rounded}}%</span>
#                 <div class="metric-bar">
#                   <div class="metric-bar-inner ${{colorClass}}" style="width: ${{rounded}}%;"></div>
#                 </div>
#               </div>
#             `;
#           }}

#           function renderTable(containerId, data) {{
#             const container = document.getElementById(containerId);
#             if (data.length === 0) {{
#               container.innerHTML = '<p class="text-muted">No signals found.</p>';
#               return;
#             }}

#             let table = '<table class="table table-dark table-striped table-hover">';
            
#             // Header
#             table += '<thead><tr>';
#             for (const col of COLUMNS) {{
#               table += `<th>${{col.header}}</th>`;
#             }}
#             table += '</tr></thead>';

#             // Body
#             table += '<tbody>';
#             for (const row of data) {{
#               table += '<tr>';
#               for (const col of COLUMNS) {{
#                 let cellData = row[col.key];
                
#                 // Custom renderers
#                 if (col.key === 'ob_metric' || col.key === 'divergence_metric') {{
#                   cellData = renderMetric(cellData);
#                 }} else if (col.key.endsWith('_bd')) {{
#                   cellData = renderBoolArray(cellData);
#                 }} else if (typeof cellData === 'number' && !['price', 'iob_top', 'iob_btm'].includes(col.key)) {{
#                   // --- MODIFIED: No longer rounds iob_top and iob_btm ---
#                   cellData = cellData.toFixed(2);
#                 }} else if (col.key === 'iob_left_utc') {{
#                   cellData = new Date(cellData).toLocaleString('sv-SE', {{ timeZone: 'UTC' }}).replace('T', ' ');
#                 }}
                
#                 table += `<td>${{cellData}}</td>`;
#               }}
#               table += '</tr>';
#             }}
#             table += '</tbody></table>';
            
#             container.innerHTML = table;
#           }}

#           async function fetchData() {{
#             const loadingSpinner = document.getElementById('loading');
#             loadingSpinner.style.display = 'block';
            
#             try {{
#               const response = await fetch('/api/signals');
#               if (!response.ok) {{
#                 throw new Error(`HTTP error! status: ${{response.status}}`);
#               }}
#               const data = await response.json();
              
#               // Render tables
#               renderTable('long-signals-container', data.long);
#               renderTable('short-signals-container', data.short);
              
#               // Update timestamp
#               const lastUpdatedEl = document.getElementById('last-updated');
#               if (data.last_updated) {{
#                 const localTime = new Date(data.last_updated + 'Z').toLocaleString();
#                 lastUpdatedEl.textContent = `Last Updated: ${{localTime}}`;
#               }} else {{
#                 lastUpdatedEl.textContent = 'Last Updated: Never';
#               }}
              
#             }} catch (error) {{
#               console.error('Failed to fetch signals:', error);
#               document.getElementById('long-signals-container').innerHTML = '<p class="text-danger">Failed to load signals.</p>';
#               document.getElementById('short-signals-container').innerHTML = '<p class="text-danger">Failed to load signals.</p>';
#             }} finally {{
#               loadingSpinner.style.display = 'none';
#             }}
#           }}

#           // Fetch data on page load
#           document.addEventListener('DOMContentLoaded', fetchData);
          
#           // Optionally, refresh data every few minutes
#           // setInterval(fetchData, 60 * 1000); // every 1 minute
#         </script>
#       </body>
#     </html>
#     """)
    
# --- Run the App ---
if __name__ == "__main__":
    # This block is for direct execution (e.g., `python ./core/backend/app.py`)
    # PM2 will use the `app` object directly.
    print("Starting Uvicorn server for local development...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, app_dir=".")

# update_data()

# df_long_results, df_short_results = generate_signals()