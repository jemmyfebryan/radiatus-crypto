import sqlite3
import pandas as pd
import json
import os
import time
from tqdm.asyncio import tqdm # <-- ADD THIS
# import requests         <-- REMOVE THIS
import httpx              # <-- ADD THIS
import asyncio            # <-- ADD THIS
from datetime import datetime, timedelta
import argparse
import sys

from core.db.config import DB_FILE


# ========== DATABASE INIT ==========

def init_db():
    """Initialize SQLite database with required schema."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS asset_prices (
            timestamp INTEGER NOT NULL,
            asset TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            PRIMARY KEY (timestamp, asset)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tracked_assets (
            asset TEXT PRIMARY KEY
        )
    """)

    conn.commit()
    conn.close()


# ========== DB UTILITIES ==========

def get_latest_timestamp(asset_code):
    """Return the latest timestamp for a given asset."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(timestamp) FROM asset_prices WHERE asset = ?", (asset_code.upper(),))
    result = cursor.fetchone()[0]
    conn.close()
    return result


def insert_data(df):
    if df.empty:
        return 0

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for _, row in df.iterrows():
        # Ensure all values are Python-native, not NumPy dtypes
        data = (
            str(row["date"]),
            str(row["asset"]),
            float(row["open"]) if pd.notna(row["open"]) else None,
            float(row["high"]) if pd.notna(row["high"]) else None,
            float(row["low"]) if pd.notna(row["low"]) else None,
            float(row["close"]) if pd.notna(row["close"]) else None,
            int(row["volume"]) if pd.notna(row["volume"]) else None,
        )
        cursor.execute("""
            INSERT OR IGNORE INTO asset_prices
            (date, asset, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data)


    conn.commit()
    rows = conn.total_changes
    conn.close()
    return rows

def upsert_data_replace_overlap(df, asset_code, overlap_from_ts=None):
    """
    Insert dataframe into DB, but first remove any existing rows for the asset
    with timestamp >= overlap_from_ts (if provided). Returns number of rows inserted.
    """
    if df.empty:
        return 0

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    if overlap_from_ts:
        cursor.execute(
            "DELETE FROM asset_prices WHERE asset = ? AND timestamp >= ?",
            (asset_code.upper(), int(overlap_from_ts))
        )

    # 🔧 Convert all numpy dtypes to native Python types
    df = df.copy()
    df["timestamp"] = df["timestamp"].astype(int)
    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["asset"] = df["asset"].astype(str)

    rows_to_insert = [
        (
            int(row.timestamp),
            row.asset,
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        )
        for _, row in df.iterrows()
    ]

    cursor.executemany("""
        INSERT OR IGNORE INTO asset_prices
        (timestamp, asset, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, rows_to_insert)

    conn.commit()
    inserted = conn.total_changes
    conn.close()
    return inserted




# ========== ASSET MANAGEMENT ==========

def add_asset(asset_code):
    asset = asset_code.upper()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO tracked_assets (asset) VALUES (?)", (asset,))
    conn.commit()
    conn.close()
    print(f"✅ Added asset: {asset}")


def delete_asset(asset_code):
    asset = asset_code.upper()
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM tracked_assets WHERE asset = ?", (asset,))
    conn.commit()
    conn.close()
    print(f"🗑️ Deleted asset: {asset}")


def list_assets():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT asset FROM tracked_assets ORDER BY asset", conn)
    conn.close()

    if df.empty:
        print("⚠️ No tracked assets.")
    else:
        print("📈 Tracked assets:")
        for asset in df["asset"]:
            print(f" - {asset}")


# ========== FETCH & UPDATE DATA ==========
# ========== FETCH & UPDATE DATA ==========

async def fetch_crypto_data(asset_code, start_date=None, max_hours=None):
    """
    Fetch historical crypto price data (hourly) from CoinDesk's data API.
    
    Args:
        asset_code (str): The crypto instrument code, e.g. 'BTC-USD'.
        start_date (str | None): Optional start date (YYYY-MM-DD). If given, fetch from this date forward.
        max_hours (int | None): Optional cap on number of hours to fetch (for incremental updates).
                                Example: max_hours=48 fetches only last 2 days of data.
    
    Returns:
        pd.DataFrame: ['timestamp', 'asset', 'open', 'high', 'low', 'close', 'volume']
    """
    BASE_URL = 'https://data-api.coindesk.com/index/cc/v1/historical/hours'
    MARKET = 'cadli'
    INSTRUMENT = asset_code.upper()
    AGGREGATE = 1

    # If user only wants small updates, reduce limit
    if max_hours:
        LIMIT = min(max_hours, 2000)
    else:
        LIMIT = 2000

    # Start from the current hour
    to_ts = int(time.time()) // 3600 * 3600
    all_data = []

    # Convert start_date to timestamp if provided
    from_ts = int(pd.Timestamp(start_date).timestamp()) if start_date else None

    # Smart condition: if we only need a few hours, skip pagination entirely
    single_request = max_hours is not None and max_hours <= LIMIT

    # Use httpx.AsyncClient for async requests
    async with httpx.AsyncClient(timeout=600.0) as client:
        with tqdm(desc=f"Fetching {INSTRUMENT} data", leave=False) as pbar:
            while True:
                params = {
                    "market": MARKET,
                    "instrument": INSTRUMENT,
                    "limit": LIMIT,
                    "aggregate": AGGREGATE,
                    "fill": "true",
                    "apply_mapping": "true",
                    "response_format": "JSON",
                    "to_ts": to_ts,
                }
    
                # Use await client.get() instead of requests.get()
                response = await client.get(BASE_URL, params=params, headers={"Content-type": "application/json; charset=UTF-8"})
    
                try:
                    # Check for HTTP errors
                    response.raise_for_status()
                    json_data = response.json()
                except (ValueError, httpx.HTTPStatusError) as e:
                    print(f"❌ Failed to fetch {INSTRUMENT}: {e}")
                    break
    
                data = json_data.get("Data", [])
                if not data:
                    # Don't print "Reached beginning" if it was just a small, single request
                    if not single_request:
                        print(f"✅ Reached beginning of data for {INSTRUMENT}.")
                    break
    
                all_data.extend(data)
                pbar.update(len(data))
    
                oldest_ts = min(d['TIMESTAMP'] for d in data)
                to_ts = oldest_ts - 3600  # move 1 hour backward
    
                # stop conditions
                if single_request:
                    break  # we only wanted a single batch (fast mode)
                if from_ts and oldest_ts < from_ts:
                    break

    if not all_data:
        # Don't print warning if we just fetched 0 new rows on an incremental update
        if not (max_hours and single_request):
            print(f"⚠️ No data for {INSTRUMENT}")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df[["TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"]]
    df = df.rename(columns={
        "TIMESTAMP": "timestamp",
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume"
    })
    df["asset"] = INSTRUMENT

    # Filter out older data if start_date provided
    if from_ts:
        df = df[df["timestamp"] >= from_ts]

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "asset", "open", "high", "low", "close", "volume"]]

async def update_asset(asset_code, lock: asyncio.Lock = None):
    asset = asset_code.upper()
    # init_db() is fast, no need to thread
    # init_db() 
    
    # Run blocking DB calls in a thread
    latest_ts = await asyncio.to_thread(get_latest_timestamp, asset)

    if latest_ts:
        start_date = pd.to_datetime(latest_ts, unit='s').strftime("%Y-%m-%d")
        # fetch only 72 hours (3 days) when updating incrementally
        # Await the async fetch function
        df = await fetch_crypto_data(asset, start_date=start_date, max_hours=72)
    else:
        # Await the async fetch function
        df = await fetch_crypto_data(asset)
        
    df.dropna(inplace=True)

    overlap_from_ts = latest_ts if latest_ts else None
    
    # Check if a lock was provided (it will be for batch updates)
    if lock:
        # Wait to acquire the lock before writing to the DB
        async with lock:
            rows = await asyncio.to_thread(
                upsert_data_replace_overlap, df, asset, overlap_from_ts
            )
    else:
        # No lock provided (e.g., single 'update' command), just run it
        rows = await asyncio.to_thread(
            upsert_data_replace_overlap, df, asset, overlap_from_ts
        )
    
    if rows > 0:
        print(f"✅ Updated {asset}: {rows} rows inserted (refreshed from {overlap_from_ts}).")
    else:
        print(f"✅ Updated {asset}: No new data.")

def get_all_tracked_assets():
    """Helper to synchronously get all assets from DB."""
    conn = sqlite3.connect(DB_FILE)
    assets = pd.read_sql_query("SELECT asset FROM tracked_assets", conn)["asset"].tolist()
    conn.close()
    return assets

async def async_update_list(asset_list, delay=3):
    """
    Asynchronously update a provided list of assets with a delay
    between starting each task to avoid rate limits.
    """
    if not asset_list:
        print("⚠️ No assets provided to update.")
        return
        
    print(f"⏳ Scheduling {len(asset_list)} assets for update (delay={delay}s)...")
    
    # Create one lock for all database operations
    db_lock = asyncio.Lock()
    
    tasks = []
    # Loop to *create* tasks with a delay
    for asset in tqdm(asset_list, desc="Scheduling tasks"):
        # Create the task and add it to the list
        tasks.append(asyncio.create_task(update_asset(asset, db_lock)))
        # Wait for the specified delay before scheduling the next one
        await asyncio.sleep(delay)

    print(f"All {len(tasks)} tasks scheduled, waiting for completion...")
    
    # tqdm.asyncio.gather will now monitor the list of already-running tasks
    await tqdm.gather(*tasks, desc="Updating assets")
    print(f"✅ All {len(asset_list)} assets processed.")
    
def update_all(delay):
    """Update all tracked crypto assets."""
    # init_db()
    assets = get_all_tracked_assets()

    if not assets:
        print("⚠️ No tracked assets to update.")
        return

    # Run the main async orchestrator, passing the delay
    asyncio.run(async_update_list(assets, delay))

# ========== SUMMARY / INFO ==========

def summary(asset_code):
    """Print summary information for a given asset."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        "SELECT MIN(timestamp) as first_ts, MAX(timestamp) as last_ts, COUNT(*) as rows FROM asset_prices WHERE asset = ?",
        conn, params=(asset_code.upper(),)
    )
    conn.close()

    if df.empty or df.iloc[0]["rows"] == 0:
        print(f"⚠️ No data for {asset_code}")
    else:
        row = df.iloc[0]
        first = pd.to_datetime(row["first_ts"], unit='s')
        last = pd.to_datetime(row["last_ts"], unit='s')
        print(f"📊 {asset_code.upper()} summary:")
        print(f" - Data points: {row['rows']}")
        print(f" - From: {first}")
        print(f" - To: {last}")

def import_json(json_path, do_update=False, delay=3):
    """
    Import assets from a JSON file of the form:
    {
        "LQ45": ["AADI", "ACES", ...]
    }
    Adds them to tracked_assets and optionally updates them immediately.
    """
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or not data:
        print("❌ Invalid JSON format. Expected { 'GroupName': [symbols...] }")
        return

    all_assets = []
    for group, assets in data.items():
        if isinstance(assets, list):
            all_assets.extend([a.upper() for a in assets])
        else:
            print(f"⚠️ Skipping {group}: expected a list of tickers")

    if not all_assets:
        print("⚠️ No valid assets found in JSON.")
        return

    print(f"📥 Importing {len(all_assets)} assets from {json_path}...")
    for asset in all_assets:
        add_asset(asset) # This is fast, no async needed

    print(f"✅ Added {len(all_assets)} assets to tracking list.")

    if do_update:
        # --- THIS IS THE CHANGED PART ---
        # Run the main async orchestrator
        asyncio.run(async_update_list(all_assets, delay))
        # --------------------------------

# ========== CLI HANDLER ==========

def main():
    parser = argparse.ArgumentParser(description="Manage and update asset price data using yfinance + SQLite.")
    
    parser.add_argument(
        '--delay',
        type=float,
        default=1,
        help="Delay (in seconds) between requests for batch operations (default: 0.5)"
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add
    add_parser = subparsers.add_parser("add", help="Add a new asset to tracking list")
    add_parser.add_argument("asset", help="Asset code (e.g. TSLA)")

    # delete
    del_parser = subparsers.add_parser("delete", help="Remove an asset from tracking list")
    del_parser.add_argument("asset", help="Asset code")

    # list
    subparsers.add_parser("list", help="List all tracked assets")

    # update single
    upd_parser = subparsers.add_parser("update", help="Update data for one asset")
    upd_parser.add_argument("asset", help="Asset code")

    # update all
    subparsers.add_parser("update-all", help="Update all tracked assets")

    # summary
    summary_parser = subparsers.add_parser("summary", help="Show data summary for an asset")
    summary_parser.add_argument("asset", help="Asset code")
    
    # import-json
    import_parser = subparsers.add_parser("import-json", help="Import and optionally update assets from a JSON file")
    import_parser.add_argument("json_file", help="Path to JSON file containing asset lists")
    import_parser.add_argument("--update", action="store_true", help="Update all imported assets after adding")

    args = parser.parse_args()
    init_db()

    if args.command == "add":
        add_asset(args.asset)
    elif args.command == "delete":
        delete_asset(args.asset)
    elif args.command == "list":
        list_assets()
    elif args.command == "update":
        # Single update doesn't need a batch delay
        asyncio.run(update_asset(args.asset))
    elif args.command == "update-all":
        # Pass the delay
        update_all(args.delay)
    elif args.command == "summary":
        summary(args.asset)
    elif args.command == "import-json":
        # Pass the delay
        import_json(args.json_file, do_update=args.update, delay=args.delay)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
