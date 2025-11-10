import sqlite3
import pandas as pd
import struct
from datetime import datetime, timedelta

from core.db.config import DB_FILE

def period_to_start_date(period: str) -> str:
    today = datetime.utcnow().date()
    mapping = {
        "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
        "6mo": 180, "1y": 365, "2y": 730,
        "5y": 1825, "10y": 3650,
    }
    if period == "max":
        return None
    elif period in mapping:
        start_date = today - timedelta(days=mapping[period])
        return start_date.strftime("%Y-%m-%d")
    else:
        raise ValueError(f"Unsupported period: {period}")


def _decode_blob(value):
    """Decode SQLite BLOBs (from numpy numeric types) into Python numbers."""
    if isinstance(value, bytes):
        try:
            # Try unsigned 8-byte integer first
            return struct.unpack("<Q", value)[0]
        except struct.error:
            try:
                # Fallback: signed 8-byte integer
                return struct.unpack("<q", value)[0]
            except struct.error:
                return None
    return value

def get_db_conn():
    """Helper function to get a robust, concurrent-safe connection."""
    conn = sqlite3.connect(DB_FILE, timeout=30.0)  # Solution 1: Add timeout
    conn.execute("PRAGMA journal_mode=WAL;")      # Solution 2: Enable WAL
    return conn

def fetch_asset_from_db(
    asset_code: str, 
    period: str = "max", 
    interval: str = "1h" # <-- ADDED INTERVAL ARG
) -> pd.DataFrame:
    """
    Fetch asset OHLCV data from local SQLite DB, with resampling.
    Returns a DataFrame like yfinance.Ticker(...).history().
    
    Args:
        asset_code (str): The asset (e.g., "BTC-USD").
        period (str): The amount of history to fetch ("1y", "6mo", "max", etc.).
        interval (str): The desired data interval ("1h", "4h", "1d", "1w").
    """
    conn = get_db_conn()
    
    start_date_str = period_to_start_date(period)
    start_timestamp = None
    params = [asset_code.upper()]

    if start_date_str:
        start_timestamp = int(pd.Timestamp(start_date_str).timestamp())

    # Query always fetches the base 1-hour data
    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM asset_prices
        WHERE asset = ?
    """
    if start_timestamp:
        query += " AND timestamp >= ?"
        params.append(start_timestamp)
        
    query += " ORDER BY timestamp ASC"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        print(f"⚠️ No data found for {asset_code} (period={period})")
        return pd.DataFrame()

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].apply(_decode_blob)

    # --- Standard Formatting ---
    df["Date"] = pd.to_datetime(df["timestamp"], unit='s') 
    df = df.drop(columns=["timestamp"]) 
    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })
    df = df.set_index("Date")
    df["Volume"] = df["Volume"].astype(int)

    # --- NEW RESAMPLING LOGIC ---
    
    # If interval is '1h' (or '1H'), data is already in correct format
    if interval.lower() in ["1h", "h"]:
        return df

    try:
        # Define aggregation rules for OHLCV
        agg_rules = {
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }

        # Resample the 1-hour data to the target interval (e.g., "4h", "1d", "1w")
        resampled_df = df.resample(interval).agg(agg_rules)

        # Drop rows where all values are NaN (which happens for empty periods)
        resampled_df = resampled_df.dropna(how='all')
        
        # Ensure Volume is an integer type (Int64 supports NaNs)
        if 'Volume' in resampled_df.columns:
            resampled_df['Volume'] = resampled_df['Volume'].astype('Int64')

        return resampled_df
        
    except ValueError as e:
        # This catches invalid interval strings (e.g., "5m", "1z")
        print(f"❌ Invalid interval string: {interval}. Error: {e}")
        print(f"Returning default 1h data.")
        return df