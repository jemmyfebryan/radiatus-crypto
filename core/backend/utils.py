import math
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from runtime.manage_assets import import_json
from core.logger import get_logger
from core.db.utils import fetch_asset_from_db, DB_FILE
from core.smc.utils import smart_money_concept, ta_rsi, ta_mfi, ta_williams_r, ta_stoch

logger = get_logger(__name__)

TICKER_PATH = Path(__file__).parent / ".." / ".." / "runtime" / "TOP100.json"

def update_data() -> None:
    # Update all ticker data and store it in local sqlite
    import_json(json_path=TICKER_PATH, do_update=True, delay=1.0)
    logger.info("Price data has been updated!")

# --- Your existing function ---
def ob_buy_metric(price, timestamp, iob_btm, iob_top, iob_created, decay_half_life=30*24*3600):
    """
    Calculate a desirability metric (0 to 1) for buying based on order block proximity and recency.

    Parameters:
        price (float): current asset price
        timestamp (int): current time in timestamp (seconds)
        iob_btm (float): bottom price of bullish order block
        iob_top (float): top price of bullish order block
        iob_created (int): timestamp (seconds) when the OB was created
        decay_half_life (float): seconds until the OB influence halves (default: 30 days)

    Returns:
        float: desirability metric (0 = poor, 1 = excellent)
    """
    # --- normalize order block range ---
    if iob_top < iob_btm:
        iob_top, iob_btm = iob_btm, iob_top

    ob_mid = (iob_top + iob_btm) / 2
    ob_range = iob_top - iob_btm
    if ob_range == 0:
        return 0.0

    # --- position-based score ---
    if iob_btm <= price <= iob_top:
        # inside OB → high score (max at middle)
        position_score = 1 - abs(price - ob_mid) / (ob_range / 2)
    else:
        # outside OB → exponential decay based on distance
        distance = min(abs(price - iob_top), abs(price - iob_btm))
        position_score = math.exp(-3 * distance / ob_range)

    # --- recency-based weight ---
    age_seconds = timestamp - iob_created
    age_factor = math.exp(-math.log(2) * age_seconds / decay_half_life)  # half-life decay

    # --- final metric ---
    metric = position_score * age_factor
    return max(0.0, min(1.0, metric))


# --- NEW Symmetrical function for Selling ---
def ob_sell_metric(price, timestamp, iob_btm, iob_top, iob_created, decay_half_life=30*24*3600):
    """
    Calculate a desirability metric (0 to 1) for selling based on order block proximity and recency.

    Parameters:
        price (float): current asset price
        timestamp (int): current time in timestamp (seconds)
        iob_btm (float): bottom price of bearish order block
        iob_top (float): top price of bearish order block
        iob_created (int): timestamp (seconds) when the OB was created
        decay_half_life (float): seconds until the OB influence halves (default: 30 days)

    Returns:
        float: desirability metric (0 = poor, 1 = excellent)
    """
    # --- normalize order block range ---
    if iob_top < iob_btm:
        # This handles any case where top/btm might be swapped
        iob_top, iob_btm = iob_btm, iob_top

    ob_mid = (iob_top + iob_btm) / 2
    ob_range = iob_top - iob_btm
    if ob_range == 0:
        # Avoid division by zero if OB has no height
        return 0.0

    # --- position-based score ---
    # The logic is identical to the buy metric, as the "ideal" entry
    # is the 50% midpoint, regardless of direction.
    if iob_btm <= price <= iob_top:
        # Price is inside the OB. Score is 1.0 at the midpoint, 0.0 at the edges.
        position_score = 1 - abs(price - ob_mid) / (ob_range / 2)
    else:
        # Price is outside the OB. Score decays exponentially based on distance
        # from the nearest edge. This rewards price being *close* to the OB.
        distance = min(abs(price - iob_top), abs(price - iob_btm))
        position_score = math.exp(-3 * distance / ob_range)

    # --- recency-based weight ---
    # This logic is also identical. Newer OBs are weighted higher.
    age_seconds = timestamp - iob_created
    # Standard exponential decay formula for half-life
    age_factor = math.exp(-math.log(2) * age_seconds / decay_half_life)

    # --- final metric ---
    metric = position_score * age_factor
    # Clamp the final score between 0.0 and 1.0
    return max(0.0, min(1.0, metric))

def get_ticker() -> List[str]:
    with open(TICKER_PATH, "r") as f:
        ticker_dict: Dict = json.load(f)
    return list(ticker_dict.values())[0]

def generate_signals(
    include_hidden: bool = True,
    osc_tolerance: int = 10,
    n_swing_lookback: int = 3,
    period: str = "1mo",
    interval: str = "4h"
):
    TICKER = get_ticker()
    
    # Configuration
    # --- Bull Divergence ---
    extra_metrics = ["mfi", "rsi", "stoch", "williams_r"]
    
    # --- NEW PARAMETER ---
    # Set to 1 for original behavior (only look at 1 previous swing).
    # Set to 3 (or more) to look back at the last N swings.
    
    asset_iobs = {}
    long_data = []
    short_data = []

    # Add tqdm progress bar
    for asset_code in tqdm(TICKER, desc="Processing TICKER assets", unit="asset"):
        # Fetch Asset Price
        df = fetch_asset_from_db(asset_code, period=period, interval=interval)
        df.reset_index(inplace=True)
        df['timestamp'] = pd.to_datetime(df['Date']).astype('int64') // 10**9
        df["mfi"] = ta_mfi(df, fill_na=True)
        df["stoch"] = ta_stoch(df, fill_na=True)
        df["rsi"] = ta_rsi(df, fill_na=True)
        df["williams_r"] = ta_williams_r(df, fill_na=True)
        
        price = df.at[len(df)-1, "Close"]
        
        time_now_df = int(df.at[len(df)-1, "timestamp"])
        
        # SMC Configuration
        config = {
            "show_swings": False,
            "show_iob": True,
            "confluence_filter": False,
            "swing_length": 50,
            "internal_swing_length": 5,
            "ob_filter": "atr",
            "iob_showlast": 5,
            "delete_broken_iob": False,
            
            # Tune-able parameters
            "atr_period": 200,
            "ob_threshold_mult": 2.0,
            "swing_tolerance_type": None,  # 'atr', 'pct', or 'points'. Set to None to disable.
            "swing_tolerance_value": 0,    # e.g., 0.5 = 0.5 * ATR. If 'pct', 0.01 = 1%.   
        }
        
        # SMC
        smc_result = smart_money_concept(
            df=df,
            config=config,
            verbose=0,
        )
        df_iob = smc_result.get("df_iob")
        df_iob['iob_left_utc'] = pd.to_datetime(df_iob['iob_left'], unit='s', utc=True)
        
        for mt in extra_metrics:
            df[f"bull_divergence_{mt}"] = False

        # Store previous (price, index) tuples
        prev_ibtms_list = []
        
        for i in range(len(df)):
            if prev_ibtms_list: # Check if we have at least one previous swing
                curr_idx = i
                curr_low = df.at[i, "Low"]
                
                # Tracks if we've found a div FOR THIS BAR 'i'
                found_divergence_on_this_bar = {mt: False for mt in extra_metrics}
                
                # Loop through the last N swings, starting from the *most recent*
                for prev_ibtm, prev_ibtm_idx in prev_ibtms_list[-n_swing_lookback:][::-1]:
                    
                    for mt in extra_metrics:
                        # If we already found a div for this metric on bar 'i'
                        # (from a more recent swing), skip checking older swings.
                        if found_divergence_on_this_bar[mt]:
                            continue
                            
                        curr_osc = df.at[curr_idx, mt]
                        prev_osc = df.at[prev_ibtm_idx, mt]
                        
                        # Regular Bullish: Price LL, Osc HL
                        osc_up = (curr_osc > prev_osc + osc_tolerance)
                        price_down = (curr_low <= prev_ibtm)
                        
                        # Hidden Bullish: Price HL, Osc LL
                        osc_down = (curr_osc <= prev_osc - osc_tolerance)
                        price_up = (curr_low > prev_ibtm)
                    
                        if (osc_up and price_down) or (include_hidden and osc_down and price_up):
                            df.loc[i, f"bull_divergence_{mt}"] = True
                            found_divergence_on_this_bar[mt] = True # Mark as found
                
            if df.at[i, "ibtm"] != 0:
                # Append the new swing (price, index) to the list
                new_ibtm_price = df.at[i, "ibtm"]
                new_ibtm_idx = i - config.get("internal_swing_length")
                prev_ibtms_list.append((new_ibtm_price, new_ibtm_idx))
        
        bd_dict_bull = {}
        for mt in extra_metrics:
            # Store only the last 3 divergence signals
            bd_dict_bull[mt] = df[f"bull_divergence_{mt}"].tolist()[-3:]

        # --- Bearish Divergence (New) ---
        for mt in extra_metrics:
            df[f"bear_divergence_{mt}"] = False

        # Store previous (price, index) tuples
        prev_itops_list = []
        
        for i in range(len(df)):
            if prev_itops_list: # Check if we have at least one previous swing
                curr_idx = i
                curr_high = df.at[i, "High"] # Use High for bearish
                
                # Tracks if we've found a div FOR THIS BAR 'i'
                found_divergence_on_this_bar = {mt: False for mt in extra_metrics}
                
                # Loop through the last N swings, starting from the *most recent*
                for prev_itop, prev_itop_idx in prev_itops_list[-n_swing_lookback:][::-1]:
                    
                    for mt in extra_metrics:
                        # If we already found a div for this metric on bar 'i', skip.
                        if found_divergence_on_this_bar[mt]:
                            continue
                            
                        curr_osc = df.at[curr_idx, mt]
                        prev_osc = df.at[prev_itop_idx, mt]
                        
                        # Regular Bearish: Price HH, Osc LH
                        osc_down = (curr_osc <= prev_osc - osc_tolerance)
                        price_up = (curr_high > prev_itop)
                        
                        # Hidden Bearish: Price LH, Osc HH
                        osc_up = (curr_osc > prev_osc + osc_tolerance)
                        price_down = (curr_high <= prev_itop)
                    
                        if (osc_down and price_up) or (include_hidden and osc_up and price_down):
                            df.loc[i, f"bear_divergence_{mt}"] = True
                            found_divergence_on_this_bar[mt] = True # Mark as found
                
            if df.at[i, "itop"] != 0: # Use itop
                # Append the new swing (price, index) to the list
                new_itop_price = df.at[i, "itop"]
                new_itop_idx = i - config.get("internal_swing_length")
                prev_itops_list.append((new_itop_price, new_itop_idx))
                
        bd_dict_bear = {}
        for mt in extra_metrics:
            # Store only the last 3 divergence signals
            bd_dict_bear[mt] = df[f"bear_divergence_{mt}"].tolist()[-3:]
        
        # --- Get IOBs ---
        asset_iobs[asset_code] = df_iob.copy()
        df_iob_valid: pd.DataFrame = df_iob[df_iob["iob_invalid"] == 0]
        
        # --- Process LONG (Bullish) ---
        df_iob_bullish_valid: pd.DataFrame = df_iob_valid[df_iob_valid["iob_type"] == 1]
        df_iob_final_long = (
            df_iob_bullish_valid[["iob_top", "iob_btm", "iob_left", "iob_created", "iob_left_utc"]]
            .copy()
            .sort_values(by="iob_created", ascending=False)
            .reset_index(drop=True)
        )
        
        long_metric = np.nan
        # Initialize with NaNs
        long_row = {
            "asset": asset_code, "price": price, 
            "iob_top": np.nan, "iob_btm": np.nan, "iob_left": np.nan, 
            "iob_created": np.nan, "iob_left_utc": pd.NaT
        }
        
        if not df_iob_final_long.empty:
            long_row["iob_top"] = df_iob_final_long.at[0, "iob_top"]
            long_row["iob_btm"] = df_iob_final_long.at[0, "iob_btm"]
            long_row["iob_left"] = df_iob_final_long.at[0, "iob_left"]
            long_row["iob_created"] = df_iob_final_long.at[0, "iob_created"]
            long_row["iob_left_utc"] = df_iob_final_long.at[0, "iob_left_utc"]
            
            long_metric = ob_buy_metric(
                price=price,
                timestamp=time_now_df,
                iob_btm=long_row["iob_btm"],
                iob_top=long_row["iob_top"],
                iob_created=long_row["iob_created"],
            )
        
        long_row["metric"] = long_metric
        # Add bullish divergence data
        long_row.update({f"{mt}_bd": bd_dict_bull[mt] for mt in extra_metrics})
        long_data.append(long_row)
        
        
        # --- Process SHORT (Bearish) ---
        df_iob_bearish_valid: pd.DataFrame = df_iob_valid[df_iob_valid["iob_type"] == -1] # Assuming -1 for bearish
        df_iob_final_short = (
            df_iob_bearish_valid[["iob_top", "iob_btm", "iob_left", "iob_created", "iob_left_utc"]]
            .copy()
            .sort_values(by="iob_created", ascending=False)
            .reset_index(drop=True)
        )
        
        short_metric = np.nan
        # Initialize with NaNs
        short_row = {
            "asset": asset_code, "price": price, 
            "iob_top": np.nan, "iob_btm": np.nan, "iob_left": np.nan, 
            "iob_created": np.nan, "iob_left_utc": pd.NaT
        }
        
        if not df_iob_final_short.empty:
            short_row["iob_top"] = df_iob_final_short.at[0, "iob_top"]
            short_row["iob_btm"] = df_iob_final_short.at[0, "iob_btm"]
            short_row["iob_left"] = df_iob_final_short.at[0, "iob_left"]
            short_row["iob_created"] = df_iob_final_short.at[0, "iob_created"]
            short_row["iob_left_utc"] = df_iob_final_short.at[0, "iob_left_utc"]
            
            # !!! IMPORTANT: You must have ob_sell_metric defined !!!
            short_metric = ob_sell_metric(
                price=price,
                timestamp=time_now_df,
                iob_btm=short_row["iob_btm"],
                iob_top=short_row["iob_top"],
                iob_created=short_row["iob_created"],
            )
            
        short_row["metric"] = short_metric
        # Add bearish divergence data
        short_row.update({f"{mt}_bd": bd_dict_bear[mt] for mt in extra_metrics})
        short_data.append(short_row)

    # --- Create Final DataFrames ---

    # Column order for readability
    columns_order = [
        "asset", "price", "metric", "iob_top", "iob_btm", 
        "iob_left", "iob_created", "iob_left_utc",
        "mfi_bd", "stoch_bd", "rsi_bd", "williams_r_bd"
    ]

    # Long results
    df_long_results = pd.DataFrame(long_data)
    df_long_results["metric"] = np.round(df_long_results["metric"], 3)
    df_long_results = df_long_results[columns_order].sort_values(by="metric", ascending=False).reset_index(drop=True)

    # Short results
    df_short_results = pd.DataFrame(short_data)
    df_short_results["metric"] = np.round(df_short_results["metric"], 3)
    df_short_results = df_short_results[columns_order].sort_values(by="metric", ascending=False).reset_index(drop=True)

    return df_long_results, df_short_results
    # --- Display Results ---
    # print("--- Best Long (Bullish) Setups ---")
    # print(df_long_results.head())
    # print("\n" + "="*30 + "\n")
    # print("--- Best Short (Bearish) Setups ---")
    # print(df_short_results.head())

    # You can now use df_long_results and df_short_results as needed