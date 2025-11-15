import math
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from runtime.manage_assets import init_db, import_json
from core.logger import get_logger
from core.db.utils import fetch_asset_from_db, DB_FILE
from core.smc.utils import smart_money_concept, ta_rsi, ta_mfi, ta_williams_r, ta_stoch

logger = get_logger(__name__)

TICKER_PATH = Path(__file__).parent / ".." / ".." / "runtime" / "TOP100.json"

def update_data() -> None:
    # Update all ticker data and store it in local sqlite
    init_db()
    import_json(json_path=TICKER_PATH, do_update=True, delay=1.0)
    logger.info("Price data has been updated!")

# --- Your existing function ---
# def ob_buy_metric(price, timestamp, iob_btm, iob_top, iob_created, decay_half_life=30*24*3600):
#     """
#     Calculate a desirability metric (0 to 1) for buying based on order block proximity and recency.

#     Parameters:
#         price (float): current asset price
#         timestamp (int): current time in timestamp (seconds)
#         iob_btm (float): bottom price of bullish order block
#         iob_top (float): top price of bullish order block
#         iob_created (int): timestamp (seconds) when the OB was created
#         decay_half_life (float): seconds until the OB influence halves (default: 30 days)

#     Returns:
#         float: desirability metric (0 = poor, 1 = excellent)
#     """
#     # --- normalize order block range ---
#     if iob_top < iob_btm:
#         iob_top, iob_btm = iob_btm, iob_top

#     ob_mid = (iob_top + iob_btm) / 2
#     ob_range = iob_top - iob_btm
#     if ob_range == 0:
#         return 0.0

#     # --- position-based score ---
#     if iob_btm <= price <= iob_top:
#         # inside OB → high score (max at middle)
#         position_score = 1 - abs(price - ob_mid) / (ob_range / 2)
#     else:
#         # outside OB → exponential decay based on distance
#         distance = min(abs(price - iob_top), abs(price - iob_btm))
#         position_score = math.exp(-3 * distance / ob_range)

#     # --- recency-based weight ---
#     age_seconds = timestamp - iob_created
#     age_factor = math.exp(-math.log(2) * age_seconds / decay_half_life)  # half-life decay

#     # --- final metric ---
#     metric = position_score * age_factor
#     return max(0.0, min(1.0, metric))


# --- NEW Symmetrical function for Selling ---
# def ob_sell_metric(price, timestamp, iob_btm, iob_top, iob_created, decay_half_life=30*24*3600):
#     """
#     Calculate a desirability metric (0 to 1) for selling based on order block proximity and recency.

#     Parameters:
#         price (float): current asset price
#         timestamp (int): current time in timestamp (seconds)
#         iob_btm (float): bottom price of bearish order block
#         iob_top (float): top price of bearish order block
#         iob_created (int): timestamp (seconds) when the OB was created
#         decay_half_life (float): seconds until the OB influence halves (default: 30 days)

#     Returns:
#         float: desirability metric (0 = poor, 1 = excellent)
#     """
#     # --- normalize order block range ---
#     if iob_top < iob_btm:
#         # This handles any case where top/btm might be swapped
#         iob_top, iob_btm = iob_btm, iob_top

#     ob_mid = (iob_top + iob_btm) / 2
#     ob_range = iob_top - iob_btm
#     if ob_range == 0:
#         # Avoid division by zero if OB has no height
#         return 0.0

#     # --- position-based score ---
#     # The logic is identical to the buy metric, as the "ideal" entry
#     # is the 50% midpoint, regardless of direction.
#     if iob_btm <= price <= iob_top:
#         # Price is inside the OB. Score is 1.0 at the midpoint, 0.0 at the edges.
#         position_score = 1 - abs(price - ob_mid) / (ob_range / 2)
#     else:
#         # Price is outside the OB. Score decays exponentially based on distance
#         # from the nearest edge. This rewards price being *close* to the OB.
#         distance = min(abs(price - iob_top), abs(price - iob_btm))
#         position_score = math.exp(-3 * distance / ob_range)

#     # --- recency-based weight ---
#     # This logic is also identical. Newer OBs are weighted higher.
#     age_seconds = timestamp - iob_created
#     # Standard exponential decay formula for half-life
#     age_factor = math.exp(-math.log(2) * age_seconds / decay_half_life)

#     # --- final metric ---
#     metric = position_score * age_factor
#     # Clamp the final score between 0.0 and 1.0
#     return max(0.0, min(1.0, metric))

def ob_metric(price, timestamp, iob_btm, iob_top, iob_created):
    if iob_top < iob_btm:
        iob_top, iob_btm = iob_btm, iob_top

    ob_mid = (iob_top + iob_btm) / 2
    ob_range = iob_top - iob_btm
    half_range = ob_range / 2

    # INSIDE OB — parabolic curve, always >= 0.5
    if iob_btm <= price <= iob_top:
        x = abs(price - ob_mid) / half_range   # 0 → 1
        position_score = 0.5 + 0.5 * (1 - x*x)

    # OUTSIDE OB — exponential decay from 0.5
    else:
        distance = min(abs(price - iob_top), abs(price - iob_btm))
        k = 3 / ob_range
        position_score = 0.5 * math.exp(-k * distance)

    # AGE penalty (unchanged)
    # age_seconds = timestamp - iob_created
    # age_factor = math.exp(-math.log(2) * age_seconds / decay_half_life)

    return max(0, min(1, position_score))  # * age_factor

def score_divergence(bd_dict: dict):
    """
    Computes an auto-weighted divergence score (0−1) using only bd_dict.

    bd_dict format:
        {
            'rsi':  [True, False, True, False],
            'macd': [False, True, True, False],
            ...
        }

    Design:
    - Equal weights across metrics
    - Recency-weighted swings: w[i] = (N - i) / sum(1..N)
      meaning newest swings contribute most
    - Score ∈ [0, 1]
    """

    if not bd_dict or not isinstance(bd_dict, dict):
        return 0.0

    metrics = list(bd_dict.keys())
    num_metrics = len(metrics)
    if num_metrics == 0:
        return 0.0

    list_len = len(bd_dict[metrics[0]])
    if list_len == 0:
        return 0.0

    # --- Equal metric weights ---
    metric_weight = 1.0 / num_metrics

    # --- Recency-based swing weights ---
    # Newest (index 0) has highest weight
    # w[i] = (N - i) / sum(1..N)
    denominator = sum(range(1, list_len + 1))
    swing_weights = [(list_len - i) / denominator for i in range(list_len)]

    # --- Compute score ---
    weighted_sum = 0.0
    total_weight = 0.0

    for mt in metrics:
        results = bd_dict[mt]

        for i in range(list_len):
            val = 1.0 if results[i] else 0.0
            weight = metric_weight * swing_weights[i]

            weighted_sum += val * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return float(weighted_sum / total_weight)

def get_ticker() -> List[str]:
    with open(TICKER_PATH, "r") as f:
        ticker_dict: Dict = json.load(f)
    return list(ticker_dict.values())[0]

# with open("divergence.json", "w") as f:
#     json.dump({}, f, indent=2)

def generate_signals(
    n_swing_lookback: int = 4,
    include_hidden: bool = True,
    osc_tolerance: float = 0.25,
    osc_std_window: int = 50,
    MA_period: int = 3,
    min_swing_distance: int = 3,
    period: str = "1y",
    interval: str = "4h",
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
    for asset_idx, asset_code in tqdm(enumerate(TICKER), desc="Processing TICKER assets", unit="asset"):
        # Fetch Asset Price
        try:
            df = fetch_asset_from_db(
                asset_code,
                period=period if asset_idx < int(0.1*len(TICKER)) else "1mo",
                interval=interval
            )
            df.reset_index(inplace=True)
            df['timestamp'] = pd.to_datetime(df['Date']).astype('int64') // 10**9
            df["mfi"] = ta_mfi(df, fill_na=True)
            df["stoch"] = ta_stoch(df, fill_na=True)
            df["rsi"] = ta_rsi(df, fill_na=True)
            df["williams_r"] = ta_williams_r(df, fill_na=True)
        except Exception as e:
            logger.error(f"Error when fetch asset: {asset_code}: {str(e)}")
            continue
        
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
        
        internal_swing_length = config.get("internal_swing_length")
        
        # SMC
        smc_result = smart_money_concept(
            df=df,
            config=config,
            verbose=0,
        )
        df_iob = smc_result.get("df_iob")
        df_iob['iob_left_utc'] = pd.to_datetime(df_iob['iob_left'], unit='s', utc=True)

        # --- basic smoothing price MAs (as in your original snippet) ---
        df['High_MA'] = df['High'].rolling(window=MA_period, min_periods=1).mean()
        df['Low_MA']  = df['Low'].rolling(window=MA_period, min_periods=1).mean()

        # --- precompute oscillator rolling std for adaptive tolerance ---
        osc_std = {}
        for mt in extra_metrics:
            osc_std[mt] = df[mt].rolling(window=osc_std_window, min_periods=1).std().fillna(0.0)

        # helper to safely get slice bounds
        def swing_window_bounds(idx):
            start = max(0, idx - internal_swing_length)
            end = min(len(df) - 1, idx)  # inclusive index
            return start, end + 1         # slice end exclusive

        # helper to extract osc extreme for a swing low/high
        def osc_extreme_for_swing(idx, mt, swing_type='low'):
            s, e = swing_window_bounds(idx)
            window = df[mt].iloc[s:e]
            if window.empty:
                return df.at[idx, mt]
            return window.min() if swing_type == 'low' else window.max()

        # helper to compute adaptive tol at an index for a metric
        def adaptive_tol_at(idx, mt):
            base_std = osc_std[mt].iat[idx] if idx < len(osc_std[mt]) else osc_std[mt].iat[-1]
            return osc_tolerance * base_std

        # Initialize outputs (one for bullish, one for bearish)
        bd_dict_bull = {mt: [False] * n_swing_lookback for mt in extra_metrics}
        bd_dict_bear = {mt: [False] * n_swing_lookback for mt in extra_metrics}

        # store swing indices (not prices) for bottoms and tops
        prev_ibtm_indices: List[int] = []
        prev_itop_indices: List[int] = []
        
        # prev_ibtm_dates: List = []

        # iterate rows to detect new swings (assumes df has columns 'ibtm' and 'itop' exactly like you used)
        for i in range(len(df)):
            # ---------- BULL (bottom swings) ----------
            if df.at[i, "ibtm"] != 0:
                new_idx = i - internal_swing_length
                # prev_ibtm_dates.append({
                #     "date": df.at[new_idx, "Date"].isoformat(),
                #     "osc": df.at[new_idx, "mfi"]
                # })
                
                # safety clamp
                if new_idx < 0 or new_idx >= len(df):
                    prev_ibtm_indices.append(new_idx)
                    continue

                # enforce min spacing
                if prev_ibtm_indices and (new_idx - prev_ibtm_indices[-1]) < min_swing_distance:
                    # skip adding this swing as it's too close to the previous
                    # (you may want to keep it instead — this is a choice)
                    prev_ibtm_indices.append(new_idx)
                    continue

                # compare against last N previous bottoms
                if prev_ibtm_indices:
                    curr_idx = new_idx
                    curr_low_ma = df.at[curr_idx, "Low_MA"]
                    curr_price = df.at[curr_idx, "Low"]

                    # results per metric for this newest swing
                    current_swing_results = {mt: [] for mt in extra_metrics}

                    # iterate previous swings from nearest to older
                    for prev_idx in prev_ibtm_indices[-n_swing_lookback:][::-1]:
                        if prev_idx < 0 or prev_idx >= len(df):
                            # pad with False if an invalid index
                            for mt in extra_metrics:
                                current_swing_results[mt].append(False)
                            continue

                        prev_low_ma = df.at[prev_idx, "Low_MA"]
                        prev_price  = df.at[prev_idx, "Low"]

                        for mt in extra_metrics:
                            # oscillator value of interest for swing lows -> use min in the swing window
                            curr_osc = osc_extreme_for_swing(curr_idx, mt, swing_type='low')
                            prev_osc = osc_extreme_for_swing(prev_idx, mt, swing_type='low')

                            tol = adaptive_tol_at(curr_idx, mt)

                            # Regular Bullish: Price LL, Osc HL
                            price_down = (curr_low_ma <= prev_low_ma) or (curr_price < prev_price)
                            osc_up     = (curr_osc > prev_osc + tol)

                            # Hidden Bullish: Price HL, Osc LL
                            price_up   = (curr_low_ma > prev_low_ma) or (curr_price > prev_price)
                            osc_down   = (curr_osc < prev_osc - tol)

                            found_div = (osc_up and price_down) or (include_hidden and osc_down and price_up)
                            current_swing_results[mt].append(bool(found_div))

                    # pad to n_swing_lookback if fewer predecessors
                    for mt in extra_metrics:
                        lst = current_swing_results[mt]
                        padded = lst + [False] * (n_swing_lookback - len(lst))
                        bd_dict_bull[mt] = padded

                prev_ibtm_indices.append(new_idx)

            # ---------- BEAR (top swings) ----------
            if df.at[i, "itop"] != 0:
                new_idx = i - internal_swing_length
                if new_idx < 0 or new_idx >= len(df):
                    prev_itop_indices.append(new_idx)
                    continue

                if prev_itop_indices and (new_idx - prev_itop_indices[-1]) < min_swing_distance:
                    prev_itop_indices.append(new_idx)
                    continue

                if prev_itop_indices:
                    curr_idx = new_idx
                    curr_high_ma = df.at[curr_idx, "High_MA"]
                    curr_price = df.at[curr_idx, "High"]

                    current_swing_results = {mt: [] for mt in extra_metrics}

                    for prev_idx in prev_itop_indices[-n_swing_lookback:][::-1]:
                        if prev_idx < 0 or prev_idx >= len(df):
                            for mt in extra_metrics:
                                current_swing_results[mt].append(False)
                            continue

                        prev_high_ma = df.at[prev_idx, "High_MA"]
                        prev_price   = df.at[prev_idx, "High"]

                        for mt in extra_metrics:
                            # for tops, oscillator extreme is max over the swing window
                            curr_osc = osc_extreme_for_swing(curr_idx, mt, swing_type='high')
                            prev_osc = osc_extreme_for_swing(prev_idx, mt, swing_type='high')

                            tol = adaptive_tol_at(curr_idx, mt)

                            # Regular Bearish: Price HH, Osc LH
                            price_up  = (curr_high_ma > prev_high_ma) or (curr_price > prev_price)
                            osc_down  = (curr_osc < prev_osc - tol)

                            # Hidden Bearish: Price LH, Osc HH
                            price_down = (curr_high_ma <= prev_high_ma) or (curr_price < prev_price)
                            osc_up     = (curr_osc > prev_osc + tol)

                            found_div = (osc_down and price_up) or (include_hidden and osc_up and price_down)
                            current_swing_results[mt].append(bool(found_div))

                    # pad and write to final dict
                    for mt in extra_metrics:
                        lst = current_swing_results[mt]
                        padded = lst + [False] * (n_swing_lookback - len(lst))
                        bd_dict_bear[mt] = padded

                prev_itop_indices.append(new_idx)
        # print(prev_ibtm_dates)
        # with open("divergence.json", "r") as f:
        #     dvg = json.load(f)
        # dvg[asset_code] = prev_ibtm_dates
        # with open("divergence.json", "w") as f:
        #     json.dump(dvg, f, indent=2)
        
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
        long_divergence_metric = np.nan
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
            
            long_metric = ob_metric(
                price=price,
                timestamp=time_now_df,
                iob_btm=long_row["iob_btm"],
                iob_top=long_row["iob_top"],
                iob_created=long_row["iob_created"],
            )
            long_divergence_metric = score_divergence(
                bd_dict_bull
            )
        
        long_row["ob_metric"] = long_metric
        long_row["divergence_metric"] = long_divergence_metric
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
        short_divergence_metric = np.nan
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
            short_metric = ob_metric(
                price=price,
                timestamp=time_now_df,
                iob_btm=short_row["iob_btm"],
                iob_top=short_row["iob_top"],
                iob_created=short_row["iob_created"],
            )
            short_divergence_metric = score_divergence(
                bd_dict_bear
            )
            
        short_row["ob_metric"] = short_metric
        short_row["divergence_metric"] = short_divergence_metric
        # Add bearish divergence data
        short_row.update({f"{mt}_bd": bd_dict_bear[mt] for mt in extra_metrics})
        short_data.append(short_row)

    # --- Create Final DataFrames ---

    # Column order for readability
    columns_order = [
        "asset", "price", "ob_metric", "divergence_metric", "iob_top", "iob_btm", 
        "iob_left", "iob_created", "iob_left_utc",
        "mfi_bd", "stoch_bd", "rsi_bd", "williams_r_bd"
    ]

    # Long results
    df_long_results = pd.DataFrame(long_data)
    df_long_results["ob_metric"] = np.round(df_long_results["ob_metric"], 3)
    df_long_results["divergence_metric"] = np.round(df_long_results["divergence_metric"], 3)
    df_long_results = df_long_results[columns_order].sort_values(by="ob_metric", ascending=False).reset_index(drop=True)

    # Short results
    df_short_results = pd.DataFrame(short_data)
    df_short_results["ob_metric"] = np.round(df_short_results["ob_metric"], 3)
    df_short_results["divergence_metric"] = np.round(df_short_results["divergence_metric"], 3)
    df_short_results = df_short_results[columns_order].sort_values(by="ob_metric", ascending=False).reset_index(drop=True)

    return df_long_results, df_short_results
    # --- Display Results ---
    # print("--- Best Long (Bullish) Setups ---")
    # print(df_long_results.head())
    # print("\n" + "="*30 + "\n")
    # print("--- Best Short (Bearish) Setups ---")
    # print(df_short_results.head())

    # You can now use df_long_results and df_short_results as needed