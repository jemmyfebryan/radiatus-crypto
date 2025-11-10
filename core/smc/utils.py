import pandas as pd
import numpy as np

import time

from typing import List, Dict

def ta_atr(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """
    Calculate the Average True Range (ATR) for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["Open", "High", "Low", "Close"].
    period : int, optional
        The number of periods to use for the ATR calculation (default is 200).

    Returns
    -------
    pd.Series
        A pandas Series containing the ATR values.
    """
    # Ensure required columns exist
    required_cols = {"High", "Low", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # True Range components
    high_low = df["High"] - df["Low"]
    high_close_prev = (df["High"] - df["Close"].shift()).abs()
    low_close_prev = (df["Low"] - df["Close"].shift()).abs()

    # True Range = max(high_low, high_close_prev, low_close_prev)
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)

    # ATR = Exponential Moving Average (like in TradingView / ta.atr)
    atr = tr.ewm(span=period, adjust=False).mean()

    return atr

def ta_mfi(df: pd.DataFrame, period: int = 14, fill_na: bool = False) -> pd.Series:
    """
    Calculate the Money Flow Index (MFI) for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["High", "Low", "Close", "Volume"].
    period : int, optional
        The number of periods to use for the MFI calculation (default is 14).
    fill_na : bool, optional
        If True, earlier values (less than `period`) are calculated with
        progressively smaller windows instead of NaN. Default is False.

    Returns
    -------
    pd.Series
        A pandas Series containing the MFI values (0–100).
    """
    # Ensure required columns exist
    required_cols = {"High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Typical Price
    tp = (df["High"] + df["Low"] + df["Close"]) / 3

    # Raw Money Flow
    rmf = tp * df["Volume"]

    # Positive and Negative Money Flow
    tp_diff = tp.diff()
    positive_mf = rmf.where(tp_diff > 0, 0.0)
    negative_mf = rmf.where(tp_diff < 0, 0.0)

    # Determine rolling min periods based on fill_na
    min_periods = 1 if fill_na else period

    # Rolling sums
    positive_mf_sum = positive_mf.rolling(window=period, min_periods=min_periods).sum()
    negative_mf_sum = negative_mf.rolling(window=period, min_periods=min_periods).sum()

    # Avoid division by zero
    negative_mf_sum = negative_mf_sum.replace(0, 1e-10)

    # Money Flow Ratio and MFI
    mfr = positive_mf_sum / negative_mf_sum
    mfi = 100 - (100 / (1 + mfr))

    return mfi

def ta_obv(df: pd.DataFrame, fill_na: bool = False) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV) for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["Close", "Volume"].
    fill_na : bool, optional
        If True, forward-fills initial NaN values. Default is False.

    Returns
    -------
    pd.Series
        A pandas Series containing the OBV values.
    """
    required_cols = {"Close", "Volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Price change direction
    direction = df["Close"].diff()

    # Assign +Volume when price rises, -Volume when price falls
    obv_change = df["Volume"].where(direction > 0, -df["Volume"].where(direction < 0, 0))

    # Cumulative OBV
    obv = obv_change.cumsum()

    if fill_na:
        obv = obv.fillna(method="ffill")

    return obv

def ta_vroc(df: pd.DataFrame, period: int = 14, fill_na: bool = False) -> pd.Series:
    """
    Calculate the Volume Rate of Change (VROC) for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a "Volume" column.
    period : int, optional
        The number of periods for the rate-of-change calculation (default is 14).
    fill_na : bool, optional
        If True, earlier values (less than `period`) are calculated with smaller windows instead of NaN.

    Returns
    -------
    pd.Series
        A pandas Series containing the VROC values (percentage change).
    """
    if "Volume" not in df.columns:
        raise ValueError("DataFrame must contain 'Volume' column")

    # Determine rolling min periods
    min_periods = 1 if fill_na else period

    # Rate of change (%)
    vroc = (df["Volume"] - df["Volume"].shift(period)) / df["Volume"].shift(period) * 100

    if fill_na:
        vroc = vroc.fillna(0)

    return vroc


def ta_obv(df: pd.DataFrame, fill_na: bool = False) -> pd.Series:
    """
    Calculate the On-Balance Volume (OBV) for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["Close", "Volume"].
    fill_na : bool, optional
        If True, forward-fills initial NaN values. Default is False.

    Returns
    -------
    pd.Series
        A pandas Series containing the OBV values.
    """
    required_cols = {"Close", "Volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Price change direction
    direction = df["Close"].diff()

    # Assign +Volume when price rises, -Volume when price falls
    obv_change = df["Volume"].where(direction > 0, -df["Volume"].where(direction < 0, 0))

    # Cumulative OBV
    obv = obv_change.cumsum()

    if fill_na:
        obv = obv.fillna(method="ffill")

    return obv

def ta_chaikin_osc(
    df: pd.DataFrame,
    short_period: int = 3,
    long_period: int = 10,
    fill_na: bool = False
) -> pd.Series:
    """
    Calculate the Chaikin Oscillator for a given DataFrame.

    The Chaikin Oscillator measures the momentum of the Accumulation/Distribution Line (ADL)
    by subtracting a long-term EMA from a short-term EMA of the ADL.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["High", "Low", "Close", "Volume"].
    short_period : int, optional
        Short-term EMA period (default is 3).
    long_period : int, optional
        Long-term EMA period (default is 10).
    fill_na : bool, optional
        If True, earlier values are forward-filled instead of NaN. Default is False.

    Returns
    -------
    pd.Series
        A pandas Series containing the Chaikin Oscillator values.
    """
    required_cols = {"High", "Low", "Close", "Volume"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # --- Step 1: Money Flow Multiplier ---
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (
        df["High"] - df["Low"]
    )
    mfm = mfm.replace([float("inf"), -float("inf")], 0).fillna(0)

    # --- Step 2: Money Flow Volume ---
    mfv = mfm * df["Volume"]

    # --- Step 3: Accumulation/Distribution Line (ADL) ---
    adl = mfv.cumsum()

    # --- Step 4: Chaikin Oscillator ---
    adl_short_ema = adl.ewm(span=short_period, adjust=False).mean()
    adl_long_ema = adl.ewm(span=long_period, adjust=False).mean()

    chaikin_osc = adl_short_ema - adl_long_ema

    if fill_na:
        chaikin_osc = chaikin_osc.fillna(method="ffill")

    return chaikin_osc

def ta_rsi(df: pd.DataFrame, period: int = 14, fill_na: bool = False) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'Close' column.
    period : int, optional
        Lookback period for RSI (default is 14).
    fill_na : bool, optional
        If True, fills initial NaNs using smaller rolling windows.

    Returns
    -------
    pd.Series
        RSI values between 0 and 100.
    """
    if "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column")

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    min_periods = 1 if fill_na else period

    avg_gain = gain.rolling(window=period, min_periods=min_periods).mean()
    avg_loss = loss.rolling(window=period, min_periods=min_periods).mean()

    # Avoid division by zero
    avg_loss = avg_loss.replace(0, 1e-10)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ta_stoch(df: pd.DataFrame, period: int = 14, fill_na: bool = False) -> pd.Series:
    """
    Calculate the Stochastic Oscillator (%K).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["High", "Low", "Close"].
    period : int, optional
        Lookback period for %K (default is 14).
    fill_na : bool, optional
        If True, smaller windows are used early on.

    Returns
    -------
    pd.Series
        Stochastic Oscillator values between 0 and 100.
    """
    required_cols = {"High", "Low", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    min_periods = 1 if fill_na else period

    lowest_low = df["Low"].rolling(window=period, min_periods=min_periods).min()
    highest_high = df["High"].rolling(window=period, min_periods=min_periods).max()

    stoch_k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low)
    return stoch_k

def ta_williams_r(df: pd.DataFrame, period: int = 14, fill_na: bool = False) -> pd.Series:
    """
    Calculate the Williams %R oscillator.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["High", "Low", "Close"].
    period : int, optional
        Lookback period (default is 14).
    fill_na : bool, optional
        If True, uses progressively smaller windows initially.

    Returns
    -------
    pd.Series
        Williams %R values between -100 and 0.
    """
    required_cols = {"High", "Low", "Close"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    min_periods = 1 if fill_na else period

    highest_high = df["High"].rolling(window=period, min_periods=min_periods).max()
    lowest_low = df["Low"].rolling(window=period, min_periods=min_periods).min()

    will_r = -100 * (highest_high - df["Close"]) / (highest_high - lowest_low)
    return will_r

def ta_obv_osc(df: pd.DataFrame, short_period: int = 10, long_period: int = 20) -> pd.Series:
    """
    OBV Oscillator: Difference between short-term and long-term OBV EMAs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["Close", "Volume"].
    short_period : int
        Short-term EMA period.
    long_period : int
        Long-term EMA period.

    Returns
    -------
    pd.Series
        OBV oscillator values (positive = bullish, negative = bearish).
    """
    obv = ta_obv(df)
    short_ema = obv.ewm(span=short_period, adjust=False).mean()
    long_ema = obv.ewm(span=long_period, adjust=False).mean()
    return short_ema - long_ema


def ta_cmean_range(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the cumulative mean of (High - Low) for each row,
    equivalent to Pine Script's `ta.cum(high - low) / bar_index`.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ["High", "Low"].

    Returns
    -------
    pd.Series
        A pandas Series containing the cumulative mean range values.
    """
    # Ensure required columns exist
    required_cols = {"High", "Low"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_cols}")

    # Cumulative sum of (High - Low)
    cum_range = (df["High"] - df["Low"]).cumsum()

    # bar_index in Pine starts from 0, but to avoid division by zero we start from 1
    n = pd.Series(range(1, len(df) + 1), index=df.index)

    # Cumulative mean range
    cmean_range = cum_range / n

    return cmean_range

def ta_crossover(prev_a: float, curr_a: float, prev_b: float, curr_b: float) -> bool:
    """
    Check if a crossover occurred between two values over two time points.

    Parameters
    ----------
    prev_a : float
        Previous value of series A (e.g., Close price at i-1).
    curr_a : float
        Current value of series A (e.g., Close price at i).
    prev_b : float
        Previous value of series B (e.g., indicator line at i-1).
    curr_b : float
        Current value of series B (e.g., indicator line at i).

    Returns
    -------
    bool
        True if a crossover occurred (A crosses above B), False otherwise.
    """
    return (prev_a < prev_b) and (curr_a >= curr_b)

def ta_crossunder(prev_a: float, curr_a: float, prev_b: float, curr_b: float) -> bool:
    """
    Check if a crossunder occurred between two values over two time points.

    Parameters
    ----------
    prev_a : float
        Previous value of series A (e.g., Close price at i-1).
    curr_a : float
        Current value of series A (e.g., Close price at i).
    prev_b : float
        Previous value of series B (e.g., indicator line at i-1).
    curr_b : float
        Current value of series B (e.g., indicator line at i).

    Returns
    -------
    bool
        True if a crossunder occurred (A crosses below B), False otherwise.
    """
    return (prev_a > prev_b) and (curr_a <= curr_b)


def create_bull_concordant(df: pd.DataFrame) -> pd.Series:
    """
    Create a Series indicating bull_concordant condition.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ["Open", "High", "Low", "Close"]
    
    Returns:
        pd.Series: Boolean Series where True indicates a bull_concordant condition.
    """
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    close = df["Close"]

    bull_concordant = (high - np.maximum(close, open_)) > np.minimum(close, open_ - low)
    return bull_concordant

def create_bear_concordant(df: pd.DataFrame) -> pd.Series:
    """
    Create a Series indicating bear_concordant condition.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ["Open", "High", "Low", "Close"]
    
    Returns:
        pd.Series: Boolean Series where True indicates a bear_concordant condition.
    """
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    close = df["Close"]

    bear_concordant = (high - np.maximum(close, open_)) < np.minimum(close, open_ - low)
    return bear_concordant


def swings(df: pd.DataFrame, length: int):
    highs = df['High']
    lows = df['Low']
    
    # rolling highest and lowest (same as ta.highest / ta.lowest)
    upper = highs.rolling(length).max()
    lower = lows.rolling(length).min()
    
    os = np.zeros(len(df), dtype=int)  # 0 = up, 1 = down
    
    top = np.zeros(len(df))
    btm = np.zeros(len(df))
    
    for i in range(length, len(df)):
        prev_os = os[i-1]
        
        # emulate high[len] and low[len] as value 'length' bars ago
        high_len = highs.iloc[i - length]
        low_len = lows.iloc[i - length]
        upper_val = upper.iloc[i]
        lower_val = lower.iloc[i]
        
        if high_len > upper_val:
            os[i] = 0
        elif low_len < lower_val:
            os[i] = 1
        else:
            os[i] = prev_os
        
        if os[i] == 0 and prev_os != 0:
            top[i] = high_len
        elif os[i] == 1 and prev_os != 1:
            btm[i] = low_len
    
    return top, btm

# --- MODIFIED FUNCTION ---
# def swings(df: pd.DataFrame, length: int, config: dict):
#     highs = df['High']
#     lows = df['Low']
    
#     ATR_PERIOD = config.get("atr_period", 200)
    
#     # --- NEW: Get tolerance settings ---
#     tolerance_type = config.get("swing_tolerance_type", None)
#     tolerance_val = config.get("swing_tolerance_value", 0.0)
    
#     tolerance_series = pd.Series(0.0, index=df.index) # Default to 0
    
#     if tolerance_type == 'atr':
#         if 'atr' not in df.columns:
#             # Calculate ATR if it's missing (using a common default)
#             df['atr'] = ta_atr(df, period=ATR_PERIOD) 
#         tolerance_series = df['atr'] * tolerance_val
#     elif tolerance_type == 'pct':
#         # Use 'High' for top tolerance, 'Low' for bottom
#         pass # We'll calculate this in the loop
#     elif tolerance_type == 'points':
#         tolerance_series = pd.Series(tolerance_val, index=df.index)
#     # --- END NEW ---

#     # rolling highest and lowest
#     upper = highs.rolling(length).max()
#     lower = lows.rolling(length).min()
    
#     os = np.zeros(len(df), dtype=int)  # 0 = up, 1 = down
#     top = np.zeros(len(df))
#     btm = np.zeros(len(df))
    
#     for i in range(length, len(df)):
#         prev_os = os[i-1]
        
#         high_len = highs.iloc[i - length]
#         low_len = lows.iloc[i - length]
#         upper_val = upper.iloc[i]
#         lower_val = lower.iloc[i]
        
#         # --- NEW: Get tolerance for this specific bar ---
#         bar_tolerance_high = 0.0
#         bar_tolerance_low = 0.0
        
#         if tolerance_type == 'pct':
#             bar_tolerance_high = upper_val * tolerance_val
#             bar_tolerance_low = lower_val * tolerance_val
#         else:
#             # Use the pre-calculated ATR or points tolerance
#             # We use tolerance at [i-length] as that's the pivot bar
#             bar_tolerance_high = tolerance_series.iloc[i-length]
#             bar_tolerance_low = tolerance_series.iloc[i-length]
#         # --- END NEW ---
        
        
#         # --- MODIFIED: Apply tolerance to the check ---
#         if high_len > (upper_val + bar_tolerance_high):
#             os[i] = 0
#         elif low_len < (lower_val - bar_tolerance_low):
#             os[i] = 1
#         # --- END MODIFIED ---
#         else:
#             os[i] = prev_os
        
#         if os[i] == 0 and prev_os != 0:
#             top[i] = high_len
#         elif os[i] == 1 and prev_os != 1:
#             btm[i] = low_len
    
#     return top, btm

def ob_coord(
    df: pd.DataFrame,
    use_max: bool,
    loc: int,
    n: int,
    target_top: List,
    target_btm: List,
    target_left: List,
    target_type: List,
    config: Dict):
    """
    Python equivalent of the PineScript function `ob_coord`.
    - `df` must have columns: High, Low, timestamp, atr, cmean_range
    - All series are ordered oldest→newest.
    """

    high = df["High"].to_list()
    low = df["Low"].to_list()
    time = df["timestamp"].to_list()
    ob_filter = config.get("ob_filter")
    ob_threshold_mult = config.get("ob_threshold_mult", 2)

    min_val = float('inf')
    max_val = 0.0
    idx = 1

    # Choose threshold array
    if ob_filter == 'atr':
        ob_threshold = df["atr"].to_list()
    else:
        ob_threshold = df["cmean_range"].to_list()

    # The PineScript loop `for i = 1 to (n - loc) - 1`
    # iterates from 1 up to (n - loc) - 1 inclusive
    # PineScript high[i] means "i bars ago" → Python index = -(i+1)
    # But since we invert indexing, we go from oldest to newest
    for i in range(n - 1, loc, -1):
        if (high[i] - low[i]) < ob_threshold[i] * ob_threshold_mult:
            if use_max:
                if high[i] > max_val:
                    max_val = high[i]
                    min_val = low[i]
                    idx = i
            else:
                if low[i] < min_val:
                    min_val = low[i]
                    max_val = high[i]
                    idx = i

    # prepend to targets (unshift)
    target_top.insert(0, max_val)
    target_btm.insert(0, min_val)
    target_left.insert(0, time[idx])
    target_type.insert(0, -1 if use_max else 1)


def smart_money_concept(df: pd.DataFrame, config: Dict, verbose: int = 0):
    start_time = time.time()
    if verbose: print("Start SMC Algorithm")
    
    # Configuration
    SHOW_SWINGS: bool = config.get("show_swings")
    SHOW_IOB: bool = config.get("show_iob")
    SWING_LENGTH: int = config.get("swing_length")
    INTERNAL_SWING_LENGTH: int = config.get("internal_swing_length")
    OB_FILTER = config.get("ob_filter")
    DELETE_BROKEN_IOB = config.get("delete_broken_iob")
    ATR_PERIOD = config.get("atr_period", 200)
    
    
    # Dataframe important columns
    close = df["Close"].to_numpy()
    timestamps = df["timestamp"].to_numpy()
    
    
    # Global Variable
    trend, itrend = 0, 0
    
    top_y, top_x = 0.0, 0
    btm_y, btm_x = 0.0, 0
    
    itop_y, itop_x = 0.0, 0
    prev_itop_y = 0.0
    ibtm_y, ibtm_x = 0.0, 0
    prev_ibtm_y = 0.0
    
    top_cross, btm_cross = True, True
    itop_cross, ibtm_cross = True, True
    
    txt_top, txt_btm = "", ""
    
    # Alerts
    bull_ichoch_alert = False
    bull_ibos_alert   = False
    bear_ichoch_alert = False
    bear_ibos_alert = False
    
    bull_iob_break = False
    bear_iob_break = False
    
    # ORDER BLOCK ARRAYS
    iob_top: List[float] = []
    iob_btm: List[float] = []
    iob_left: List[int] = []
    iob_type: List[int] = []
    iob_loc = []
    iob_invalid: List[int] = []
    iob_created: List[int] = []
    
    if verbose: print(f"Finished initialization... ({time.time() - start_time}s)")
    start_time = time.time()
    
    
    # Function initialization
    n = len(df)
    if OB_FILTER == "atr":
        atr = ta_atr(df=df, period=ATR_PERIOD)
        df["atr"] = atr
    else:
        cmean_range = ta_cmean_range(df=df)
        df["cmean_range"] = cmean_range
    
    if verbose: print(f"Finished function init... ({time.time() - start_time}s)")
    start_time = time.time()
    
    # Bull and Bear Concordant
    if config.get("confluence_filter"):
        df["bull_concordant"] = create_bull_concordant(df=df)
        df["bear_concordant"] = create_bear_concordant(df=df)
    else:
        df["bull_concordant"] = True
        df["bear_concordant"] = True
    
    
    # Swings
    ## Non-internal Swings
    (top, btm) = swings(df=df, length=SWING_LENGTH) # , config=config
    df["top"] = top
    df["btm"] = btm
    ## Internal Swings
    (itop, ibtm) = swings(df=df, length=INTERNAL_SWING_LENGTH) # , config=config
    df["itop"] = itop
    df["ibtm"] = ibtm
    
    
    if verbose: print(f"Finished swings... ({time.time() - start_time}s)")
    start_time = time.time()
    
    top_btm_cross_time = 0.0
    itop_btm_cross_time = 0.0
    structure_time = 0.0
    deleting_time = 0.0
    
    # Main Pinescript Loop
    for i in range(len(df)):
        start_time = time.time()
        
        # Top&Btm Crossing
        ## Non-internal Top Crossing
        if df.at[i, "top"] != 0:
            top_cross = True
            txt_top = "HH" if df.at[i, "top"] > top_y else "LH"
            if SHOW_SWINGS:
                # TODO: implement show_swings
                pass
            #TODO: Extend recent top to last bar
            top_y = df.at[i, "top"]
            #TODO: Implement trail_up and trail_up_x
        ## Non-internal Bottom Crossing
        if df.at[i, "btm"] != 0:
            btm_cross = True
            txt_btm = "LL" if df.at[i, "btm"] < btm_y else "HL"
            if SHOW_SWINGS:
                # TODO: implement show_swings
                pass
            #TODO: Extend recent bot to last bar
            btm_y = df.at[i, "btm"]
            #TODO: Implement trail_dn and trail_dn_x
            
        top_btm_cross_time += (time.time() - start_time)
        start_time = time.time()
        
            
        ## Internal Top&Btm Crossing
        if df.at[i, "itop"] != 0:
            itop_cross = True
            prev_itop_y = itop_y
            itop_y = df.at[i, "itop"]
            itop_x = i - INTERNAL_SWING_LENGTH
        else:
            prev_itop_y = itop_y
            
        ## Internal Bottom Crossing
        if df.at[i, "ibtm"] != 0:
            ibtm_cross = True
            prev_ibtm_y = ibtm_y
            ibtm_y = df.at[i, "ibtm"]
            ibtm_x = i - INTERNAL_SWING_LENGTH
        else:
            prev_ibtm_y = ibtm_y
            
        itop_btm_cross_time += (time.time() - start_time)
        start_time = time.time()
            
        # Detect Internal Bullish Structure
        if i > 0:
            itop_y_close_crossover = ta_crossover(
                prev_a=df.at[i - 1, "Close"],
                curr_a=df.at[i, "Close"],
                prev_b=prev_itop_y,
                curr_b=itop_y
            )
        else:
            itop_y_close_crossover = False
        if itop_y_close_crossover and itop_cross and top_y != itop_y and df.at[i, "bull_concordant"]:
            choch: bool = None
            
            if itrend < 0:
                choch = True
                bull_ichoch_alert = True
            else:
                bull_ibos_alert = True
                
            txt = "CHoCH" if choch else "BOS"
            
            #TODO: implement 'if show_internals'
            
            itop_cross = False
            itrend = 1
            
            if SHOW_IOB:
                ob_coord(
                    df=df,
                    use_max=False,
                    loc=itop_x,
                    n=i,
                    target_top=iob_top,
                    target_btm=iob_btm,
                    target_left=iob_left,
                    target_type=iob_type,
                    config=config
                )
                iob_invalid.insert(0, 0)
                iob_created.insert(0, df.at[i, "timestamp"])
                
        # Detect Internal Bearish Structure
        if i > 0:
            ibtm_y_close_crossunder = ta_crossunder(
                prev_a=df.at[i - 1, "Close"],
                curr_a=df.at[i, "Close"],
                prev_b=prev_ibtm_y,
                curr_b=ibtm_y
            )
        else:
            ibtm_y_close_crossunder = False
        if ibtm_y_close_crossunder and ibtm_cross and btm_y != ibtm_y and df.at[i, "bear_concordant"]:
            choch: bool = None
            
            if itrend > 0:
                choch = True
                bear_ichoch_alert = True
            else:
                bear_ibos_alert = True
                
            txt = "CHoCH" if choch else "BOS"
            
            #TODO: implement 'if show_internals'
            
            ibtm_cross = False
            itrend = -1
            
            if SHOW_IOB:
                ob_coord(
                    df=df,
                    use_max=True,
                    loc=ibtm_x,
                    n=i,
                    target_top=iob_top,
                    target_btm=iob_btm,
                    target_left=iob_left,
                    target_type=iob_type,
                    config=config
                )
                iob_invalid.insert(0, 0)
                iob_created.insert(0, df.at[i, "timestamp"])
                # iob_loc.insert(0, df.at[i, "ibtm_x"])
        
        structure_time += (time.time() - start_time)
        start_time = time.time()
        
        # Order Blocks
        ## Delete internal order blocks box coordinates if top/bottom is broken
        # --- Fast delete broken IOBs ---
        if iob_type and DELETE_BROKEN_IOB:  # skip if empty
            close_val = df.at[i, "Close"]   
            iob_arr = np.array(iob_type)
            iob_top_arr = np.array(iob_top)
            iob_btm_arr = np.array(iob_btm)

            # Boolean mask of valid (unbroken) entries
            valid_mask = ~(((close_val < iob_btm_arr) & (iob_arr == 1)) |
                        ((close_val > iob_top_arr) & (iob_arr == -1)))

            # Keep only valid entries
            if not np.all(valid_mask):
                bull_iob_break = np.any((~valid_mask) & (iob_arr == 1))
                bear_iob_break = np.any((~valid_mask) & (iob_arr == -1))

                iob_type = iob_arr[valid_mask].tolist()
                iob_top = iob_top_arr[valid_mask].tolist()
                iob_btm = iob_btm_arr[valid_mask].tolist()
                iob_left = np.array(iob_left)[valid_mask].tolist()
        
        close_val = df.at[i, "Close"]
        ts = df.at[i, "timestamp"]
        
        # temporary NumPy arrays each iteration
        iob_type_arr = np.array(iob_type)
        iob_btm_arr = np.array(iob_btm)
        iob_top_arr = np.array(iob_top)
        iob_invalid_arr = np.array(iob_invalid, dtype=object)

        active = np.array([not x for x in iob_invalid_arr], dtype=bool)

        bull_mask = (close_val < iob_btm_arr) & (iob_type_arr == 1) & active
        iob_invalid_arr[bull_mask] = ts

        bear_mask = (close_val > iob_top_arr) & (iob_type_arr == -1) & active
        iob_invalid_arr[bear_mask] = ts

        # write back to list if needed
        iob_invalid = list(iob_invalid_arr)
            
        deleting_time += (time.time() - start_time)
        start_time = time.time()
            
    if verbose: print(f"Cross Time: {top_btm_cross_time}\nInternal Cross Time: {itop_btm_cross_time}\nStructure Time: {structure_time}\nDeleting Time: {deleting_time}")
                
    df_iob = pd.DataFrame({
        "iob_top": iob_top,
        "iob_btm": iob_btm,
        "iob_left": iob_left,
        "iob_type": iob_type,
        "iob_invalid": iob_invalid,
        "iob_created": iob_created,
    })
    df_iob.sort_values(by="iob_left", ascending=True, inplace=True)
    df_iob.reset_index(drop=True, inplace=True)
                
    return {
        "df": df,
        "df_iob": df_iob,
    }
