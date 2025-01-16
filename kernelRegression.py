from libraries import np
import pandas as pd
import pandas_ta as ta

# ====================================================
# ============= COSINE KERNEL REGRESSION =============
# ====================================================

# =====================
# Indicators functions
# =====================
def rsi(src, length):
    delta = src.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def stochastic(src, high, low, length):
    lowest_low = low.rolling(window=length).min()
    highest_high = high.rolling(window=length).max()
    return 100 * (src - lowest_low) / (highest_high - lowest_low)

def cci(src, length):
    sma = src.rolling(window=length).mean()
    mean_deviation = src.rolling(window=length).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    return (src - sma) / (0.015 * mean_deviation)

def chande_momentum(src, length, bar_index):
    """
    Calculate the Dynamic Chande Momentum Oscillator (CMO) 
    and rescale it using CMO_ReScale.
    """
    # Adjust the length to be within valid bounds
    length = min(length, bar_index + 1)
    
    # Calculate momentum
    momm = np.diff(src, prepend=src[0])
    
    # Positive and negative momentum
    m1 = np.where(momm >= 0, momm, 0.0)
    m2 = np.where(momm < 0, -momm, 0.0)
    
    # Sum of positive and negative momentum over the specified length
    sm1 = np.convolve(m1, np.ones(length, dtype=float), mode='valid')[:len(src)]
    sm2 = np.convolve(m2, np.ones(length, dtype=float), mode='valid')[:len(src)]
    
    # Calculate Chande Momentum Oscillator
    div = sm1 + sm2
    chandeMO = np.where(div != 0, 100 * (sm1 - sm2) / div, 0)
    
    return chandeMO * 1.15

def dynamic_bbpct(src, length, multi=2):
    """
    Computes the Dynamic Bollinger Band Percentage (BBPCT).

    Parameters:
        src (np.ndarray): Source array.
        length (int): Window length for SMA and standard deviation.
        multi (float): Multiplier for standard deviation.

    Returns:
        np.ndarray: BBPCT values.
    """
    basis = np.convolve(src, np.ones(length) / length, mode='valid')
    dev = multi * np.sqrt(
        np.convolve((src - np.convolve(src, np.ones(length) / length, mode='valid'))**2,
                    np.ones(length) / length, mode='valid')
                    )

    # Adjust lengths to match due to SMA and rolling operations
    offset = len(src) - len(basis)
    upper = basis + dev
    lower = basis - dev
    
    bbpct = (src[offset:] - lower) / (upper - lower)

    # return bbpct_rescale(bbpct)
    return (bbpct - 0.5) * 120

def dynamic_fisher(hl2, length, bar_index):
    """
    Calculate the Dynamic Fisher Transform.
    
    Parameters:
        hl2 (array): The array of (high + low) / 2 values.
        length (int): Lookback period.
        bar_index (int): Current bar index.
    
    Returns:
        array: Rescaled Fisher Transform values.
    """
    len_dynamic = min(length, 1 + bar_index)
    high_ = np.zeros_like(hl2)
    low_ = np.zeros_like(hl2)
    value1 = np.zeros_like(hl2)
    value2 = np.zeros_like(hl2)
    fish1 = np.zeros_like(hl2)

    for i in range(len(hl2)):
        current_len = min(len_dynamic, i + 1)
        high_[i] = np.max(hl2[max(0, i - current_len + 1):i + 1])
        low_[i] = np.min(hl2[max(0, i - current_len + 1):i + 1])
        
        if high_[i] != low_[i]:
            value1[i] = 0.66 * ((hl2[i] - low_[i]) / (high_[i] - low_[i]) - 0.5) + 0.67 * (value1[i - 1] if i > 0 else 0)
        else:
            value1[i] = 0.67 * (value1[i - 1] if i > 0 else 0)

        value2[i] = max(min(value1[i], 0.999), -0.999)

        if i > 0:
            fish1[i] = 0.5 * np.log((1 + value2[i]) / (1 - value2[i])) + 0.5 * fish1[i - 1]

    return fish1 * 30

def dynamic_vzo(hlc3, volume, length):
    """
    Computes the Volume Zone Oscillator (VZO) dynamically.

    Parameters:
        hlc3 (np.ndarray): Array of (high + low + close) / 3 values.
        volume (np.ndarray): Volume array.
        length (int): Lookback period for the VZO calculation.

    Returns:
        np.ndarray: Dynamic VZO values.
    """
    vp = dynamic_ema(np.sign(np.diff(hlc3, prepend=hlc3[0])) * volume, length / 3)
    tv = dynamic_ema(volume, length / 3)
    return (vp / tv) * 110
    # return vzo_rescale(vp, tv)

def dynamic_ema(src, length):
    """
    Computes the Exponential Moving Average (EMA) dynamically.

    Parameters:
        src (np.ndarray): Source array.
        length (float): Smoothing factor for EMA.

    Returns:
        np.ndarray: EMA values.
    """
    alpha = 2 / (length + 1)
    ema = np.zeros_like(src)
    ema[0] = src[0]
    for i in range(1, len(src)):
        ema[i] = alpha * src[i] + (1 - alpha) * ema[i - 1]
    return ema



# ============================
# Kernel and cosine functions
# ============================
def cosine_kernel(x, z):
    y = np.cos(z * x)
    return abs(y) if abs(x) <= np.pi / (2 * z) else 0.0

def kernel_regression(src, lookback, tuning):
    """
    Frequency tunable kernel regression function.

    Parameters:
        src (np.ndarray): Source input array (time series).
        lookback (int): Lookback period.
        tuning (float): Frequency tuner.

    Returns:
        float: Kernel regression value for the current index.
    """
    current_weight = 0.0
    total_weight = 0.0

    for i in range(min(lookback, len(src))):
        y = src[-(i + 1)]  # Get source value i steps back
        w = cosine_kernel(i / lookback, tuning)
        current_weight += y * w
        total_weight += w

    return current_weight / total_weight if total_weight != 0 else 0.0

# ===============================
# Multi cosine kernel regression
# ===============================
def multi_cosine_regression(src, lookback, steps):
    """
    Cosine composite kernel regression function.

    Parameters:
        src (np.ndarray): Source input array (time series).
        lookback (int): Lookback period.
        steps (int): Number of cosine functions for composite regression.

    Returns:
        float: Composite kernel regression value for the current index.
    """
    regression = 0.0

    for i in range(1, min(steps, len(src))):
        regression += kernel_regression(src, lookback, i)

    return regression / steps if steps != 0 else 0.0

# =================
# Main calculation
# =================
def calculate_indicator(src, high, low, close, lookback, tuning, variant, rsiLength, stochasticLength, cciLength):
    print("Initiating Cosine kernel regrtession")
    # Example for calculating RSI, you can add other indicators like Stochastic, CCI, etc.
    rsi_value = rsi(close, rsiLength)
    stochastic_value = stochastic(close, high, low, stochasticLength)
    cci_value = cci(close, cciLength)
    print("Calculated RSI")   
    # You can combine multiple indicators like in the original script
    active_indicators = [rsi_value,stochastic_value,cci_value]  # Add other indicators as needed
    
    # Calculate the regression output based on the selected variant
    if variant == 'Tuneable':
        return kernel_regression(active_indicators, lookback, tuning)
    elif variant == 'Stepped':
        return multi_cosine_regression(active_indicators, lookback, tuning)
    else:
        return np.nan
