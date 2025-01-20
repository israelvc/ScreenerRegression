from libraries import np
# import pandas as pd
# import pandas_ta as ta

# ====================================================
# ============= COSINE KERNEL REGRESSION =============
# ====================================================

# ============================
# Lagged Indicators functions
# ============================
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
    momm = np.diff(src, prepend=src.iloc[0])
    
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
    # Compute the basis (SMA)
    basis = np.convolve(src, np.ones(length) / length, mode='valid')
    
    # Compute rolling standard deviation
    rolling_mean = np.convolve(src, np.ones(length) / length, mode='valid')
    rolling_var = np.convolve((src[length - 1:] - rolling_mean)**2, np.ones(length) / length, mode='valid')
    dev = multi * np.sqrt(rolling_var)

    # Trim 'basis' and 'dev' to the same length
    min_length = min(len(basis), len(dev))
    basis = basis[:min_length]
    dev = dev[:min_length]

    # Calculate upper and lower bands
    upper = basis + dev
    lower = basis - dev

    # Align src for BBPCT calculation
    aligned_src = src[length - 1:][:min_length]
    bbpct = (aligned_src - lower) / (upper - lower)

    # Scale BBPCT to match expected output
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

def alma(series, window_size, offset, sigma):
    """
    Computes the Arnaud Legoux Moving Average (ALMA).

    Parameters:
        series (np.ndarray): Input data series (e.g., prices or indicators).
        window_size (int): Window size for the ALMA calculation.
        offset (float): ALMA offset (usually between 0 and 1).
        sigma (float): Sigma for the Gaussian weighting function.

    Returns:
        np.ndarray: ALMA values.
    """
    m = offset * (window_size - 1)
    s = window_size / sigma
    weights = np.exp(-((np.arange(window_size) - m) ** 2) / (2 * s ** 2))
    weights /= weights.sum()

    # Apply the weights using a rolling window
    alma_values = np.convolve(series, weights[::-1], mode='valid')
    # Padding for alignment with original series length
    alma_padded = np.concatenate([np.full(window_size - 1, np.nan), alma_values])

    return alma_padded

def dynamic_alma(series, base_window_size, offset, sigma, atr_period, atr_threshold, high, low, close):
    """
    Computes the Dynamic ALMA with a volatility-based window size adjustment.

    Parameters:
        series (np.ndarray): Input data series.
        base_window_size (int): Base window size for ALMA.
        offset (float): ALMA offset.
        sigma (float): Sigma for Gaussian weighting.
        atr_period (int): Period for ATR calculation.
        atr_threshold (float): Multiplier for ATR volatility filtering.
        high (np.ndarray): High prices.
        low (np.ndarray): Low prices.
        close (np.ndarray): Close prices.

    Returns:
        np.ndarray: Dynamically adjusted ALMA values.
    """
    # Calculate ATR
    tr = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr = tr.rolling(window=atr_period).mean()
    sma_atr = atr.rolling(window=atr_period).mean()

    # Volatility factor
    volatility_factor = atr / sma_atr
    volatility_factor[atr.isna() | sma_atr.isna()] = 1.0

    # Adjusted window size
    adjusted_window_size = (base_window_size * (1 + (volatility_factor - 1) * 0.5)).round().clip(lower=1).astype(int)

    # Calculate dynamic ALMA
    dynamic_alma_values = np.full(len(series), np.nan)
    for i in range(len(series)):
        if i >= adjusted_window_size.iloc[i] - 1:
            window = adjusted_window_size.iloc[i]
            dynamic_alma_values[i] = alma(series[i - window + 1:i + 1], window, offset, sigma)[-1]

    return dynamic_alma_values



# ========================================
# Kernel and cosine functions calculation
# ========================================
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
def prepareDataForCosineKernelRegression(tickerData, rsiLength, stochasticLength, cciLength,
                                         cmoLength, bbpctLength, fisherTransformLength, vzoLength):
    print('Preparing dataset for Cosine Kernel regression')

    active_indicators = 7
    hlc3Data = (tickerData['High'] + tickerData['Low'] + tickerData['Close']) / 3    
    hl2Data = (tickerData['High'] + tickerData['Low']) / 2
    cmo_values = []
    fisher_values = []
    
    for i in range(len(tickerData)):
        if i >= cmoLength - 1:
            cmo = chande_momentum(tickerData['Close'].iloc[:i+1], cmoLength, i)
            cmo_values.append(cmo[-1])
        else:
            cmo_values.append(np.nan)

    for i in range(len(tickerData)):
        if i >= fisherTransformLength - 1:
            fisher = dynamic_fisher(hl2Data.iloc[:i+1].values, fisherTransformLength, i)
            fisher_values.append(fisher[-1])
        else:
            fisher_values.append(np.nan)

    tickerData['val_FISH'] = fisher_values
    tickerData['val_CMO'] = cmo_values
    tickerData['val_RSI'] = rsi(tickerData['Close'], rsiLength)
    tickerData['val_STOCH'] = stochastic(tickerData['Close'], tickerData['High'], tickerData['Low'], stochasticLength)
    tickerData['val_BBPCT'] = dynamic_bbpct(tickerData['Close'], length = bbpctLength, multi = 2)
    tickerData['val_CCI'] = cci(tickerData['Close'], cciLength)
    tickerData['val_VZO'] = dynamic_vzo(hlc3Data.values, tickerData['Volume'].values, vzoLength)

    
    tickerData['Average_Indicators'] = (
        tickerData[['val_FISH', 'val_CMO', 'val_RSI', 'val_STOCH', 'val_BBPCT', 'val_CCI', 'val_VZO']]
        .fillna(0)
        .sum(axis=1) / active_indicators
    )

    # ALMA smoothing
    tickerData['Dynamic_ALMA'] = dynamic_alma(
        series = tickerData['Average_Indicators'],
        base_window_size = 9,
        offset = 0,
        sigma = 6,
        atr_period = 14,
        atr_threshold = 1.0,
        high = tickerData['High'],
        low = tickerData['Low'],
        close = tickerData['Close']
    )

    return tickerData

def calculate_indicator(dataSource, lookbackR = 144, tuning = 15, variant = "Tuneable"):   
    value = dataSource['Average_Indicators'].values
    
    if variant == "Tuneable":
        out = [kernel_regression(value[:i + 1], lookbackR, tuning) for i in range(len(value))]
    elif variant == "Stepped":
        out = [multi_cosine_regression(value[:i + 1], lookbackR, tuning) for i in range(len(value))]
    else:
        raise ValueError("Invalid variant. Use 'Tuneable' or 'Stepped'.")

    # Calculate `out2` with adjusted tuning
    adjusted_tuning = round(tuning / 5)
    if variant == "Tuneable":
        out2 = [kernel_regression(value[:i + 1], lookbackR, adjusted_tuning) for i in range(len(value))]
    elif variant == "Stepped":
        out2 = [multi_cosine_regression(value[:i + 1], lookbackR, adjusted_tuning) for i in range(len(value))]
    
    # Trend and conditions
    out = np.array(out)
    out2 = np.array(out2)

    # Fast trend conditions
    fastTrend_up = (out > np.roll(out, 1)) & ~(np.roll(out, 1) > np.roll(out, 2))
    fastTrend_dn = (out < np.roll(out, 1)) & ~(np.roll(out, 1) < np.roll(out, 2))
    fastTrend = fastTrend_up | fastTrend_dn

    # Slow trend conditions
    slowTrend_up = (out2 > 0) & ~(np.roll(out2, 1) > 0)
    slowTrend_dn = (out2 < 0) & ~(np.roll(out2, 1) < 0)
    slowTrend = slowTrend_up | slowTrend_dn

    # Overbought and oversold conditions
    overbought = (out > 50) & ~(np.roll(out, 1) > 50)
    oversold = (out < -50) & ~(np.roll(out, 1) < -50)
    
    dataSource['out'] = out
    dataSource['out2'] = out2
    dataSource['fastTrend_up'] = fastTrend_up
    dataSource['fastTrend_dn'] = fastTrend_dn
    dataSource['fastTrend'] = fastTrend
    dataSource['slowTrend_up'] = slowTrend_up
    dataSource['slowTrend_dn'] = slowTrend_dn
    dataSource['slowTrend'] = slowTrend
    dataSource['overbought'] = overbought
    dataSource['oversold'] = oversold

    boolean_columns = [
        'fastTrend_up', 'fastTrend_dn', 'fastTrend',
        'slowTrend_up', 'slowTrend_dn', 'slowTrend',
        'overbought', 'oversold'
    ]

    dataSource[boolean_columns] = dataSource[boolean_columns].astype(int)

    return dataSource #out, out2, fastTrend, slowTrend, overbought, oversold
