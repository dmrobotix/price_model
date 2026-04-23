"""
price.py

This module handles loading and forecasting of historical market price data for Bitcoin.
It supports multiple forecasting models—linear, power law, and logistic—to allow flexibility
in price prediction based on historical trends. The CSV file is expected to have headers:
"Time" and "Price (USD)", where Time is formatted as "YYYY-MM-DD HH:MM:SS".
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

def load_market_prices(file_path):
    """
    Load market price data from a CSV file.
    Assumes "Time" is UTC-naive in the CSV.
    """
    df = pd.read_csv(file_path)
    df['Time'] = pd.to_datetime(df['Time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    df = df.dropna(subset=['Time'])
    df = df.sort_values('Time')
    return df

def build_price_lookup(df):
    """
    Build a fast, streaming price lookup function.
    Accepts integer timestamps (seconds) or datetimes.
    """
    # Work with a sorted copy
    df_sorted = df.sort_values("Time").reset_index(drop=True)

    # Convert "Time" column (datetime) to integer seconds (UTC-naive assumption)
    # view("int64") gives nanoseconds. // 10^9 gives seconds.
    times_sec = (df_sorted["Time"].values.astype('datetime64[ns]').view('int64') // 10**9)
    prices = df_sorted["Price (USD)"].to_numpy(dtype=float)

    n = len(times_sec)
    idx = 0 
    first_t = times_sec[0]
    last_t = times_sec[-1]

    def lookup(target):
        nonlocal idx
        
        # Handle input types: int/float (seconds) or datetime
        t = 0
        if isinstance(target, (int, float, np.integer, np.floating)):
            t = int(target)
        elif isinstance(target, (datetime, pd.Timestamp)):
            # Convert datetime to seconds. Assume UTC-naive if no tzinfo.
            if target.tzinfo is None:
                # Force UTC then timestamp
                t = int(target.replace(tzinfo=timezone.utc).timestamp())
            else:
                t = int(target.timestamp())
        else:
            raise TypeError(f"Invalid type for price lookup: {type(target)}")

        # Clamping
        if t <= first_t:
            return float(prices[0])
        if t >= last_t:
            # If we are past the last known price, return the last one
            return float(prices[-1])

        # Advance idx (streaming optimization)
        # While the *next* time point is less than or equal to current target time, move forward
        while idx + 1 < n and times_sec[idx + 1] <= t:
            idx += 1

        return float(prices[idx])

    return lookup
    
def get_price_for_date(df, target_date, _cache={}):
    """
    Fast lookup of the market price at or immediately before target_date.
    Uses a cached NumPy array and binary search instead of filtering each time.
    """
    times = _cache.get("times")
    prices = _cache.get("prices")

    if times is None:
        # One-time setup: sort and cache arrays
        df_sorted = df.sort_values("Time").reset_index(drop=True)
        times = df_sorted["Time"].to_numpy()
        prices = df_sorted["Price (USD)"].to_numpy()
        _cache["times"] = times
        _cache["prices"] = prices

    # Convert to numpy datetime64 for searchsorted
    target_ts = np.datetime64(target_date)

    # Find index of last time <= target_date
    idx = times.searchsorted(target_ts, side="right") - 1

    if idx < 0:
        # target_date is earlier than the first data point
        return float(prices[0])

    return float(prices[idx])

def linear_forecast(df, target_date):
    """
    Forecast the market price at target_date using a linear regression model.
    
    Parameters:
        df (DataFrame): Historical price data.
        target_date (datetime): The date for which to forecast the price.
        
    Returns:
        float: The forecasted price.
    """
    df = df.copy()
    # Convert time to Unix timestamp.
    df['timestamp'] = df['Time'].astype('int64') // 10**9
    X = df['timestamp'].values.reshape(-1, 1)
    y = df['Price (USD)'].values  # Using raw price; alternatively, log-transform if needed.
    model = LinearRegression().fit(X, y)
    target_ts = np.array([[int(target_date.timestamp())]])
    return model.predict(target_ts)[0]

def powerlaw_model(t, a, b):
    """
    Power law model: Price = a * t^b.
    """
    return a * np.power(t, b)

def powerlaw_forecast(df, target_date):
    """
    Forecast the market price using a power law model.
    
    Parameters:
        df (DataFrame): Historical price data.
        target_date (datetime): The date for which to forecast the price.
        
    Returns:
        float: The forecasted price.
    """
    df = df.copy()
    t0 = df['Time'].min()
    # Add a small epsilon (e.g., 1 second) to avoid zero time difference.
    epsilon = 1.0
    df['t'] = (df['Time'] - t0).dt.total_seconds() + epsilon
    X = df['t'].values
    y = df['Price (USD)'].values
    params, _ = curve_fit(powerlaw_model, X, y, maxfev=10000)
    target_t = (target_date - t0).total_seconds() + epsilon
    return powerlaw_model(target_t, *params)


def logistic_model(t, L, k, t0):
    """
    Logistic model: Price = L / (1 + exp(-k*(t - t0))).
    """
    return L / (1 + np.exp(-k*(t - t0)))

def logistic_forecast(df, target_date):
    """
    Forecast the market price using a logistic model.
    
    Parameters:
        df (DataFrame): Historical price data.
        target_date (datetime): The date for which to forecast the price.
        
    Returns:
        float: The forecasted price.
    """
    df = df.copy()
    t0 = df['Time'].min()
    df['t'] = (df['Time'] - t0).dt.total_seconds()
    X = df['t'].values
    y = df['Price (USD)'].values
    # Provide an initial guess: L is roughly twice the max observed price, k is small, t0 near median.
    initial_guess = [max(y)*2, 1e-4, np.median(X)]
    params, _ = curve_fit(logistic_model, X, y, p0=initial_guess, maxfev=10000)
    target_t = (target_date - t0).total_seconds()
    return logistic_model(target_t, *params)

def forecast_price(df, target_date, model_type='linear', forecast_target_date=None, forecast_target_price=None):
    """
    Forecast the price at target_date using the specified model type.
    
    Parameters:
        df (DataFrame): Historical price data.
        target_date (datetime): The date for which to forecast the price.
        model_type (str): One of 'linear', 'powerlaw', or 'logistic'.
        forecast_target_date (datetime, optional): A date to anchor the forecast.
        forecast_target_price (float, optional): The expected price at forecast_target_date.
        
    Returns:
        float: The forecasted (and optionally adjusted) price.
    """
    if model_type == 'linear':
        price_pred = linear_forecast(df, target_date)
    elif model_type == 'powerlaw':
        price_pred = powerlaw_forecast(df, target_date)
    elif model_type == 'logistic':
        price_pred = logistic_forecast(df, target_date)
    else:
        raise ValueError("Unknown model type. Choose 'linear', 'powerlaw', or 'logistic'.")
    
    if forecast_target_date is not None and forecast_target_price is not None:
        # Forecast at the anchor date using the same model.
        if model_type == 'linear':
            anchor_pred = linear_forecast(df, forecast_target_date)
        elif model_type == 'powerlaw':
            anchor_pred = powerlaw_forecast(df, forecast_target_date)
        elif model_type == 'logistic':
            anchor_pred = logistic_forecast(df, forecast_target_date)
        adjustment = forecast_target_price / anchor_pred
        price_pred *= adjustment
    
    return price_pred

def build_price_forecaster(df: pd.DataFrame, model_type: str = "powerlaw", anchor_date: datetime = None, anchor_price: float = None):
    """
    Fit the chosen model once on df (with columns 'Time' and 'Price (USD)'),
    and return a function f(date: datetime) -> float that predicts price in O(1).
    """
    print("Building price forecaster...")
    # make a local copy & prep
    prices = df.copy()
    prices = prices.sort_values('Time').dropna(subset=['Time','Price (USD)'])

    if model_type in ("fixed", "constant"):
        # Use anchor_price as the constant forecast level.
        # If not provided, fall back to the last historical observed price.
        level = float(anchor_price) if anchor_price is not None else float(prices["Price (USD)"].iloc[-1])

        def forecaster(dt: datetime) -> float:
            return level

        print("Price forecaster finished.")
        return forecaster
    
    elif model_type == "linear":
        # timestamps → seconds
        X = (prices['Time'].astype('int64') // 10**9).values.reshape(-1,1)
        y = prices['Price (USD)'].values
        linreg = LinearRegression().fit(X, y)
        
        def forecaster(dt):
            t = np.array([[int(dt.timestamp())]])
            p = linreg.predict(t)[0]
            if anchor_date and anchor_price:
                a_ts = int(anchor_date.timestamp())
                a_pred = linreg.predict([[a_ts]])[0]
                p *= (anchor_price / a_pred)
            return float(p)
    
    elif model_type == "powerlaw":
        # 1) reference zero‐time
        t0 = prices['Time'].min()
        # 2) pull last real data point
        last_row  = prices.iloc[-1]
        last_date = last_row['Time']
        last_price= last_row['Price (USD)']
        
        # we require both anchors to do the two‐point fit
        if anchor_date is not None and anchor_price is not None:
            # compute normalized seconds
            def secs(dt): return (dt - t0).total_seconds() + 1.0
            t_last   = secs(last_date)
            t_anchor = secs(anchor_date)
            # exponent that exactly passes through both anchors
            b = np.log(anchor_price/last_price) / np.log(t_anchor/t_last)
            # coefficient so forecast(last) == last_price
            A = last_price / (t_last**b)

            def forecaster(dt):
                dt_sec = secs(dt)
                return float(A * (dt_sec**b))
        
        else:
            # fallback: ordinary fit to historical curve only
            t = (prices['Time'] - t0).dt.total_seconds() + 1.0
            y = prices['Price (USD)'].values
            (a, b), _ = curve_fit(lambda t,a,b: a * t**b, t.values, y, maxfev=10000)
    
            def forecaster(dt):
                dt_sec = (dt - t0).total_seconds() + 1.0
                return float(a * (dt_sec**b))
    
    elif model_type == "logistic":
        t0 = prices['Time'].min()
        T = (prices['Time'] - t0).dt.total_seconds()
        y = prices['Price (USD)'].values
        init = [max(y)*2, 1e-4, np.median(T)]
        (L,k,t0p), _ = curve_fit(lambda t,L,k,t0p: L/(1+np.exp(-k*(t-t0p))),
                                 T.values, y, p0=init, maxfev=10000)
        
        def forecaster(dt):
            dt_sec = (dt - t0).total_seconds()
            p = L / (1 + np.exp(-k*(dt_sec - t0p)))
            if anchor_date and anchor_price:
                ap = L / (1 + np.exp(-k*(((anchor_date - t0).total_seconds()) - t0p)))
                p *= (anchor_price / ap)
            return float(p)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    print("Price forecaster finished.")
    return forecaster

