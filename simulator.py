# simulator.py
# All Monte Carlo simulation logic lives here

import numpy as np
import yfinance as yf


def get_stock_data(ticker, period="2y"):
    """
    Downloads stock data and calculates
    mean daily return and volatility.
    Returns mean, std, and closing prices.
    """
    data    = yf.download(ticker, period=period, progress=False)
    closes  = data['Close']
    returns = closes.pct_change().dropna()

    mean = returns.mean().iloc[0]
    std  = returns.std().iloc[0]

    return mean, std, closes


def run_simulation(mean, std, initial_amount, days, num_simulations=1000):
    """
    Runs Monte Carlo simulation.
    Returns array of final portfolio values.
    """
    final_values = np.zeros(num_simulations)

    for i in range(num_simulations):
        price = initial_amount

        for day in range(days):
            random_return = np.random.normal(mean, std)
            price         = price * (1 + random_return)

        final_values[i] = price

    return final_values


def calculate_metrics(final_values, initial_amount):
    """
    Takes simulation results and calculates
    all key risk metrics.
    """
    loss_count   = np.sum(final_values < initial_amount)
    loss_prob    = (loss_count / len(final_values)) * 100
    avg_final    = final_values.mean()
    best_case    = final_values.max()
    worst_case   = final_values.min()
    avg_return   = ((avg_final - initial_amount) / initial_amount) * 100
    median_final = np.median(final_values)

    return {
        "loss_probability" : round(loss_prob, 1),
        "avg_final"        : round(avg_final, 2),
        "best_case"        : round(best_case, 2),
        "worst_case"       : round(worst_case, 2),
        "avg_return_pct"   : round(avg_return, 2),
        "median_final"     : round(median_final, 2)
    }


def get_time_analysis(mean, std, initial_amount):
    """
    Runs simulation across 6 different time periods.
    Returns a dict of results for each period.
    """
    time_periods = {
        "1 Month"  : 21,
        "3 Months" : 63,
        "6 Months" : 126,
        "1 Year"   : 252,
        "2 Years"  : 504,
        "3 Years"  : 756
    }

    results = {}

    for period_name, days in time_periods.items():
        final_vals         = run_simulation(mean, std, initial_amount, days)
        metrics            = calculate_metrics(final_vals, initial_amount)
        results[period_name] = metrics

    return results