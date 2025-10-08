# risk_analyzer.py
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
from prophet import Prophet
from pypfopt import EfficientFrontier, risk_models, expected_returns

# -------------------------------
# Step 1: Fetch price data
# -------------------------------
def fetch_data(tickers, start="2020-01-01", end="2023-01-01"):
    """
    Fetch adjusted closing prices for multiple tickers.
    Returns a DataFrame where each column is a ticker.
    """
    # Download data grouped by ticker
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)

    # Prepare a DataFrame with only adjusted closing prices
    price_data = pd.DataFrame()

    for ticker in tickers:
        ticker_upper = ticker.upper()
        if ticker_upper in data.columns:
            # Some versions of yfinance may return single-level
            price_data[ticker_upper] = data[ticker_upper]['Close']
        else:
            # For MultiIndex
            price_data[ticker_upper] = data[ticker_upper]['Close']

    return price_data

# -------------------------------
# Step 2: Calculate daily returns
# -------------------------------
def calculate_returns(price_data):
    """
    Calculate daily returns from price data
    """
    returns = price_data.pct_change().dropna()
    return returns

# -------------------------------
# Step 3: Portfolio return calculation
# -------------------------------
def portfolio_return(returns, weights, tickers):
    """
    Calculate portfolio daily returns given individual asset returns and weights.
    Ensures weights match tickers and selects only relevant columns.
    """
    # Ensure weights match number of tickers
    if len(weights) != len(tickers):
        raise ValueError("Number of weights must match number of tickers")

    # Select only the tickers we have weights for
    returns = returns[tickers]

    # Normalize weights to sum to 1
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    return returns.dot(weights)

# -------------------------------
# Step 4: Risk Metrics
# -------------------------------
def annualized_return(portfolio_returns, trading_days=252):
    return np.mean(portfolio_returns) * trading_days

def annualized_volatility(portfolio_returns, trading_days=252):
    return np.std(portfolio_returns) * np.sqrt(trading_days)

def sharpe_ratio(portfolio_returns, risk_free_rate=0.04, trading_days=252):
    ann_ret = annualized_return(portfolio_returns, trading_days)
    ann_vol = annualized_volatility(portfolio_returns, trading_days)
    return (ann_ret - risk_free_rate) / ann_vol

def max_drawdown(portfolio_returns):
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def historical_var(portfolio_returns, alpha=5):
    """
    Historical Value at Risk
    """
    return -np.percentile(portfolio_returns, alpha)

def historical_cvar(portfolio_returns, alpha=5):
    """
    Conditional VaR: mean of worst losses beyond alpha%
    """
    threshold = np.percentile(portfolio_returns, alpha)
    tail = portfolio_returns[portfolio_returns <= threshold]
    return -tail.mean()

# -------------------------------
# Step 5: Demo run
# -------------------------------
if __name__ == "__main__":
    # Example portfolio
    tickers = ["AAPL", "MSFT", "TSLA"]  # tickers
    weights = [0.4, 0.4, 0.2]           # must match tickers length

    # Fetch price data
    print("Fetching price data...")
    data = fetch_data(tickers)
    print("Columns in data:", data.columns.tolist())
    print(data.tail())

    # Compute daily returns
    returns = calculate_returns(data)
    print("\nSample daily returns:")
    print(returns.tail())

    # Compute portfolio returns
    port_ret = portfolio_return(returns, weights, tickers)

    # Compute metrics
    print("\nPortfolio Risk Metrics:")
    print(f"Annualized Return: {annualized_return(port_ret):.2%}")
    print(f"Annualized Volatility: {annualized_volatility(port_ret):.2%}")
    print(f"Sharpe Ratio (RF=4%): {sharpe_ratio(port_ret):.2f}")
    print(f"Max Drawdown: {max_drawdown(port_ret):.2%}")
    print(f"95% Historical VaR: {historical_var(port_ret, alpha=5):.2%}")
    print(f"95% Historical CVaR: {historical_cvar(port_ret, alpha=5):.2%}")


# Add this function to risk_analyzer.py

def calculate_var_cvar(returns, weights, confidence_level=0.95):
    """
    Calculates Value at Risk (VaR) and Conditional Value at Risk (CVaR)
    using the historical simulation method.
    """
    # Calculate daily portfolio returns
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Calculate VaR: The worst loss that won't be exceeded with a given confidence level
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    
    # Calculate CVaR: The average loss on days where the loss exceeded the VaR
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    
    return var, cvar

# --- Feature: Monte Carlo Simulation ---
def run_monte_carlo_simulation(portfolio_returns, n_simulations=1000, T=252):
    """
    Runs a Monte Carlo simulation for future portfolio performance.
    Takes the portfolio's daily returns Series as input.
    """
    # Calculate daily drift (average return) and volatility
    mu = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    
    # Create a simulation dataframe
    simulation_df = pd.DataFrame()
    
    for i in range(n_simulations):
        # Generate random daily returns
        daily_returns = np.random.normal(mu, sigma, T)
        
        # Simulate the price path starting from $1
        price_path = np.cumprod(1 + daily_returns)
        simulation_df[f'Simulation {i+1}'] = price_path
        
    return simulation_df



# --- Feature: GARCH Volatility Forecasting ---
def forecast_garch_volatility(portfolio_returns, horizon=30):
    """
    Forecasts future volatility using a GARCH(1,1) model.
    Takes the portfolio's daily returns Series as input.
    """
    # Scale returns by 100 for model stability
    scaled_returns = portfolio_returns * 100
    
    # Fit the GARCH(1,1) model
    model = arch_model(scaled_returns, vol='Garch', p=1, q=1, rescale=False)
    model_fit = model.fit(disp='off')
    
    # Forecast variance
    forecast = model_fit.forecast(horizon=horizon)
    
    # Get the last day's forecast, take the square root for volatility, and annualize it
    # Then scale back by dividing by 100
    forecasted_vol = np.sqrt(forecast.variance.iloc[-1].values[0]) * np.sqrt(252) / 100
    
    return forecasted_vol


# --- Feature: Prophet Value Forecasting ---
def forecast_prophet_value(price_data, weights, horizon=365):
    """
    Forecasts the future value of the portfolio using Prophet.
    Takes the historical price data and portfolio weights as input.
    """
    # Calculate historical portfolio value, starting at $1
    normalized_prices = price_data / price_data.iloc[0]
    portfolio_value = (normalized_prices * weights).sum(axis=1)
    
    # Prepare data for Prophet: needs 'ds' (datestamp) and 'y' (value) columns
    df_prophet = portfolio_value.reset_index()
    df_prophet.columns = ['ds', 'y']
    
    # Initialize and fit the Prophet model
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(df_prophet)
    
    # Create a dataframe for future dates
    future = model.make_future_dataframe(periods=horizon)
    
    # Generate the forecast
    forecast = model.predict(future)
    
    return forecast




# --- Feature: Portfolio Optimization ---
def optimize_portfolio(price_data):
    """
    Calculates the optimal portfolio weights to maximize the Sharpe ratio.
    Takes the DataFrame of prices as input.
    """
    # Calculate expected returns and the annualized sample covariance matrix of asset returns
    mu = expected_returns.mean_historical_return(price_data)
    S = risk_models.sample_cov(price_data)
    
    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    
    # Clean the raw weights (rounding small values and ensuring they sum to 1)
    cleaned_weights = ef.clean_weights()
    
    # Get the expected performance of the optimal portfolio
    performance = ef.portfolio_performance(verbose=False)
    
    return cleaned_weights, performance