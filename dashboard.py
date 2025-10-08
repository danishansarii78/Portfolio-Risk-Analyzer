
# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px 
from datetime import datetime
import matplotlib.pyplot as plt  # <-- NEW ADDITION: Import Matplotlib

# Import all the necessary functions from your risk_analyzer.py file
from risk_analyzer import (
    fetch_data,
    calculate_returns,
    portfolio_return,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    run_monte_carlo_simulation,
    calculate_var_cvar,
    forecast_garch_volatility,  # <-- ADD THIS
    forecast_prophet_value,
    optimize_portfolio 
) 

# --- Page Configuration ---
st.set_page_config(
    page_title="Portfolio Risk Analyzer",
    page_icon="📊",
    layout="wide"
)

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.title("Portfolio Inputs")
    
    tickers_input = st.text_input(
        "Enter stock tickers (comma-separated)", 
        "AAPL, MSFT, GOOGL, TSLA"
    )
    
    weights_input = st.text_input(
        "Enter portfolio weights (comma-separated)", 
        "0.25, 0.25, 0.25, 0.25"
    )

    st.markdown("---") # Visual separator

    start_date = st.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.date_input("End Date", datetime.now().date())
    
    st.markdown("---") # Visual separator

    analyze_button = st.button("Analyze Portfolio", type="primary")


# --- Main Dashboard Area ---
st.title("📊 AI-Driven Portfolio Risk Analyzer")
st.caption(f"Analysis performed from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# --- Run Analysis on Button Click ---
if analyze_button:
    # 1. Process and Validate Inputs
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    try:
        weights = np.array([float(w.strip()) for w in weights_input.split(",")])
        
        if len(tickers) != len(weights):
            st.error("Error: The number of tickers and weights must be the same.")
        elif not np.isclose(np.sum(weights), 1.0):
            st.warning("Warning: The sum of weights is not 1.0. They will be normalized.")
            weights = weights / np.sum(weights) # Normalize weights to sum to 1
        elif start_date >= end_date:
            st.error("Error: Start date must be before end date.")
        else:
            # 2. Perform Analysis
            with st.spinner("Fetching data and calculating metrics..."):
                # Fetch price data
                price_data = fetch_data(tickers, start=str(start_date), end=str(end_date))
                
                if price_data.empty or price_data.isnull().all().any():
                     st.error("Could not fetch valid data for one or more tickers. Please check the symbols and date range.")
                else:
                    # Calculate returns
                    returns = calculate_returns(price_data)
                    
                    # Calculate portfolio return series for charting
                    port_ret_series = portfolio_return(returns, weights, tickers)
                    
                    # Calculate key metrics
                    p_return = annualized_return(port_ret_series)
                    p_volatility = annualized_volatility(port_ret_series)
                    p_sharpe = sharpe_ratio(port_ret_series)
                    
                    # Calculate VaR and CVaR using the dedicated function
                    var, cvar = calculate_var_cvar(returns, weights)

                    # 3. Display Results
                    st.header("📈 Key Performance Indicators")
                   
                    garch_forecast = forecast_garch_volatility(port_ret_series)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Annualized Return", f"{p_return:.2%}")
                    col2.metric("Annualized Volatility (Risk)", f"{p_volatility:.2%}")
                    col3.metric("30-Day Forecast Volatility (GARCH)", f"{garch_forecast:.2%}")
                    col4.metric("Sharpe Ratio", f"{p_sharpe:.2f}")

                    st.divider()
# --- Portfolio Optimization Section ---
                    st.divider()
                    st.header("⚖️ AI-Powered Portfolio Optimization")
                    st.markdown("This model calculates the optimal weights for your selected stocks to maximize the risk-adjusted return (Sharpe Ratio).")

# Run the optimization
                    optimal_weights, optimal_performance = optimize_portfolio(price_data)

                    col1, col2 = st.columns([1, 2])

                    with col1:
                     st.subheader("Suggested Optimal Weights")
    # Display weights as a pie chart
                     df_optimal = pd.DataFrame.from_dict(optimal_weights, orient='index', columns=['Weight'])
                     fig_optimal_pie = px.pie(df_optimal, values='Weight', names=df_optimal.index, title="Optimal Portfolio Composition")
                     st.plotly_chart(fig_optimal_pie, use_container_width=True)

                    with col2:
                     st.subheader("Optimal Portfolio's Expected Performance")
    # Display the performance metrics
                    st.metric(
                    "Expected Annual Return", 
                    f"{optimal_performance[0]:.2%}"
    )
                    st.metric(
                    "Expected Annual Volatility", 
                    f"{optimal_performance[1]:.2%}"
    )
                    st.metric(
                    "Expected Sharpe Ratio", 
                    f"{optimal_performance[2]:.2f}"
    )
                    # --- VaR and CVaR Section ---
                    st.header("📉 Value at Risk (VaR) & Conditional VaR (CVaR)")
                    st.markdown("These metrics quantify the potential downside risk of the portfolio for a single day.")

                    col1, col2 = st.columns(2)
                    col1.metric(
                        "Value at Risk (VaR) at 95%",
                        f"{var:.2%}",
                        help="This is the most you can expect to lose on a given day, with 95% confidence. A VaR of -2.5% means that on 95 out of 100 days, your losses are not expected to exceed 2.5%."
                    )
                    col2.metric(
                        "Conditional VaR (CVaR) at 95%",
                        f"{cvar:.2%}",
                        help="This is the average loss you can expect on the worst 5% of days. It provides insight into the 'tail risk' of your portfolio."
                    )

                    st.divider()

                    # --- Charting Section ---
                    st.header("📊 Visualizations")
                    st.subheader("Portfolio Growth")
                    cumulative_returns = (1 + port_ret_series).cumprod()
                    st.line_chart(cumulative_returns)
                    
                    # --- NEW ADDITION: Matplotlib Histogram ---
                    st.subheader("Distribution of Daily Returns")
                    fig, ax = plt.subplots()
                    ax.hist(port_ret_series, bins=50, alpha=0.75, edgecolor='black')
                    ax.set_title("Histogram of Portfolio Daily Returns")
                    ax.set_xlabel("Daily Return")
                    ax.set_ylabel("Frequency")
                    st.pyplot(fig) # Use st.pyplot() to display matplotlib figures


# Inside the 'if analyze_button:' block, after other st.divider() sections

# --- Monte Carlo Simulation ---
                    st.divider()
                    st.header("🔮 Future Projections: Monte Carlo Simulation")
                    st.markdown("This simulation runs 1,000 possible scenarios for your portfolio's growth over the next year (252 trading days).")

# Run the simulation using the calculated portfolio return series
                    mc_simulation_results = run_monte_carlo_simulation(port_ret_series)

# Create and display the Plotly chart
                    fig_mc = px.line(mc_simulation_results, title="Monte Carlo Simulation: 1,000 Portfolio Paths")
                    fig_mc.update_layout(showlegend=False, yaxis_title="Portfolio Value Growth (Initial $1)", xaxis_title="Trading Days")
                    st.plotly_chart(fig_mc, use_container_width=True)

# Display summary statistics from the simulation's final day
                    final_values = mc_simulation_results.iloc[-1]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Best Case (95th percentile)", f"${final_values.quantile(0.95):.2f}", help="The portfolio value at the 95th percentile after one year.")
                    col2.metric("Median Outcome (50th percentile)", f"${final_values.median():.2f}", help="The most likely portfolio value after one year.")
                    col3.metric("Worst Case (5th percentile)", f"${final_values.quantile(0.05):.2f}", help="The portfolio value at the 5th percentile after one year.")

# In dashboard.py, at the end of the 'if analyze_button:' block

# --- Prophet Forecasting Section ---
                    st.divider()
                    st.header("📈 AI-Powered Value Forecast (Prophet)")
                    st.markdown("Forecasting the potential value of the portfolio over the next year, including optimistic (yhat_upper) and pessimistic (yhat_lower) scenarios.")

# Run the Prophet forecast
                    prophet_forecast = forecast_prophet_value(price_data, weights)

# Create and display the Prophet forecast chart
                    fig_prophet = px.line(prophet_forecast, x='ds', y='yhat', title='Portfolio Value Forecast')
# Add upper and lower bounds
                    fig_prophet.add_scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_upper'], fill='tozeroy', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='Upper Bound')
                    fig_prophet.add_scatter(x=prophet_forecast['ds'], y=prophet_forecast['yhat_lower'], fill='tozeroy', fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'), name='Lower Bound')
                    fig_prophet.update_layout(yaxis_title="Portfolio Value (Initial $1)", xaxis_title="Date")
                    st.plotly_chart(fig_prophet, use_container_width=True)

    except ValueError:
        st.error("Please ensure weights are valid, comma-separated numbers.")


        
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Please enter your portfolio details in the sidebar and click 'Analyze Portfolio'.")