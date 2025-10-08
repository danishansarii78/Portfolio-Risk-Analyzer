# Portfolio-Risk-Analyzer
Portfolio-Risk-Analyzer-
🤖 AI-Driven Portfolio Risk Analyzer
Python Version Streamlit License

An interactive web dashboard built to perform sophisticated risk analysis on stock portfolios using Python, quantitative finance, and AI forecasting models.

This tool empowers investors and analysts to move beyond simple return tracking and incorporate a suite of advanced techniques for a deeper understanding of portfolio risk and potential.



📖 About The Project
This application provides a user-friendly interface to perform complex financial analysis that is typically reserved for institutional investors. By simply providing a list of stock tickers and their corresponding weights, users can generate a comprehensive risk profile for their portfolio.

The project integrates historical data fetching, statistical analysis, and predictive modeling to deliver actionable insights, helping to answer critical questions like:

What is my portfolio's true risk-adjusted return?
What is the maximum I can expect to lose on a bad day?
Is my portfolio allocated optimally?
What are the potential future growth scenarios?
✨ Key Features
Feature	Description	Key Library/Technique
Core Performance Metrics	Calculates annualized return, volatility (risk), and the Sharpe Ratio.	NumPy, Pandas
Downside Risk Analysis	Quantifies worst-case daily losses using Value at Risk (VaR) and Conditional VaR (CVaR).	NumPy
AI-Powered Optimization	Suggests optimal portfolio weights to maximize the risk-adjusted return.	PyPortfolioOpt
Volatility Forecasting	Models volatility clustering and forecasts future risk using a GARCH model.	ARCH
Value Forecasting	Forecasts the portfolio's future value trajectory with uncertainty intervals.	Prophet
Scenario Simulation	Runs thousands of Monte Carlo simulations to model a range of possible future outcomes.	NumPy
Interactive Dashboard	A clean, user-friendly interface for inputting data and visualizing results.	Streamlit
Data Visualization	Interactive charts for portfolio growth, forecasts, and return distributions.	Plotly, Matplotlib
🛠️ Tech Stack
Application Framework: Streamlit
Data & Financial APIs: yfinance
Data Manipulation & Analysis: Pandas, NumPy
Quantitative Finance & AI: PyPortfolioOpt, ARCH, Prophet
Visualization: Plotly, Matplotlib
🚀 Getting Started
Follow these steps to get a local copy of the project up and running.

Prerequisites
Python (version 3.9 or higher is recommended)
pip package manager
Git for cloning the repository
Installation
Clone the repository Open your terminal (e.g., PowerShell) and run:

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
Create and activate a virtual environment This isolates the project's dependencies from your system.

python -m venv env
.\env\Scripts\Activate.ps1
(Note: If you encounter an execution policy error on Windows, run Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser and try activating again.)

Install the required packages All dependencies are listed in the requirements.txt file.

pip install -r requirements.txt
Run the application Launch the Streamlit web server.

streamlit run dashboard.py
Your default browser will open a new tab with the application.

Usage
Once the application is running:

Use the sidebar to enter the stock tickers you want to analyze, separated by commas (e.g., GOOGL, AAPL, MSFT).
Enter the corresponding portfolio weights, also separated by commas (e.g., 0.5, 0.25, 0.25).
Select the historical date range for the analysis.
Click the "Analyze Portfolio" button.
Explore the generated metrics, forecasts, and charts in the main dashboard area.
