def risk_metrics(returns, weights):
    cov_matrix = returns.cov()
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_vol = np.sqrt(port_var)
    mean_ret = np.dot(returns.mean(), weights) * 252  # annualized
    sharpe = mean_ret / port_vol
    return {"Mean Return": mean_ret, "Volatility": port_vol, "Sharpe Ratio": sharpe}
