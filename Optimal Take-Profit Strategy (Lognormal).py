# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 20:05:50 2025

@author: Yunus
"""

import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
initial_portfolio = 45000
target_portfolio = 110000
num_years = 5
days_per_year = 365
total_days = num_years * days_per_year
num_simulations = 200

# Market return assumptions
mu = 0.15  # annual expected return
sigma = 0.20  # annual volatility
daily_mu = mu / days_per_year
daily_sigma = sigma / np.sqrt(days_per_year)

# Take profit milestones and rules
milestones = [50000, 60000, 70000, 80000, 90000, 100000]
take_profits = [1000, 2000, 3000, 4000, 5000, 10000]  # Take profits at each milestone
reserve = 0
reserve_log = []

# Function to simulate one path with TP logic
def simulate_with_take_profit():
    portfolio = initial_portfolio
    portfolio_path = [portfolio]
    milestone_idx = 0
    reserve = 0
    peak = portfolio

    for _ in range(total_days):
        # Simulate daily return
        log_return = np.random.normal(daily_mu, daily_sigma)
        portfolio *= np.exp(log_return)

        
        # Check for milestone
        if milestone_idx < len(milestones) and portfolio >= milestones[milestone_idx]:
            reserve += take_profits[milestone_idx]
            portfolio -= take_profits[milestone_idx]
            milestone_idx += 1

        # Reinvest if drawdown from peak
        if portfolio > peak:
            peak = portfolio
        elif peak - portfolio >= 0.05 * peak and reserve > 0:
            reinvest_amount = min(reserve, 0.5 * reserve)
            portfolio += reinvest_amount
            reserve -= reinvest_amount

        portfolio_path.append(portfolio)

    return portfolio_path

# Run simulations
all_paths = np.array([simulate_with_take_profit() for _ in range(num_simulations)])

# Plot
plt.figure(figsize=(12, 6))
for path in all_paths:
    plt.plot(path, color='skyblue', alpha=0.3)
plt.plot(np.mean(all_paths, axis=0), color='navy', label='Average Growth', linewidth=2)
plt.axhline(y=100000, color='green', linestyle='--', label='Target: 100k')
plt.title('Monte Carlo Simulation: Portfolio Growth with Dynamic Take Profit Strategy')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
