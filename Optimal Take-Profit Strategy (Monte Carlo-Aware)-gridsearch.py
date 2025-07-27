# -*- coding: utf-8 -*-
"""
Created on Fri Jul 25 20:05:50 2025

@author: Yunus
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

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

# Function to simulate one path with TP logic, now parameterized
def simulate_with_take_profit(milestones, take_profits):
    portfolio = initial_portfolio
    portfolio_path = [portfolio]
    milestone_idx = 0
    reserve = 0
    peak = portfolio

    for _ in range(total_days):
        # Simulate daily return
        daily_return = np.random.normal(daily_mu, daily_sigma)
        portfolio *= (1 + daily_return)
        
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

# Define ranges for grid search
milestone_grid = [
    [50000, 60000, 70000, 80000, 90000, 100000],
    [52000, 64000, 76000, 88000, 100000, 112000],
    [48000, 58000, 68000, 78000, 88000, 98000],
]
take_profit_grid = [
    [1000, 2000, 3000, 4000, 5000, 10000],
    [1500, 2500, 3500, 4500, 5500, 12000],
    [800, 1800, 2800, 3800, 4800, 9000],
]

best_avg = -float('inf')
best_milestones = None
best_take_profits = None

print("Starting grid search...")
for milestones, take_profits in itertools.product(milestone_grid, take_profit_grid):
    all_paths = np.array([simulate_with_take_profit(milestones, take_profits) for _ in range(num_simulations)])
    avg_final = np.mean(all_paths[:, -1])
    print(f"Milestones: {milestones}, Take Profits: {take_profits}, Avg Final: {avg_final:.2f}")
    if avg_final > best_avg:
        best_avg = avg_final
        best_milestones = milestones
        best_take_profits = take_profits

print("\nBest Strategy:")
print("Milestones:", best_milestones)
print("Take Profits:", best_take_profits)
print("Best Average Final Portfolio:", best_avg)

# Run and plot the best strategy
all_paths = np.array([simulate_with_take_profit(best_milestones, best_take_profits) for _ in range(num_simulations)])

plt.figure(figsize=(12, 6))
for path in all_paths:
    plt.plot(path, color='skyblue', alpha=0.3)
plt.plot(np.mean(all_paths, axis=0), color='navy', label='Average Growth', linewidth=2)
plt.axhline(y=100000, color='green', linestyle='--', label='Target: 100k')
plt.title('Monte Carlo Simulation: Portfolio Growth with Best Take Profit Strategy')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 