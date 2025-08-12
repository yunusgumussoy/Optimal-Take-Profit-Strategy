import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t

# --- Reproducibility ---
np.random.seed(42)

# --- Simulation parameters ---
initial_portfolio = 45000
target_portfolio = 110000
num_years = 5
days_per_year = 365
total_days = num_years * days_per_year
num_simulations = 200

# --- Market return assumptions ---
mu = 0.15   # annual expected return
sigma = 0.20  # annual volatility
daily_mu = mu / days_per_year
daily_sigma = sigma / np.sqrt(days_per_year)

# --- Take profit milestones and rules ---
milestones = [50000, 60000, 70000, 80000, 90000, 100000]
take_profits = [1000, 2000, 3000, 4000, 5000, 10000]  # take profits at each milestone

# --- Function: simulate with take-profit ---
def simulate_with_take_profit():
    portfolio = initial_portfolio
    portfolio_path = [portfolio]
    milestone_idx = 0
    reserve = 0
    peak = portfolio

    for _ in range(total_days):
        # Simulate daily return using Student-t with unit variance scaling
        df = 4
        student_t_sample = t.rvs(df) / np.sqrt(df / (df - 2))
        scaled_t_return = daily_mu + daily_sigma * student_t_sample
        portfolio *= np.exp(scaled_t_return)

        # Check for hitting multiple milestones in one day
        while milestone_idx < len(milestones) and portfolio >= milestones[milestone_idx]:
            reserve += take_profits[milestone_idx]
            portfolio -= take_profits[milestone_idx]
            milestone_idx += 1

        # Reinvest if drawdown from peak
        if portfolio > peak:
            peak = portfolio
        elif peak - portfolio >= 0.05 * peak and reserve > 0:
            reinvest_amount = 0.5 * reserve
            portfolio += reinvest_amount
            reserve -= reinvest_amount

        portfolio_path.append(portfolio)

    return portfolio_path

# --- Function: simulate without take-profit ---
def simulate_no_take_profit():
    portfolio = initial_portfolio
    portfolio_path = [portfolio]

    for _ in range(total_days):
        df = 4
        student_t_sample = t.rvs(df) / np.sqrt(df / (df - 2))
        scaled_t_return = daily_mu + daily_sigma * student_t_sample
        portfolio *= np.exp(scaled_t_return)
        portfolio_path.append(portfolio)

    return portfolio_path

# --- Run simulations ---
paths_with_tp = np.array([simulate_with_take_profit() for _ in range(num_simulations)])
paths_no_tp = np.array([simulate_no_take_profit() for _ in range(num_simulations)])

final_with_tp = paths_with_tp[:, -1]
final_no_tp = paths_no_tp[:, -1]

# --- Probability of hitting target ---
prob_with_tp = np.mean(final_with_tp >= target_portfolio) * 100
prob_no_tp = np.mean(final_no_tp >= target_portfolio) * 100

# --- Print results ---
print(f"--- Results over {num_years} years ---")
print(f"With Take-Profit: Avg Final = {np.mean(final_with_tp):,.0f}, Target Hit Prob = {prob_with_tp:.2f}%")
print(f"No Take-Profit:   Avg Final = {np.mean(final_no_tp):,.0f}, Target Hit Prob = {prob_no_tp:.2f}%")

# --- Plot final value distributions ---
plt.figure(figsize=(12, 5))
plt.hist(final_with_tp, bins=30, alpha=0.6, color='blue', edgecolor='black', label='With Take-Profit')
plt.hist(final_no_tp, bins=30, alpha=0.6, color='orange', edgecolor='black', label='No Take-Profit')
plt.axvline(target_portfolio, color='red', linestyle='--', linewidth=2, label='Target')
plt.title('Distribution of Final Portfolio Values')
plt.xlabel('Final Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Plot average growth paths ---
plt.figure(figsize=(12, 6))
plt.plot(np.mean(paths_with_tp, axis=0), color='blue', linewidth=2, label='Average With Take-Profit')
plt.plot(np.mean(paths_no_tp, axis=0), color='orange', linewidth=2, label='Average No Take-Profit')
plt.axhline(y=target_portfolio, color='green', linestyle='--', label=f'Target: {target_portfolio/1000:.0f}k')
plt.title('Average Portfolio Growth: With vs Without Take-Profit')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
