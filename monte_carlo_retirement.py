"""
Monte Carlo Retirement Portfolio Simulation
-------------------------------------------
Simulates the survival of an investment portfolio under annual,
inflation-adjusted withdrawals over a multi-decade horizon.

Key finding: adjusting withdrawals for inflation drops the survival
rate from ~90% (fixed withdrawals) to ~50% under these assumptions.

Outputs:
    - Portfolio survival rate and terminal wealth percentiles (console)
    - images/simulation_paths.png    : wealth paths + percentile band
    - images/final_distribution.png  : distribution of surviving portfolios
    - images/log_distribution.png    : log-scale view (approx. log-normal)

Author: Luis Camilo Araujo Sanabria
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Save all figures next to this script, regardless of where it is run from.
IMAGES_DIR = Path(__file__).parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# Simulation core
# ----------------------------------------------------------------------
def simulate_retirement(
    initial_portfolio: float = 500_000,
    annual_withdrawal: float = 25_000,
    mean_return: float = 0.07,
    std_return: float = 0.15,
    inflation: float = 0,
    n_years: int = 30,
    n_simulations: int = 10_000,
    seed: int = 42,
):
    """
    Run a Monte Carlo simulation of portfolio survival.

    Withdrawals grow with inflation each year. If the portfolio is
    depleted (ruin), its value is floored at zero and stays there.

    Returns
    -------
    final_values : np.ndarray
        Terminal portfolio value per simulation (floored at 0).
    paths : np.ndarray
        Wealth paths, shape (n_simulations, n_years + 1).
    """
    rng = np.random.default_rng(seed)
    paths = np.zeros((n_simulations, n_years + 1))
    paths[:, 0] = initial_portfolio

    for i in range(n_simulations):
        portfolio = initial_portfolio
        withdrawal = annual_withdrawal

        for year in range(1, n_years + 1):
            annual_return = rng.normal(mean_return, std_return)
            portfolio = portfolio * (1 + annual_return) - withdrawal
            withdrawal *= 1 + inflation  # inflation-adjusted withdrawal

            if portfolio <= 0:
                portfolio = 0.0  # ruin: wealth cannot go negative
                paths[i, year:] = 0.0
                break

            paths[i, year] = portfolio

    final_values = paths[:, -1]
    return final_values, paths


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def print_summary(final_values: np.ndarray) -> None:
    """Print survival rate and key percentiles of terminal wealth."""
    survival_rate = np.mean(final_values > 0)
    print(f"Portfolio survival rate: {survival_rate:.1%}\n")

    p5, p25, p50, p75, p95 = np.percentile(final_values, [5, 25, 50, 75, 95])
    print("Terminal wealth percentiles:")
    print(f"  5th percentile:  ${p5:,.0f}")
    print(f"  25th percentile: ${p25:,.0f}")
    print(f"  Median:          ${p50:,.0f}")
    print(f"  75th percentile: ${p75:,.0f}")
    print(f"  95th percentile: ${p95:,.0f}")


# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------
def _millions_formatter():
    """Format axis ticks as $X.XM for readability."""
    return plt.FuncFormatter(lambda x, _: f"${x / 1e6:.1f}M")


def _save_and_show(filename: str) -> None:
    """Save the current figure to the images folder, then display it."""
    plt.savefig(IMAGES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.show()


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def plot_paths(paths: np.ndarray, n_display: int = 500) -> None:
    """
    Plot a sample of simulated wealth paths (green = survived,
    red = ruined) with the 5th-95th percentile band and median.

    The y-axis is capped at the 99th percentile of terminal wealth so
    that a handful of extreme outliers do not compress the region
    where most of the probability mass lives.
    """
    final_values = paths[:, -1]
    survival_rate = np.mean(final_values > 0)
    years = np.arange(paths.shape[1])

    plt.figure(figsize=(12, 6))

    # Sample of individual paths, colored by outcome
    for path, final in zip(paths[:n_display], final_values[:n_display]):
        color = "green" if final > 0 else "red"
        plt.plot(years, path, alpha=0.05, color=color, linewidth=0.5)

    # Percentile band and median across ALL simulations
    p5 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    median = np.percentile(paths, 50, axis=0)
    plt.fill_between(years, p5, p95, color="steelblue", alpha=0.25,
                     label="5th-95th percentile band")
    plt.plot(years, median, color="navy", linewidth=2, label="Median path")

    plt.axhline(y=0, color="black", linestyle="--", linewidth=1.5)

    # Cap the axis at the 99th percentile: fat right tails otherwise
    # squash the median and band against the bottom of the chart.
    y_cap = np.percentile(final_values, 99)
    plt.ylim(-0.02 * y_cap, y_cap)

    plt.title(
        f"Monte Carlo Retirement Simulation (inflation-adjusted withdrawals)\n"
        f"Survival rate: {survival_rate:.1%}  |  "
        f"y-axis capped at 99th percentile (top 1% of paths exceed limit)"
    )
    plt.xlabel("Years")
    plt.ylabel("Portfolio value")
    plt.gca().yaxis.set_major_formatter(_millions_formatter())
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")

    _save_and_show("simulation_paths.png")


def plot_final_distribution(final_values: np.ndarray) -> None:
    """
    Plot the distribution of terminal wealth for SURVIVING portfolios.

    Ruined portfolios (~half of all simulations) are excluded from the
    histogram and reported as a bold annotation instead: a single bar
    at zero would otherwise dominate the chart and hide the shape of
    the surviving distribution.
    """
    survivors = final_values[final_values > 0]
    ruin_rate = np.mean(final_values == 0)

    plt.figure(figsize=(10, 5))
    plt.hist(survivors, bins=80, edgecolor="black", color="steelblue")

    median = np.median(survivors)
    plt.axvline(x=median, color="navy", linestyle="-", linewidth=1.5,
                label=f"Median (survivors): ${median / 1e6:.2f}M")

    # Cap the x-axis at the survivors' 99th percentile for readability
    plt.xlim(0, np.percentile(survivors, 99))

    plt.annotate(
        f"Ruined portfolios: {ruin_rate:.1%}\n(excluded from histogram)",
        xy=(0.62, 0.82), xycoords="axes fraction",
        fontsize=12, fontweight="bold", color="darkred",
    )

    plt.title("Distribution of Final Portfolio Values (surviving paths)")
    plt.xlabel("Final portfolio value")
    plt.ylabel("Frequency")
    plt.gca().xaxis.set_major_formatter(_millions_formatter())
    plt.grid(alpha=0.3)
    plt.legend()

    _save_and_show("final_distribution.png")


def plot_log_distribution(final_values: np.ndarray) -> None:
    """
    Plot surviving terminal wealth on a log-scale x-axis.

    Compounded returns produce right-skewed, approximately log-normal
    wealth: on a log axis the distribution looks roughly bell-shaped,
    which makes the skewness of the linear view easy to explain.
    """
    survivors = final_values[final_values > 0]

    plt.figure(figsize=(10, 5))
    log_bins = np.logspace(
        np.log10(survivors.min()), np.log10(survivors.max()), 60
    )
    plt.hist(survivors, bins=log_bins, edgecolor="black", color="steelblue")
    plt.xscale("log")

    plt.title("Terminal Wealth of Surviving Portfolios (log scale)\n"
              "Approximately log-normal, a direct consequence of compounding")
    plt.xlabel("Final portfolio value (log scale)")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3, which="both")

    _save_and_show("log_distribution.png")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    final_values, paths = simulate_retirement()

    print_summary(final_values)
    plot_paths(paths)
    plot_final_distribution(final_values)
    plot_log_distribution(final_values)
