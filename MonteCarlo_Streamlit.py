"""
PROBUS Portfolio Simulator — Streamlit web application

Run locally with:
    python3 -m streamlit run MonteCarloSim_Streamlit.py

The Monte Carlo calculation functions below are carried over from the original
Tkinter application. The interface has been replaced with Streamlit.
"""

import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st

TRANSITIONS = {
    "bull": {"bull": 0.96, "neutral": 0.04, "bear": 0.00, "crisis": 0.00},
    "neutral": {"bull": 0.05, "neutral": 0.90, "bear": 0.04, "crisis": 0.01},
    "bear": {"bull": 0.02, "neutral": 0.07, "bear": 0.90, "crisis": 0.01},
    "crisis": {"bull": 0.00, "neutral": 0.20, "bear": 0.30, "crisis": 0.50},
}

REGIME_RANGES = {
    "neutral": {"return": (0.00, 0.00), "vol": (0.00, 0.00)},
    "bull": {"return": (0.02, 0.06), "vol": (-0.07, -0.03)},
    "bear": {"return": (-0.08, -0.03), "vol": (0.05, 0.15)},
    "crisis": {"return": (-0.30, -0.15), "vol": (0.15, 0.35)},
}

CALIBRATION_MONTHS = 100000
DF = 5

# CRA prescribed minimum factors for all other RRIFs. The factor is based on
# age at the beginning of the year and the portfolio value at the start of it.
RRIF_MINIMUM_FACTORS = {
    71: 0.0528, 72: 0.0540, 73: 0.0553, 74: 0.0567, 75: 0.0582,
    76: 0.0598, 77: 0.0617, 78: 0.0636, 79: 0.0658, 80: 0.0682,
    81: 0.0708, 82: 0.0738, 83: 0.0771, 84: 0.0808, 85: 0.0851,
    86: 0.0899, 87: 0.0955, 88: 0.1021, 89: 0.1099, 90: 0.1192,
    91: 0.1306, 92: 0.1449, 93: 0.1634, 94: 0.1879, 95: 0.2000,
}


def rrif_minimum_factor(age):
    """Return the federal minimum factor for a standard (all other) RRIF."""
    if age < 71:
        return 0.0
    return RRIF_MINIMUM_FACTORS.get(age, 0.20)


def next_regime(current):
    states = list(TRANSITIONS[current].keys())
    probs = list(TRANSITIONS[current].values())
    return np.random.choice(states, p=probs)


def draw_regime_parameters(regime):
    r_low, r_high = REGIME_RANGES[regime]["return"]
    v_low, v_high = REGIME_RANGES[regime]["vol"]
    return np.random.uniform(r_low, r_high), np.random.uniform(v_low, v_high)


def run_simulation(inputs, progress_callback=None):
    opening_portfolio = inputs["opening_portfolio"]
    years = inputs["years"]
    monthly_cashflow = inputs["monthly_cashflow"]
    annual_return = inputs["annual_return"] / 100.0
    annual_volatility = inputs["annual_volatility"] / 100.0
    num_sims = inputs["num_sims"]
    starting_age = inputs.get("current_age")
    use_rrif = inputs.get("use_rrif", False)

    months = years * 12

    # Calibration
    current_regime = "neutral"
    regime_shifts = []
    return_shift, _ = draw_regime_parameters(current_regime)

    for _ in range(CALIBRATION_MONTHS):
        new_regime = next_regime(current_regime)
        if new_regime != current_regime:
            current_regime = new_regime
            return_shift, _ = draw_regime_parameters(current_regime)
        regime_shifts.append(return_shift)

    calibration_offset = -np.mean(regime_shifts)

    # Storage and Monte Carlo simulation
    all_paths = np.zeros((num_sims, months + 1))
    terminal_values = np.zeros(num_sims)
    bankrupt_count = 0

    for sim in range(num_sims):
        portfolio = opening_portfolio
        ruined = False
        current_regime = "neutral"
        return_shift, vol_shift = draw_regime_parameters(current_regime)
        path = [portfolio]

        for month in range(months):
            if ruined:
                path.append(0)
                continue

            new_regime = next_regime(current_regime)
            if new_regime != current_regime:
                current_regime = new_regime
                return_shift, vol_shift = draw_regime_parameters(current_regime)

            effective_return = annual_return + calibration_offset + return_shift
            effective_volatility = max(
                0.02, annual_volatility + vol_shift
            )

            monthly_mean = np.log(max(0.001, 1 + effective_return)) / 12
            monthly_vol = effective_volatility / np.sqrt(12)
            t_std = np.sqrt(DF / (DF - 2))
            scale = monthly_vol / t_std

            simulation_year = month // 12
            age_this_year = starting_age + simulation_year if starting_age is not None else None

            # RRIF minimum is withdrawn once at the beginning of each year,
            # before that year's investment returns are applied.
            if use_rrif and month % 12 == 0 and age_this_year >= 71:
                portfolio -= portfolio * rrif_minimum_factor(age_this_year)
            elif not use_rrif or age_this_year < 71:
                portfolio += monthly_cashflow
            if portfolio <= 0:
                portfolio = 0
                ruined = True
                path.append(0)
                continue

            log_return = monthly_mean + np.random.standard_t(DF) * scale
            portfolio *= np.exp(log_return)
            path.append(portfolio)

        if ruined:
            bankrupt_count += 1

        all_paths[sim] = path
        terminal_values[sim] = portfolio

        if progress_callback and (sim == num_sims - 1 or (sim + 1) % max(1, num_sims // 100) == 0):
            progress_callback(sim + 1, num_sims)

    # Median and smooth growth paths
    median_path = np.median(all_paths, axis=0)

    naive_path = [opening_portfolio]
    portfolio = opening_portfolio
    monthly_growth = (1 + annual_return) ** (1 / 12) - 1

    for month in range(months):
        simulation_year = month // 12
        age_this_year = starting_age + simulation_year if starting_age is not None else None
        if use_rrif and month % 12 == 0 and age_this_year >= 71:
            portfolio -= portfolio * rrif_minimum_factor(age_this_year)
        elif not use_rrif or age_this_year < 71:
            portfolio += monthly_cashflow
        if portfolio > 0:
            portfolio *= (1 + monthly_growth)
        else:
            portfolio = 0
        naive_path.append(portfolio)

    # Statistics
    median_terminal = np.median(terminal_values)
    mean_terminal = np.mean(terminal_values)
    p5 = np.percentile(terminal_values, 5)
    p95 = np.percentile(terminal_values, 95)
    best_terminal = np.max(terminal_values)
    worst_terminal = np.min(terminal_values)
    probability_ruin = bankrupt_count / num_sims

    median_index = np.argmin(np.abs(terminal_values - median_terminal))
    median_terminal_sim = terminal_values[median_index]

    cashflows = [-opening_portfolio]
    for month in range(months):
        simulation_year = month // 12
        age_this_year = starting_age + simulation_year if starting_age is not None else None
        if use_rrif and month % 12 == 0 and age_this_year >= 71:
            # Use the median path's opening balance as the approximate RRIF
            # payment stream for the displayed median annualized IRR.
            cashflows.append(median_path[month] * rrif_minimum_factor(age_this_year))
        elif not use_rrif or age_this_year < 71:
            if monthly_cashflow > 0:
                cashflows.append(-monthly_cashflow)
            elif monthly_cashflow < 0:
                cashflows.append(abs(monthly_cashflow))
            else:
                cashflows.append(0)
        else:
            cashflows.append(0)

    cashflows[-1] += median_terminal_sim
    monthly_irr = npf.irr(cashflows)

    if np.isfinite(monthly_irr):
        annual_irr = (1 + monthly_irr) ** 12 - 1
    else:
        annual_irr = np.nan

    return {
        "all_paths": all_paths,
        "median_path": median_path,
        "naive_path": naive_path,
        "median_terminal": median_terminal,
        "mean_terminal": mean_terminal,
        "naive_terminal": naive_path[-1],
        "difference": median_terminal - naive_path[-1],
        "p5": p5,
        "p95": p95,
        "best_terminal": best_terminal,
        "worst_terminal": worst_terminal,
        "probability_ruin": probability_ruin,
        "annual_irr": annual_irr,
        "num_sims": num_sims,
    }


# ---------------------------------------------------------------------
# STREAMLIT WEB INTERFACE
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="PROBUS Portfolio Simulator",
    page_icon="📈",
    layout="wide",
)

st.title("PROBUS Portfolio Simulator")
st.caption("Market-regime Monte Carlo analysis")
st.markdown(
    "Explore how contributions, withdrawals, investment returns and market "
    "volatility may affect a portfolio over time."
)

if "simulation_results" not in st.session_state:
    st.session_state.simulation_results = None
if "simulation_inputs" not in st.session_state:
    st.session_state.simulation_inputs = None

# Keep the input controls together in a form so changing a value does not
# trigger a simulation until the user explicitly presses Run Simulation.
with st.form("assumptions_form"):
    st.subheader("Simulation Assumptions")
    col1, col2 = st.columns(2)

    with col1:
        opening_portfolio = st.number_input(
            "Opening Portfolio ($)", min_value=1.0, value=1_000_000.0,
            step=10_000.0, format="%.2f"
        )
        years = st.number_input(
            "Years to Simulate", min_value=1, max_value=100,
            value=25, step=1
        )
        monthly_cashflow = st.number_input(
            "Monthly Contribution (+) / Withdrawal (-) ($)",
            value=-5_000.0, step=250.0, format="%.2f"
        )

    with col2:
        annual_return = st.number_input(
            "Expected Annual Return (%)", value=7.0,
            step=0.5, format="%.2f"
        )
        annual_volatility = st.number_input(
            "Expected Annual Volatility (%)", min_value=0.0,
            value=15.0, step=0.5, format="%.2f"
        )
        num_sims = st.number_input(
            "Number of Simulations", min_value=1, max_value=100_000,
            value=5_000, step=500
        )

    st.divider()
    rrif_col1, rrif_col2 = st.columns([1, 2])
    with rrif_col1:
        current_age = st.number_input(
            "Current Age (at December 31)", min_value=0, max_value=120,
            value=None, step=1, placeholder="Optional"
        )
    with rrif_col2:
        use_rrif = st.checkbox("Apply federal RRIF minimum withdrawal rates")

    st.caption(
        "A positive monthly cash flow is a contribution; a negative value is a withdrawal. "
        "With RRIF rates enabled, this monthly amount applies through age 70. "
        "From age 71, the annual minimum is withdrawn at the start of each year, before investment growth."
    )

    submitted = st.form_submit_button(
        "▶ Run Simulation", type="primary", use_container_width=True
    )

if submitted:
    if annual_return <= -100:
        st.error("Expected annual return must be greater than -100%.")
    elif use_rrif and current_age is None:
        st.error("Enter your current age to apply RRIF minimum withdrawal rates.")
    else:
        inputs = {
            "opening_portfolio": float(opening_portfolio),
            "years": int(years),
            "monthly_cashflow": float(monthly_cashflow),
            "annual_return": float(annual_return),
            "annual_volatility": float(annual_volatility),
            "num_sims": int(num_sims),
            "current_age": int(current_age) if current_age is not None else None,
            "use_rrif": bool(use_rrif),
        }
        estimated_points = inputs["num_sims"] * (inputs["years"] * 12 + 1)
        if estimated_points > 20_000_000:
            st.warning(
                "This is a large simulation and may require substantial memory. "
                "Consider reducing the number of simulations or years if your Mac becomes sluggish."
            )
        try:
            with st.spinner(
                f"Running {inputs['num_sims']:,} simulations over {inputs['years']} years. "
                "Please wait..."
            ):
                result = run_simulation(inputs)
            st.session_state.simulation_results = result
            st.session_state.simulation_inputs = inputs
            st.success("Simulation complete.")
        except Exception as exc:
            st.error(f"The simulation encountered an error: {exc}")

results = st.session_state.simulation_results
inputs = st.session_state.simulation_inputs

input_tab, results_tab, chart_tab = st.tabs(
    ["Inputs", "Financial Results", "Chart"]
)

with input_tab:
    st.info(
        "Set your assumptions above and select **Run Simulation**. "
        "Your latest results will appear in the Financial Results and Chart tabs."
    )
    if inputs:
        st.write("**Last simulation settings**")
        st.json(inputs)

with results_tab:
    st.subheader("Simulation Results")
    if results is None:
        st.info("Run a simulation to populate these results.")
    else:
        currency_items = [
            ("Median Terminal Value", results["median_terminal"]),
            ("Mean Terminal Value", results["mean_terminal"]),
            ("Smooth Growth Forecast", results["naive_terminal"]),
            ("Difference vs. Smooth Forecast", results["difference"]),
            ("5th Percentile", results["p5"]),
            ("95th Percentile", results["p95"]),
            ("Best Simulation", results["best_terminal"]),
            ("Worst Simulation", results["worst_terminal"]),
        ]
        metric_cols = st.columns(2)
        for index, (label, value) in enumerate(currency_items):
            with metric_cols[index % 2]:
                st.metric(label, f"${value:,.0f}")

        irr_value = results["annual_irr"]
        irr_text = f"{irr_value:.2%}" if np.isfinite(irr_value) else "Not available"
        extra1, extra2 = st.columns(2)
        extra1.metric("Median IRR (annualized)", irr_text)
        extra2.metric("Probability of Ruin", f"{results['probability_ruin']:.2%}")

        lines = [
            "PROBUS PORTFOLIO SIMULATOR RESULTS",
            "=" * 42,
            f"Median Terminal Value: ${results['median_terminal']:,.0f}",
            f"Mean Terminal Value: ${results['mean_terminal']:,.0f}",
            f"Smooth Growth Forecast: ${results['naive_terminal']:,.0f}",
            f"Difference vs. Smooth Forecast: ${results['difference']:,.0f}",
            f"5th Percentile: ${results['p5']:,.0f}",
            f"95th Percentile: ${results['p95']:,.0f}",
            f"Best Simulation: ${results['best_terminal']:,.0f}",
            f"Worst Simulation: ${results['worst_terminal']:,.0f}",
            f"Median IRR (annualized): {irr_text}",
            f"Probability of Ruin: {results['probability_ruin']:.2%}",
            f"Number of Simulations: {results['num_sims']:,}",
        ]
        st.download_button(
            "Save Results to Text File",
            data="\n".join(lines),
            file_name="PROBUS_Simulation_Results.txt",
            mime="text/plain",
        )

with chart_tab:
    st.subheader("Market Regime Monte Carlo Simulation")
    if results is None:
        st.info("Run a simulation to display the chart.")
    else:
        fig, ax = plt.subplots(figsize=(11, 6))
        num_display = min(20, results["num_sims"])
        sample_indices = np.random.choice(
            results["num_sims"], size=num_display, replace=False
        )
        for idx in sample_indices:
            ax.plot(results["all_paths"][idx], color="lightgray", linewidth=1)
        ax.plot(
            results["median_path"], color="blue", linewidth=3,
            label="Median Monte Carlo Outcome"
        )
        ax.plot(
            results["naive_path"], color="green", linewidth=3,
            label="Smooth Growth Forecast"
        )
        ax.axhline(
            0, color="red", linestyle="--", linewidth=2, label="Ruin Level"
        )
        ax.set_title("Market Regime Monte Carlo Simulation")
        ax.set_xlabel("Months")
        ax.set_ylabel("Portfolio Value ($)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, p: f"${x/1_000_000:.1f}M"
                if abs(x) >= 1_000_000 else f"${x:,.0f}"
            )
        )
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

st.divider()
st.caption(
    "Educational simulation only. Results are scenario estimates, not forecasts or investment advice."
)
