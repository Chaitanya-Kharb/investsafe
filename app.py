# app.py
# Main Streamlit web app

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # needed for streamlit

from simulator   import get_stock_data, run_simulation, calculate_metrics, get_time_analysis
from ai_explainer import explain_risk

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title = "InvestSafe — Know Before You Invest",
    page_icon  = "📈",
    layout     = "wide"
)

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.title("📈 InvestSafe")
st.markdown("#### Know your risk *before* you invest your money")
st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR — USER INPUTS
# ─────────────────────────────────────────
st.sidebar.title("⚙️ Configure Your Investment")
st.sidebar.markdown("Fill in the details below")

# Stock selection
stock_options = {
    "Reliance Industries" : "RELIANCE.NS",
    "TCS"                 : "TCS.NS",
    "Infosys"             : "INFY.NS",
    "HDFC Bank"           : "HDFCBANK.NS",
    "Wipro"               : "WIPRO.NS",
    "Tata Motors"         : "TATAMOTORS.NS",
    "Bajaj Finance"       : "BAJFINANCE.NS",
    "Nifty 50 ETF"        : "NIFTYBEES.NS"
}

selected_stock = st.sidebar.selectbox(
    "Select a Stock",
    options = list(stock_options.keys())
)

# Investment amount slider
investment_amount = st.sidebar.slider(
    "Investment Amount (₹)",
    min_value  = 1000,
    max_value  = 100000,
    value      = 10000,
    step       = 1000,
    format     = "₹%d"
)

# Time period selection
time_period = st.sidebar.selectbox(
    "Investment Duration",
    options = ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "3 Years"]
)

# Map time period to days
days_map = {
    "1 Month"  : 21,
    "3 Months" : 63,
    "6 Months" : 126,
    "1 Year"   : 252,
    "2 Years"  : 504,
    "3 Years"  : 756
}

# Number of simulations
num_sims = st.sidebar.select_slider(
    "Number of Simulations",
    options = [500, 1000, 2000, 5000],
    value   = 1000
)

# Analyze button
analyze_btn = st.sidebar.button(
    "🔍 Analyze Risk",
    use_container_width = True
)

st.sidebar.markdown("---")
st.sidebar.markdown("*Built with Monte Carlo Simulation + Groq AI*")

# ─────────────────────────────────────────
# MAIN CONTENT — runs when button clicked
# ─────────────────────────────────────────
if analyze_btn:

    ticker = stock_options[selected_stock]
    days   = days_map[time_period]

    # ── Loading spinner while fetching data ──
    with st.spinner(f"Fetching {selected_stock} data..."):
        mean, std, closes = get_stock_data(ticker)
        annual_vol        = std * np.sqrt(252) * 100

    # ── Run simulation ──
    with st.spinner(f"Running {num_sims} simulations..."):
        final_values = run_simulation(mean, std,
                                      investment_amount,
                                      days, num_sims)
        metrics      = calculate_metrics(final_values, investment_amount)

    # ── Display success message ──
    st.success(f"Analysis complete for {selected_stock}!")

    # ─────────────────────────────────────────
    # ROW 1 — 4 KEY METRIC CARDS
    # ─────────────────────────────────────────
    st.markdown("### 📊 Key Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        loss_color = "🔴" if metrics['loss_probability'] > 50 else "🟡" if metrics['loss_probability'] > 30 else "🟢"
        st.metric(
            label = f"{loss_color} Loss Probability",
            value = f"{metrics['loss_probability']}%"
        )

    with col2:
        gain_loss = metrics['avg_final'] - investment_amount
        arrow     = "↑" if gain_loss > 0 else "↓"
        st.metric(
            label = "💰 Average Final Value",
            value = f"₹{metrics['avg_final']:,.0f}",
            delta = f"{arrow} ₹{abs(gain_loss):,.0f}"
        )

    with col3:
        st.metric(
            label = "🚀 Best Case",
            value = f"₹{metrics['best_case']:,.0f}"
        )

    with col4:
        st.metric(
            label = "⚠️ Worst Case",
            value = f"₹{metrics['worst_case']:,.0f}"
        )

    st.markdown("---")

    # ─────────────────────────────────────────
    # ROW 2 — 2 CHARTS SIDE BY SIDE
    # ─────────────────────────────────────────
    st.markdown("### 📉 Simulation Charts")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**All Simulated Futures**")

        # Run full path simulation for chart
        all_paths = np.zeros((days, num_sims))
        for i in range(num_sims):
            price = investment_amount
            for d in range(days):
                r              = np.random.normal(mean, std)
                price          = price * (1 + r)
                all_paths[d,i] = price

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        fig1.patch.set_facecolor('#1A1D27')
        ax1.set_facecolor('#1A1D27')
        ax1.plot(all_paths, alpha=0.03, color='#1D9E75', linewidth=0.5)
        ax1.plot(all_paths.mean(axis=1), color='orange',
                 linewidth=2.5, label='Average path')
        ax1.axhline(y=investment_amount, color='red',
                    linewidth=1.5, linestyle='--',
                    label=f'Invested ₹{investment_amount:,}')
        ax1.set_xlabel("Trading Days", color='white')
        ax1.set_ylabel("Portfolio Value (₹)", color='white')
        ax1.tick_params(colors='white')
        ax1.legend(facecolor='#1A1D27', labelcolor='white')
        ax1.spines['bottom'].set_color('#444')
        ax1.spines['left'].set_color('#444')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)
        plt.close()

    with chart_col2:
        st.markdown("**Distribution of Final Values**")

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        fig2.patch.set_facecolor('#1A1D27')
        ax2.set_facecolor('#1A1D27')

        # Color bars differently — loss vs gain
        n, bins, patches = ax2.hist(final_values, bins=50,
                                     edgecolor='none', alpha=0.85)
        for patch, left_edge in zip(patches, bins):
            if left_edge < investment_amount:
                patch.set_facecolor('#E24B4A')   # red for loss
            else:
                patch.set_facecolor('#1D9E75')   # green for gain

        ax2.axvline(x=investment_amount, color='white',
                    linewidth=2, linestyle='--',
                    label=f'Your ₹{investment_amount:,}')
        ax2.set_xlabel("Final Portfolio Value (₹)", color='white')
        ax2.set_ylabel("Number of Simulations", color='white')
        ax2.tick_params(colors='white')
        ax2.legend(facecolor='#1A1D27', labelcolor='white')
        ax2.spines['bottom'].set_color('#444')
        ax2.spines['left'].set_color('#444')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")

    # ─────────────────────────────────────────
    # ROW 3 — RISK OVER TIME TABLE
    # ─────────────────────────────────────────
    st.markdown("### ⏱️ How Risk Changes Over Time")

    with st.spinner("Calculating risk across time periods..."):
        time_results = get_time_analysis(mean, std, investment_amount)

    import pandas as pd

    time_df = pd.DataFrame([
        {
            "Time Period"        : period,
            "Loss Probability"   : f"{data['loss_probability']}%",
            "Average Final Value": f"₹{data['avg_final']:,.0f}",
            "Best Case"          : f"₹{data['best_case']:,.0f}",
            "Worst Case"         : f"₹{data['worst_case']:,.0f}",
            "Avg Return"         : f"{data['avg_return_pct']}%"
        }
        for period, data in time_results.items()
    ])

    st.dataframe(time_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ─────────────────────────────────────────
    # ROW 4 — AI EXPLANATION
    # ─────────────────────────────────────────
    st.markdown("### 🤖 AI Risk Explanation")

    with st.spinner("Generating AI explanation..."):
        explanation = explain_risk(
            stock_name        = selected_stock,
            investment_amount = investment_amount,
            time_period       = time_period,
            loss_probability  = metrics['loss_probability'],
            avg_final_value   = metrics['avg_final'],
            annual_volatility = annual_vol,
            mean_daily_return = mean
        )

    st.info(explanation)

    # ─────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────
    st.markdown("---")
    st.caption("⚠️ This app is for educational purposes only. Not financial advice. "
               "Past performance does not guarantee future results.")

else:
    # ── Welcome screen shown before clicking analyze ──
    st.markdown("## 👋 Welcome to InvestSafe")
    st.markdown("""
    This app helps young investors **understand risk before investing.**

    **How it works:**
    1. Select a stock from the sidebar
    2. Enter your investment amount
    3. Choose how long you want to invest
    4. Click **Analyze Risk**

    The app will run **1000 simulations** of possible futures
    and show you realistic outcomes — best case, worst case,
    and most likely — explained in plain language by AI.
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🎲 Monte Carlo Simulation")
        st.markdown("Runs 1000 possible futures using real historical stock data")
    with col2:
        st.markdown("#### 📊 Visual Risk Analysis")
        st.markdown("See exactly where your money could end up in clear charts")
    with col3:
        st.markdown("#### 🤖 AI Explanation")
        st.markdown("Get a plain-language explanation of your risk — no jargon")