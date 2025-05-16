from flask import Flask, render_template, request
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
import base64
import os

app = Flask(__name__)

# Exit valuation scenarios
exit_values = [0, 50_000_000, 100_000_000, 350_000_000, 500_000_000,
               1_000_000_000, 5_000_000_000, 10_000_000_000]

# Scenario with no follow-on
def simulate_scenario(fund_size, num_checks, check_size, valuation, exit_probs):
    ownership = check_size / valuation
    outcomes = np.random.choice(exit_values, size=10000 * num_checks, p=exit_probs)
    dpi_matrix = (ownership * outcomes) / check_size
    dpi_matrix = dpi_matrix.reshape(-1, num_checks)
    dpi_totals = dpi_matrix.sum(axis=1)
    return dpi_totals

# Scenario with follow-on
def simulate_with_follow_on(fund_size, num_initial, initial_check, init_valuation,
                             num_follow, follow_check, follow_valuation, exit_probs):
    initial_ownership = initial_check / init_valuation
    follow_ownership = follow_check / follow_valuation

    total_investments = num_initial + num_follow
    outcomes = np.random.choice(exit_values, size=10000 * total_investments, p=exit_probs)
    dpi_matrix = outcomes.reshape(-1, total_investments)

    init_returns = dpi_matrix[:, :num_initial] * initial_ownership / initial_check
    follow_returns = dpi_matrix[:, num_initial:] * follow_ownership / follow_check
    dpi_totals = init_returns.sum(axis=1) + follow_returns.sum(axis=1)
    return dpi_totals

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        form = request.form
        inputs = form.to_dict()

        # Exit probability assumptions
        probs = [float(form.get(f'exit_{i}', 0)) / 100 for i in range(8)]
        total_prob = sum(probs)
        if total_prob == 0:
            probs = [1 / 8] * 8  # fallback
        else:
            probs = [p / total_prob for p in probs]  # normalize to 100%

        # Scenario 1 Inputs
        s1_fund = float(form.get('s1_fund_size', 0))
        s1_checks = int(form.get('s1_num_checks', 0))
        s1_check_size = float(form.get('s1_check_size', 0))
        s1_valuation = float(form.get('s1_valuation', 1))
        s1_follow_on = 's1_follow_toggle' in form

        # Scenario 2 Inputs
        s2_fund = float(form.get('s2_fund_size', 0))
        s2_checks = int(form.get('s2_num_checks', 0))
        s2_check_size = float(form.get('s2_check_size', 0))
        s2_valuation = float(form.get('s2_valuation', 1))
        s2_follow_on = 's2_follow_toggle' in form

        # Run Scenario 1 Simulation
        if s1_follow_on:
            s1_follow_num = int(form.get('s1_follow_num', 0))
            s1_follow_size = float(form.get('s1_follow_size', 0))
            s1_follow_valuation = float(form.get('s1_follow_valuation', 1))
            dpi1 = simulate_with_follow_on(s1_fund, s1_checks, s1_check_size, s1_valuation,
                                           s1_follow_num, s1_follow_size, s1_follow_valuation, probs)
        else:
            dpi1 = simulate_scenario(s1_fund, s1_checks, s1_check_size, s1_valuation, probs)

        # Run Scenario 2 Simulation
        if s2_follow_on:
            s2_follow_num = int(form.get('s2_follow_num', 0))
            s2_follow_size = float(form.get('s2_follow_size', 0))
            s2_follow_valuation = float(form.get('s2_follow_valuation', 1))
            dpi2 = simulate_with_follow_on(s2_fund, s2_checks, s2_check_size, s2_valuation,
                                           s2_follow_num, s2_follow_size, s2_follow_valuation, probs)
        else:
            dpi2 = simulate_scenario(s2_fund, s2_checks, s2_check_size, s2_valuation, probs)

        # DPI Histogram
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(dpi1, bins=50, alpha=0.6, label='Scenario 1')
        ax.hist(dpi2, bins=50, alpha=0.6, label='Scenario 2')
        ax.set_title('DPI Distribution Comparison')
        ax.set_xlabel('DPI')
        ax.set_ylabel('Count')
        ax.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        dpi_plot = base64.b64encode(buf.read()).decode('utf-8')

        # Summary Stats
        df = pd.DataFrame({'Scenario 1': dpi1, 'Scenario 2': dpi2})
        stats_table = df.describe().to_html(classes='table table-bordered')

        return render_template('index.html',
                               inputs=inputs,
                               dpi_plot_url=f'data:image/png;base64,{dpi_plot}',
                               stats_table=stats_table)

    return render_template('index.html', inputs={})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
