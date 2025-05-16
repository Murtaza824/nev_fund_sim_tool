from flask import Flask, render_template, request, send_file
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd
import tempfile
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    summary_stats = None
    csv_data = None
    saved_inputs = request.form if request.method == 'POST' else {}

    if request.method == 'POST':
        def parse_scenario(prefix):
            return {
                'fund_size': float(request.form.get(f'{prefix}_fund_size', 0)),
                'num_checks': int(request.form.get(f'{prefix}_num_checks', 0)),
                'initial_check_size': float(request.form.get(f'{prefix}_initial_check_size', 0)),
                'initial_valuation': float(request.form.get(f'{prefix}_initial_valuation', 1)),
                'follow_on': request.form.get(f'{prefix}_follow_on') == 'on',
                'num_follow_ons': int(request.form.get(f'{prefix}_num_follow_ons') or 0),
                'follow_on_check_size': float(request.form.get(f'{prefix}_follow_on_check_size') or 0),
                'follow_on_valuation': float(request.form.get(f'{prefix}_follow_on_valuation') or 1),
            }

        s1 = parse_scenario('s1')
        s2 = parse_scenario('s2')

        exit_vals = [0, 50e6, 100e6, 350e6, 500e6, 1e9, 5e9, 10e9]
        probs = [float(request.form.get(f'prob_{int(v/1e6)}', 0)) / 100 for v in exit_vals]
        probs = np.array(probs) / sum(probs) if sum(probs) > 0 else np.ones(len(exit_vals)) / len(exit_vals)

        dilution = 0.80
        followon_dilution = 0.40
        num_simulations = 10000

        def simulate(s):
            ownership = (s['initial_check_size'] / s['initial_valuation']) * dilution
            follow_ownership = (s['follow_on_check_size'] / s['follow_on_valuation']) * followon_dilution if s['follow_on'] else 0
            returns = []
            for _ in range(num_simulations):
                initial = sum(np.random.choice(exit_vals, s['num_checks'], p=probs)) * ownership
                follow = sum(np.random.choice(exit_vals, s['num_follow_ons'], p=probs)) * follow_ownership if s['follow_on'] else 0
                returns.append(initial + follow)
            return np.array(returns) / (s['fund_size'] * 0.78)  # DPI = Return / Invested Capital

        dpi1 = simulate(s1)
        dpi2 = simulate(s2)

        # Create histogram
        fig, ax = plt.subplots(figsize=(20,7), dpi=150)
        ax.hist(dpi1, bins=100, alpha=0.6, label='Scenario 1')
        ax.hist(dpi2, bins=100, alpha=0.6, label='Scenario 2')
        ax.set_xlabel('DPI')
        ax.set_ylabel('Count')
        ax.set_title('DPI Distribution Comparison')
        ax.legend()
        ax.grid(True)

        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Save to temp CSV
        csv_df = pd.DataFrame({"Scenario 1": dpi1, "Scenario 2": dpi2})
        summary_stats = csv_df.describe().round(2).to_html(classes='summary-table')
        temp_csv = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        csv_df.to_csv(temp_csv.name, index=False)
        csv_data = os.path.basename(temp_csv.name)

        # Save chart image
        img_path = os.path.join(tempfile.gettempdir(), "dpi_chart.png")
        with open(img_path, 'wb') as f:
            f.write(base64.b64decode(plot_url))

    return render_template('index.html', plot_url=plot_url, stats_table=summary_stats, csv_data=csv_data, inputs=saved_inputs)

@app.route('/download/<filename>')
def download_csv(filename):
    path = os.path.join(tempfile.gettempdir(), filename)
    return send_file(path, as_attachment=True)

@app.route('/chart')
def chart_popup():
    path = os.path.join(tempfile.gettempdir(), "dpi_chart.png")
    return send_file(path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
