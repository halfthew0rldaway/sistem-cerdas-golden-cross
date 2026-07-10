from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
from werkzeug.utils import secure_filename
import os

app = Flask(__name__, static_folder='dashboard')
app.config['UPLOAD_FOLDER'] = 'data_tester'

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({"message": "File uploaded successfully", "filename": filename})
    return jsonify({"error": "Invalid file type. Only CSV allowed."}), 400

@app.route('/')
def index():
    return send_from_directory('dashboard', 'index.html')

@app.route('/golden_cross_interactive.html')
def serve_chart():
    return send_from_directory('.', 'golden_cross_interactive.html')

@app.route('/api/metrics')
def get_metrics():
    json_path = os.path.join('.', 'metrics_results.json')
    if os.path.exists(json_path):
        return send_from_directory('.', 'metrics_results.json', mimetype='application/json')
    return jsonify({"error": "Belum ada data. Jalankan Pipeline terlebih dahulu."}), 404

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('dashboard', path)

@app.route('/api/run', methods=['POST'])
def run_command():
    data = request.json
    scenario = data.get('scenario')
    target_file = os.path.join(app.config['UPLOAD_FOLDER'], data.get('filename', 'dataset_yfinance_gabungan.csv'))
    
    cmd = []
    if scenario == 'main':
        cmd = ["python", "main_golden_cross.py", "--file", target_file]
    elif scenario == 'compare':
        cmd = ["python", "compare_models.py", "--file", target_file]
    elif scenario == 'custom':
        ma_short = data.get('ma_short', '20')
        ma_long = data.get('ma_long', '100')
        cmd = ["python", "main_golden_cross.py", "--file", target_file, "--ma-short", str(ma_short), "--ma-long", str(ma_long)]
    elif scenario == 'generate':
        cmd = ["python", "generate_dataset_yf.py"]
    else:
        return jsonify({"error": "Unknown scenario"}), 400

    try:
        # Run command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return jsonify({"output": result.stdout})
    except subprocess.CalledProcessError as e:
        return jsonify({"output": e.stdout + "\n" + e.stderr, "error": True}), 500
    except Exception as e:
        return jsonify({"output": str(e), "error": True}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 FLASK WEB GUI BERJALAN!")
    print("🔗 BUKA URL INI DI BROWSER: http://127.0.0.1:5000")
    print("="*50 + "\n")
    app.run(host='127.0.0.1', port=5000)
