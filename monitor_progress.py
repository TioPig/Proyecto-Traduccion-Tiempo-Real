from flask import Flask, render_template, jsonify
import json
from datetime import timedelta
from tqdm import tqdm

app = Flask(__name__)

progress_json_path = 'progress.json'

def get_progress_data():
    all_stages = [
        'start',
        'data_generation',
        'data_generated',
        'training',
        'process_unicharset',
        'combine_unicharset',
        'generate_font_properties',
        'generate_tr_files',
        'complete_shapeclustering',
        'run_mftraining',
        'run_cntraining',
        'rename_files',
        'combine_training_data',
        'training_completed'
    ]

    try:
        with open(progress_json_path, 'r') as f:
            progress = json.load(f)
        
        last_completed_stage = progress.get('last_completed_stage', 'Unknown')
        current_substage = progress.get('substage', 'Unknown')
        details = progress.get('details', {})
        
        current_index = all_stages.index(last_completed_stage) if last_completed_stage in all_stages else -1
        completed_stages = all_stages[:current_index + 1]
        
        current_stage = current_substage if current_substage != 'Unknown' else all_stages[current_index + 1] if current_index + 1 < len(all_stages) else 'Completed'
        
        pending_stages = all_stages[all_stages.index(current_stage) + 1:] if current_stage in all_stages else []
        
        total_value = next((details[key] for key in ['total', 'total_batches', 'total_iterations'] if key in details), 100)
        progress_value = details.get('progress', 0)
        percentage = (progress_value / total_value) * 100  # Ajustado a 90% del valor original
        
        elapsed_time = details.get('elapsed_time', 0)
        remaining_time = details.get('remaining_time', 'Unknown')
        
        elapsed_time_str = str(timedelta(seconds=int(elapsed_time)))
        remaining_time_str = str(timedelta(seconds=int(remaining_time))) if remaining_time != 'Unknown' else 'Unknown'
        
        return {
            'current_stage': current_stage,
            'substage': current_substage,
            'completed_stages': completed_stages,
            'pending_stages': pending_stages,
            'percentage': round(percentage, 2),
            'elapsed_time': elapsed_time_str,
            'remaining_time': remaining_time_str,
            'details': details
        }
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'current_stage': 'Unknown',
            'substage': 'Unknown',
            'completed_stages': [],
            'pending_stages': all_stages,
            'percentage': 0,
            'elapsed_time': "Unknown",
            'remaining_time': "Unknown",
            'details': {}
        }



@app.route('/')
def index():
    return render_template('progress.html', **get_progress_data())

@app.route('/update_progress')
def update_progress():
    return jsonify(get_progress_data())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3391)
