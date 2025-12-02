from flask import Flask, render_template, jsonify
from dataclasses import asdict
from enum import Enum
from datetime import timedelta
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'Libs')
sys.path.append(lib_dir)
from Libs.progress_tracker import ProgressTracker, Stage, StageStatus, ScriptStatus
app = Flask(__name__)

progress_json_path = 'progress.json'
progress_tracker = ProgressTracker(progress_json_path)

def get_progress_data():
    progress = progress_tracker.load_progress()
    
    if progress is None:
        return {
            'current_stage': 'No iniciado',
            'stage_status': 'Desconocido',
            'completed_stages': [],
            'pending_stages': [stage.value for stage in Stage],
            'percentage': 0,
            'script_status': 'Desconocido',
            'details': {}
        }
    
    current_stage = progress.stage.value
    stage_status = progress.stage_status.value
    script_status = progress.detail.script_status.value
    
    all_stages = [stage.value for stage in Stage]
    current_index = all_stages.index(current_stage)
    completed_stages = all_stages[:current_index]
    pending_stages = all_stages[current_index + 1:]
    
    processed_data = progress.detail.processed_data
    total_data = progress.detail.total_data
    percentage = (processed_data / total_data) * 100 if total_data > 0 else 0
    
    return {
        'current_stage': current_stage,
        'stage_status': stage_status,
        'completed_stages': completed_stages,
        'pending_stages': pending_stages,
        'percentage': round(percentage, 2),
        'script_status': script_status,
        'details': {
            'processed_data': processed_data,
            'total_data': total_data,
            'script_status': script_status
        }
    }


@app.route('/')
def index():
    return render_template('progress.html', **get_progress_data())

@app.route('/update_progress')
def update_progress():
    return jsonify(get_progress_data())

if __name__ == '__main__':
    app.run(
        debug=True,
        host='0.0.0.0',
        port=3391
    )

