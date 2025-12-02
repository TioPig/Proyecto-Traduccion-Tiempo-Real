import json
import os
import re

def detect_stage_from_log():
    log_path = r'C:\Users\talol\Desktop\Proyecto Traduccion Tiempo Real\logs\tesseract_training_current.log'
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as log_file:
            lines = log_file.readlines()
            last_lines = lines[-6:]
            for line in reversed(last_lines):
                if "Generación de datos de entrenamiento completada" in line:
                    return 'data_generated', 'completed', None
                if "Generando datos de entrenamiento para" in line:
                    font = line.split("para ")[-1].strip()
                    return 'data_generation', 'fonts', {'current_font': font}
                if "Imagen y archivo .box generados para" in line:
                    parts = line.split(" - ")
                    font = parts[0].split("para ")[-1]
                    size = parts[1].split("tamaño ")[-1]
                    block = parts[2].split("bloque ")[-1]
                    return 'data_generation', 'image_generation', {'font': font, 'size': size, 'block': block}
                if "Procesando unicharset" in line:
                    batch = re.search(r'lote (\d+)', line)
                    return 'training', 'unicharset', {'batch': batch.group(1) if batch else None}
                if "Generando font_properties" in line:
                    return 'training', 'font_properties', None
                if "Generando archivo .tr para" in line:
                    file_name = line.split("para ")[-1].strip()
                    return 'training', 'generating_tr_files', {'current_file': file_name}
                if "Ejecutando shapeclustering" in line:
                    return 'training', 'shapeclustering', None
                if "Ejecutando mftraining" in line:
                    return 'training', 'mftraining', None
                if "Ejecutando cntraining" in line:
                    return 'training', 'cntraining', None
                if "Renombrando archivos" in line:
                    return 'training', 'renaming_files', None
                if "Combinando datos de entrenamiento" in line:
                    return 'training', 'combining_data', None
                if "Proceso de entrenamiento completado con éxito" in line:
                    return 'training_completed', None, None
    return 'unknown', None, None

def update_progress():
    stage, substage, details = detect_stage_from_log()
    progress = {
        'last_completed_stage': stage,
        'substage': substage,
        'details': details
    }
    with open('progress.json', 'w') as f:
        json.dump(progress, f)
    return progress

def load_checkpoint():
    if os.path.exists('progress.json'):
        with open('progress.json', 'r') as f:
            progress = json.load(f)
    else:
        progress = {}

    if 'last_completed_stage' not in progress:
        stage, substage, details = detect_stage_from_log()
        progress['last_completed_stage'] = stage
        progress['substage'] = substage
        progress['details'] = details
    
    progress.setdefault('substage', None)
    progress.setdefault('details', None)
    return progress

def create_checkpoint(current_stage, substage=None, details=None):
    print("Creando checkpoint de progreso")
    
    checkpoint_data = {
        'last_completed_stage': current_stage,
        'substage': substage,
        'details': details
    }
    
    with open('progress.json', 'w') as f:
        json.dump(checkpoint_data, f)
    
    print(f"Checkpoint creado: Etapa {current_stage}, Sub-etapa {substage}")
    if details:
        print(f"Detalles: {details}")

if __name__ == "__main__":
    current_progress = update_progress()
    print("Estado actual del proceso:")
    print(f"Etapa: {current_progress['last_completed_stage']}")
    print(f"Sub-etapa: {current_progress['substage']}")
    if current_progress['details']:
        print("Detalles:")
        for key, value in current_progress['details'].items():
            print(f"  - {key.replace('_', ' ').capitalize()}: {value}")
