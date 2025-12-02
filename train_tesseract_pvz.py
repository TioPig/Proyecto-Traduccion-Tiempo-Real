import subprocess
import os
import logging
import shutil
import random
import json
import time
import io
import unicodedata
import traceback
from datetime import datetime, timedelta
from tqdm import tqdm
from logging.handlers import RotatingFileHandler
from PIL import Image, ImageDraw, ImageFont, ImageColor
from colors import background_colors, text_colors

# Crear carpeta para logs
logs_folder = 'logs'
os.makedirs(logs_folder, exist_ok=True)

# Configuración del logging principal
logger = logging.getLogger('current_logger')
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(os.path.join(logs_folder, 'tesseract_training_current.log'), mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logging.basicConfig(filename=os.path.join(logs_folder, 'tesseract_training_current.log'), 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w', 
                    encoding='utf-8')

# Configuración del logging histórico
historical_logger = logging.getLogger('historical')
historical_logger.setLevel(logging.INFO)
handler = RotatingFileHandler(os.path.join(logs_folder, 'tesseract_training_historical.log'), 
                              maxBytes=1000000, 
                              backupCount=5, 
                              encoding='utf-8')
handler.setFormatter(formatter)
historical_logger.addHandler(handler)

# Función para registrar en ambos logs
def log_info(message):
    logger.info(message)
    historical_logger.info(message)

def log_error(message):
    logger.error(message)
    historical_logger.error(message)
    log_error_to_file(message)

def log_error_to_file(error_message):
    with open('error_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"{datetime.now()} - {error_message}\n")


# Rutas y configuraciones
tesseract_path = r'C:\Program Files\Tesseract-OCR'
output_folder = 'tesseract_output'
os.makedirs(output_folder, exist_ok=True)

# Lista de fuentes
fonts_folder = r'C:\Users\talol\Desktop\Proyecto Traduccion Tiempo Real\Fuentes'
fonts = [os.path.join(fonts_folder, f) for f in os.listdir(fonts_folder) if f.endswith('.ttf')]

def run_command(command):
    log_info(f"Ejecutando comando: {command}")
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    log_info(f"Salida: {result.stdout}")
    if result.returncode != 0:
        log_error(f"Error (código {result.returncode}): {result.stderr}")
    return result

def find_file(filename, search_dirs):
    for directory in search_dirs:
        filepath = os.path.join(directory, filename)
        if os.path.exists(filepath):
            return filepath
    return None

def process_cedict(file_path):
    chinese_characters = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split(' ')
            if len(parts) > 1:
                chinese_characters.update(parts[0])
    return list(chinese_characters)

def generate_training_data():
    with open('training_text.txt', 'r', encoding='utf-8') as f:
        training_text = f.read().splitlines()

    chinese_chars = process_cedict('cedict_1_0_ts_utf-8_mdbg.txt')
    training_text.extend(chinese_chars)

    lines_per_image = 25
    image_width = 1600

    font_sizes = [9, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56]

    total_iterations = len(fonts) * len(font_sizes) * (len(training_text) // lines_per_image)
    start_time = time.time()

    with tqdm(total=total_iterations, desc="Generando datos de entrenamiento") as pbar:
        for font_path in fonts:
            font_name = os.path.basename(font_path).split('.')[0][:3]
            log_info(f"Generando datos de entrenamiento para {font_name}")
           
            for font_size in font_sizes:
                line_height = font_size + 4
                image_height = lines_per_image * line_height
               
                for i in range(0, len(training_text), lines_per_image):
                    text_block = training_text[i:i+lines_per_image]
                   
                    subdir = os.path.join(output_folder, f"{font_name}_{font_size}")
                    os.makedirs(subdir, exist_ok=True)
                   
                    file_name = f"p{i//lines_per_image:04d}"
                    image_path = os.path.join(subdir, f"{file_name}.png")
                    box_path = os.path.join(subdir, f"{file_name}.box")
                   
                    color_index = (i // lines_per_image) % len(background_colors)
                    bg_color = background_colors[color_index]
                    text_color = text_colors[(color_index + 1) % len(text_colors)]
                   
                    image = Image.new('RGB', (image_width, image_height), color=bg_color)
                   
                    if random.choice([True, False]):
                        draw = ImageDraw.Draw(image)
                        for x in range(0, image_width, 10):
                            for y in range(0, image_height, 10):
                                draw.point((x, y), fill='lightgray')
                   
                    draw = ImageDraw.Draw(image)
                    font = ImageFont.truetype(font_path, font_size)
                   
                    for j, line in enumerate(text_block):
                        draw.text((10, j * line_height), line, font=font, fill=text_color)
                   
                    image.save(image_path, format='PNG')
                   
                    with open(box_path, 'w', encoding='utf-8') as box_file:
                        for j, line in enumerate(text_block):
                            for k, char in enumerate(line):
                                if char.strip():
                                    left = 10 + k * (font_size // 2)
                                    top = j * line_height
                                    right = left + font_size
                                    bottom = top + line_height
                                    box_file.write(f"{char} {left} {top} {right} {bottom} 0\n")
                   
                    pbar.update(1)
                    progress_percentage = (pbar.n / total_iterations) * 100
                    save_progress('data_generation', 'generate_training_data', {
                        'progress_percentage': round(progress_percentage, 2)
                    })

                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time * total_iterations / pbar.n
                    remaining_time = estimated_total_time - elapsed_time
                    log_info(f"Progreso: {pbar.n}/{total_iterations} ({pbar.n/total_iterations*100:.2f}%) - "
                             f"Tiempo transcurrido: {timedelta(seconds=int(elapsed_time))} - "
                             f"Tiempo estimado restante: {timedelta(seconds=int(remaining_time))}")

    log_info("Generación de datos de entrenamiento completada")
    save_progress('data_generation', 'completed')

def load_progress():
    try:
        with open('progress.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {'last_completed_stage': 'start', 'substage': None, 'details': None}

def save_progress(stage, substage=None, details=None, max_retries=5, delay=1):
    progress = {
        'last_completed_stage': stage,
        'substage': substage,
        'details': details
    }
    for attempt in range(max_retries):
        try:
            with open('progress.json', 'w') as f:
                json.dump(progress, f)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                log_error(f"Failed to save progress after {max_retries} attempts")
                raise

def stage_process_unicharset():
    log_info("Procesando y combinando unicharset")

    box_files = []
    for root, dirs, files in os.walk(output_folder):
        box_files.extend([os.path.join(root, f) for f in files if f.endswith('.box')])
    
    batch_size = 100
    total_batches = (len(box_files) + batch_size - 1) // batch_size
    start_time = time.time()

    with tqdm(total=total_batches, desc="Procesando unicharset") as pbar:
        for i in range(0, len(box_files), batch_size):
            batch = box_files[i:i+batch_size]
            unicharset_cmd = [
                'unicharset_extractor.exe',
                '--output_unicharset', os.path.join(output_folder, f'pvz_{i//batch_size}.unicharset')
            ] + batch
           
            try:
                if run_command(unicharset_cmd).returncode != 0:
                    raise Exception(f"Error en el lote {i//batch_size}")
            except Exception as e:
                log_error(f"Error en unicharset_extractor: {str(e)}")
                save_progress('training', 'process_unicharset', {'progress': i, 'total_batches': total_batches})
                return False
           
            pbar.update(1)
            save_progress('training', 'process_unicharset', {'progress': ((i + batch_size)/100), 'total_batches': total_batches})
            
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * total_batches / pbar.n
            remaining_time = estimated_total_time - elapsed_time
            log_info(f"Progreso: {pbar.n}/{total_batches} ({pbar.n/total_batches*100:.2f}%) - "
                     f"Tiempo transcurrido: {timedelta(seconds=int(elapsed_time))} - "
                     f"Tiempo estimado restante: {timedelta(seconds=int(remaining_time))}")

    unicharset_files = [f for f in os.listdir(output_folder) if f.endswith('.unicharset')]
    
    if not unicharset_files:
        log_error("No se encontraron archivos unicharset en el directorio de salida")
        return False
    
    if 'pvz_0.unicharset' not in unicharset_files:
        log_error("No se encontró el archivo pvz_0.unicharset")
        return False

    def is_chinese_char(char):
        return '\u4e00' <= char <= '\u9fff'

    combined_chars = set()
    for unicharset_file in unicharset_files:
        with open(os.path.join(output_folder, unicharset_file), 'r', encoding='utf-8') as f:
            lines = f.readlines()[1:]  # Saltar la primera línea (conteo de caracteres)
            combined_chars.update(char for char in (line.split()[0] for line in lines) if is_chinese_char(char))

    # Escribir el unicharset combinado y limpiado
    combined_unicharset_path = os.path.join(output_folder, 'pvz.unicharset')
    with open(combined_unicharset_path, 'w', encoding='utf-8') as f:
        f.write(f"{len(combined_chars)}\n")
        for char in sorted(combined_chars):
            f.write(f"{char} {ord(char)} {unicodedata.category(char)} 0\n")

    # Crear archivos necesarios
    open(os.path.join(output_folder, 'pvz.config'), 'w').close()
    with open(os.path.join(output_folder, 'radical-stroke.txt'), 'w', encoding='utf-8') as f:
        f.write("# Placeholder for radical-stroke data\n")

    log_info(f"Combinación y limpieza de unicharset completada. Total de caracteres chinos: {len(combined_chars)}")
    save_progress('training', 'unicharset_combined_and_cleaned')
    return True

def stage_generate_font_properties():
    box_files = []
    for root, dirs, files in os.walk(output_folder):
        box_files.extend([os.path.join(root, f) for f in files if f.endswith('.box')])

    start_time = time.time()

    with tqdm(total=len(box_files), desc="Generando font_properties") as pbar:
        try:
            with open(os.path.join(output_folder, 'font_properties'), 'w') as f:
                for box_file in box_files:
                    relative_path = os.path.relpath(box_file, output_folder)
                    f.write(f'{os.path.splitext(relative_path)[0]} 0 0 0 0 0\n')
                    pbar.update(1)
                    save_progress('training', 'generate_font_properties', {'progress': pbar.n, 'total': len(box_files)})
                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time * len(box_files) / pbar.n
                    remaining_time = estimated_total_time - elapsed_time
                    log_info(f"Progreso: {pbar.n}/{len(box_files)} ({pbar.n/len(box_files)*100:.2f}%) - "
                             f"Tiempo transcurrido: {timedelta(seconds=int(elapsed_time))} - "
                             f"Tiempo estimado restante: {timedelta(seconds=int(remaining_time))}")
        except Exception as e:
            log_error(f"Error al generar font_properties: {str(e)}")
            save_progress('training', 'generate_font_properties', {'error': str(e)})
            return False
   
    save_progress('training', 'font_properties_generated')
    return True

def stage_generate_tr_files():
    box_files = []
    for root, dirs, files in os.walk(output_folder):
        box_files.extend([os.path.join(root, f) for f in files if f.endswith('.box')])

    tr_files = []
    start_time = time.time()

    with tqdm(total=len(box_files), desc="Generando archivos .tr") as pbar:
        for box_file in box_files:
            base_name = os.path.splitext(box_file)[0]
            image_file = f"{base_name}.png"
            tr_cmd = [
                'tesseract.exe',
                image_file,
                base_name,
                'nobatch', 'box.train'
            ]
            try:
                if run_command(tr_cmd).returncode != 0:
                    raise Exception(f"Error al generar .tr para {base_name}")
            except Exception as e:
                log_error(f"Error al generar .tr para {base_name}: {str(e)}")
                save_progress('training', 'generate_tr_files', {'progress': pbar.n, 'total': len(box_files)})
                return False
            tr_files.append(f"{base_name}.tr")
            pbar.update(1)
            save_progress('training', 'generate_tr_files', {'progress': pbar.n, 'total': len(box_files)})
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * len(box_files) / pbar.n
            remaining_time = estimated_total_time - elapsed_time
            log_info(f"Progreso: {pbar.n}/{len(box_files)} ({pbar.n/len(box_files)*100:.2f}%) - "
                     f"Tiempo transcurrido: {timedelta(seconds=int(elapsed_time))} - "
                     f"Tiempo estimado restante: {timedelta(seconds=int(remaining_time))}")
   
    save_progress('training', 'tr_files_generated')
    return True

def stage_complete_shapeclustering():
    log_info("Ejecutando shapeclustering")
    tr_files = []
    for root, dirs, files in os.walk(output_folder):
        tr_files.extend([os.path.join(root, f) for f in files if f.endswith('.tr')])

    batch_size = 100
    total_batches = (len(tr_files) + batch_size - 1) // batch_size
    start_time = time.time()

    with tqdm(total=total_batches, desc="Ejecutando shapeclustering") as pbar:
        for i in range(0, len(tr_files), batch_size):
            batch = tr_files[i:i+batch_size]
            shape_cmd = [
                'shapeclustering.exe',
                '-F', os.path.join(output_folder, 'font_properties'),
                '-U', os.path.join(output_folder, 'pvz.unicharset'),
                '-O', os.path.join(output_folder, f'pvz.shapetable.{i//batch_size}')
            ] + batch

            try:
                if run_command(shape_cmd).returncode != 0:
                    raise Exception(f"Error en el lote {i//batch_size}")
            except Exception as e:
                log_error(f"Error en shapeclustering: {str(e)}")
                save_progress('training', 'complete_shapeclustering', {'progress': i, 'total_batches': total_batches})
                return False

            pbar.update(1)
            save_progress('training', 'complete_shapeclustering', {'progress': ((i + batch_size)/100), 'total_batches': total_batches})
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * total_batches / pbar.n
            remaining_time = estimated_total_time - elapsed_time
            log_info(f"Progreso: {pbar.n}/{total_batches} ({pbar.n/total_batches*100:.2f}%) - "
                     f"Tiempo transcurrido: {timedelta(seconds=int(elapsed_time))} - "
                     f"Tiempo estimado restante: {timedelta(seconds=int(remaining_time))}")

    save_progress('training', 'shapeclustering_completed')
    return True

def stage_run_mftraining():
    log_info("Ejecutando mftraining")
    tr_files = []
    for root, dirs, files in os.walk(output_folder):
        tr_files.extend([os.path.join(root, f) for f in files if f.endswith('.tr')])

    font_properties_path = os.path.join(output_folder, 'font_properties')
    if not os.path.exists(font_properties_path):
        with open(font_properties_path, 'w') as f:
            f.write("Not 0 0 0 0 0\n")

    xheights_path = os.path.join(output_folder, 'xheights')
    if not os.path.exists(xheights_path):
        with open(xheights_path, 'w') as f:
            f.write("Not 20\n")

    batch_size = 80
    max_batches = 500
    total_batches = min(max_batches, (len(tr_files) + batch_size - 1) // batch_size)
    start_time = time.time()

    with tqdm(total=total_batches, desc="Ejecutando mftraining") as pbar:
        for i in range(0, min(len(tr_files), (max_batches * batch_size)), batch_size):
            batch = tr_files[i:i+batch_size]
            mf_cmd = [
                'mftraining.exe',
                '-F', font_properties_path,
                '-X', xheights_path,
                '-U', os.path.join(output_folder, 'pvz.unicharset'),
                '-O', os.path.join(output_folder, f'pvz.{i//batch_size}.'),
                '-D', output_folder
            ] + batch

            try:
                result = run_command(mf_cmd)
                if result.returncode != 0:
                    log_error(f"Error en mftraining para el lote {i//batch_size}")
                    log_error(f"Salida estándar: {result.stdout}")
                    log_error(f"Salida de error: {result.stderr}")
                    raise Exception(f"Error en el lote {i//batch_size}")
               
                log_info(f"Procesado lote {i//batch_size} exitosamente")
                log_info(f"Salida del lote {i//batch_size}: {result.stdout}")

            except Exception as e:
                log_error(f"Excepción en mftraining: {str(e)}")
                log_error(f"Detalles del error: {traceback.format_exc()}")
                save_progress('training', 'run_mftraining', {'progress': i, 'total_batches': total_batches, 'error': str(e)})
                return False

            pbar.update(1)
            save_progress('training', 'run_mftraining', {'progress': i + batch_size, 'total_batches': total_batches})
            
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time * total_batches / pbar.n
            remaining_time = estimated_total_time - elapsed_time
            log_info(f"Progreso: {pbar.n}/{total_batches} ({pbar.n/total_batches*100:.2f}%) - "
                     f"Tiempo transcurrido: {timedelta(seconds=int(elapsed_time))} - "
                     f"Tiempo estimado restante: {timedelta(seconds=int(remaining_time))}")

    save_progress('training', 'mftraining_completed')
    return True


def stage_run_cntraining():
    log_info("Ejecutando cntraining")
    
    cn_cmd = [
        'cntraining.exe',
        '-F', os.path.join(output_folder, 'font_properties'),
        '-U', os.path.join(output_folder, 'pvz.unicharset'),
        '-O', os.path.join(output_folder, 'pvz.'),
        '-D', output_folder
    ]
    
    try:
        if run_command(cn_cmd).returncode != 0:
            raise Exception("Error al ejecutar cntraining")
        log_info("cntraining completado con éxito")
    except Exception as e:
        log_error(f"Error en cntraining: {str(e)}")
        save_progress('training', 'run_cntraining', {'error': 'Error en cntraining'})
        return False
    
    save_progress('training', 'cntraining_completed')
    return True

def write_file_list(files, list_file_path):
    with open(list_file_path, 'w') as f:
        for file in files:
            f.write(f"{file}\n")

def stage_rename_files():
    log_info("Renombrando archivos")
    files_to_rename = ['inttemp', 'normproto', 'pffmtable']
    start_time = time.time()

    with tqdm(total=len(files_to_rename), desc="Renombrando archivos") as pbar:
        for file in files_to_rename:
            src = find_file(file, [os.getcwd(), output_folder])
            if src:
                dst = os.path.join(output_folder, f'pvz.{file}')
                try:
                    shutil.move(src, dst)
                    log_info(f"Archivo {file} renombrado a {dst}")
                except Exception as e:
                    log_error(f"Error al renombrar {file}: {str(e)}")
                    save_progress('training', 'rename_files', {'error': f"Error al renombrar {file}"})
                    return False
            else:
                log_error(f"Archivo {file} no encontrado")
                save_progress('training', 'rename_files', {'error': f"Archivo {file} no encontrado"})
                return False
            pbar.update(1)
            save_progress('training', 'rename_files', {'progress': pbar.n, 'total': len(files_to_rename)})
            elapsed_time = time.time() - start_time
            log_info(f"Progreso: {pbar.n}/{len(files_to_rename)} ({pbar.n/len(files_to_rename)*100:.2f}%) - "
                     f"Tiempo transcurrido: {timedelta(seconds=int(elapsed_time))}")
    save_progress('training', 'files_renamed')
    return True

def stage_combine_training_data():
    log_info("Combinando datos de entrenamiento")
    start_time = time.time()

    combine_cmd = [
        'combine_tessdata.exe',
        os.path.join(output_folder, 'pvz.')
    ]
    try:
        result = run_command(combine_cmd).returncode == 0
        if not result:
            raise Exception("Error al combinar datos de entrenamiento")
    except Exception as e:
        log_error(f"Error al combinar datos de entrenamiento: {str(e)}")
        save_progress('training', 'combine_training_data', {'error': str(e)})
        return False

    elapsed_time = time.time() - start_time
    log_info(f"Combinación de datos completada - Tiempo transcurrido: {timedelta(seconds=int(elapsed_time))}")
    
    save_progress('training', 'data_combined')
    return True

def resume_training(substage):
    stages = [
        'process_unicharset',
        'generate_font_properties',
        'generate_tr_files',
        'complete_shapeclustering',
        'run_mftraining',
        'run_cntraining',
        'rename_files',
        'combine_training_data'
    ]
   
    stage_functions = {
        'process_unicharset': stage_process_unicharset,
        'generate_font_properties': stage_generate_font_properties,
        'generate_tr_files': stage_generate_tr_files,
        'complete_shapeclustering': stage_complete_shapeclustering,
        'run_mftraining': stage_run_mftraining,
        'run_cntraining': stage_run_cntraining,
        'rename_files': stage_rename_files,
        'combine_training_data': stage_combine_training_data
    }

    start_index = stages.index(substage) if substage in stages else 0

    for stage in stages[start_index:]:
        log_info(f"Ejecutando etapa: {stage}")
        try:
            if not stage_functions[stage]():
                raise Exception(f"Fallo en la etapa: {stage}")
        except Exception as e:
            log_error(f"Error en la etapa {stage}: {str(e)}")
            save_progress('training', stage, {'error': str(e)})
            return False
        save_progress('training', stage)

    save_progress('training_completed')
    log_info("Proceso de entrenamiento completado con éxito")
    return True

def main():
    progress = load_progress()
    current_stage = progress['last_completed_stage']
    current_substage = progress['substage']

    if current_stage == 'training_completed':
        log_info("Todos los procesos han sido completados")
        return

    if current_stage in ['start', 'data_generated']:
        if current_stage == 'start':
            log_info("Iniciando generación de datos de entrenamiento")
            generate_training_data()
            save_progress('data_generated')
        log_info("Iniciando proceso de entrenamiento de Tesseract")
        resume_training('process_unicharset')
    else:
        log_info(f"Reanudando entrenamiento desde la sub-etapa: {current_substage}")
        resume_training(current_substage)

if __name__ == "__main__":
    main()

