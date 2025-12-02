import os
import sys
from tqdm import tqdm
import time
from datetime import timedelta
import random
import subprocess
from PIL import Image, ImageDraw, ImageFont
from itertools import product

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, 'Libs')
sys.path.append(lib_dir)

from Libs.colors import TEXT_COLORS, BG_COLORS
from Libs.log_config import setup_logging
from Libs.progress_tracker import ProgressTracker, Stage, StageStatus, ScriptStatus

# Configuración
FONTS_DIR = 'Fonts'
fonts = [os.path.join(FONTS_DIR, f) for f in os.listdir(FONTS_DIR) if f.endswith('.ttf')]
OUTPUT_DIR = 'output'
FONT_SIZES = [9, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56]

# Asegúrate de que el directorio de salida exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializar el logger y el tracker de progreso
logger = setup_logging()
tracker = ProgressTracker()

def main_workflow():
    def placeholder_function():
        print("Función en proceso")

    stages = [
        (Stage.GENERATE_TRAINING_DATA, generate_training_data),
        (Stage.PROCESS_UNICHARSET, process_unicharset),
        (Stage.GENERATE_FONT_PROPERTIES, placeholder_function),
        (Stage.CREATE_TR_FILES, placeholder_function),
        (Stage.RUN_SHAPECLUSTERING, placeholder_function),
        (Stage.RUN_MFTRAINING, placeholder_function),
        (Stage.RUN_CNTRAINING, placeholder_function),
        (Stage.RENAME_FILES, placeholder_function),
        (Stage.COMBINE_TRAINING_DATA, placeholder_function)
    ]

    current_progress = tracker.load_progress()
    start_index = 0

    if current_progress:
        start_index = next((i for i, (stage, _) in enumerate(stages) if stage == current_progress.stage), 0)

    for i in range(start_index, len(stages)):
        stage, stage_function = stages[i]
        current_progress = tracker.load_progress()
        
        if current_progress and current_progress.stage == stage and current_progress.stage_status == StageStatus.FINISHED:
            logger.info(f"Etapa {stage.value} ya completada. Avanzando a la siguiente.")
            continue
        
        logger.info(f"Iniciando etapa: {stage.value}")
        stage_function()
        
        tracker.update_progress(
            stage=stage,
            stage_status=StageStatus.FINISHED,
            processed_data=1,
            total_data=1,
            script_status=ScriptStatus.ACTIVE
        )
        logger.info(f"Etapa completada: {stage.value}")

    logger.info("Proceso de entrenamiento completado.")

def generate_training_data():
    current_progress = tracker.load_progress()
    if current_progress and current_progress.stage == Stage.GENERATE_TRAINING_DATA and current_progress.stage_status == StageStatus.FINISHED:
        logger.info("La generación de datos de entrenamiento ya está completada. Pasando a la siguiente etapa.")
        return

    with open('training_dictionary.txt', 'r', encoding='utf-8') as f:
        training_text = f.read().splitlines()

    lines_per_image = 25
    image_width = 1600
    font_sizes = [9, 12, 16, 20, 24, 28, 32, 36, 40, 48, 56]

    total_iterations = len(fonts) * len(font_sizes) * (len(training_text) // lines_per_image)
    start_time = time.time()

    with tqdm(total=total_iterations, desc="Generando datos de entrenamiento") as pbar:
        for font_path in fonts:
            font_name = os.path.basename(font_path).split('.')[0][:3]
            logger.info(f"Generando datos de entrenamiento para {font_name}")
            
            for font_size in font_sizes:
                line_height = font_size + 4
                image_height = lines_per_image * line_height
                
                for i in range(0, len(training_text), lines_per_image):
                    text_block = training_text[i:i+lines_per_image]
                    
                    subdir = os.path.join(OUTPUT_DIR, f"{font_name}_{font_size}")
                    os.makedirs(subdir, exist_ok=True)
                    
                    file_name = f"pvz{i//lines_per_image:05d}"
                    image_path = os.path.join(subdir, f"{file_name}.png")
                    box_path = os.path.join(subdir, f"{file_name}.box")
                    
                    color_index = (i // lines_per_image) % len(BG_COLORS)
                    bg_color = BG_COLORS[color_index]
                    text_color = TEXT_COLORS[(color_index + 1) % len(TEXT_COLORS)]
                    
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
                    tracker.update_progress(
                        stage=Stage.GENERATE_TRAINING_DATA,
                        stage_status=StageStatus.STARTED,
                        processed_data=pbar.n,
                        total_data=total_iterations,
                        script_status=ScriptStatus.ACTIVE
                    )

                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time * total_iterations / pbar.n
                    remaining_time = estimated_total_time - elapsed_time
                    logger.info(f"Progreso: {pbar.n}/{total_iterations} ({pbar.n/total_iterations*100:.2f}%) - "
                                f"Tiempo transcurrido: {timedelta(seconds=int(elapsed_time))} - "
                                f"Tiempo estimado restante: {timedelta(seconds=int(remaining_time))}")

    logger.info("Generación de datos de entrenamiento completada")
    tracker.update_progress(
        stage=Stage.GENERATE_TRAINING_DATA,
        stage_status=StageStatus.FINISHED,
        processed_data=total_iterations,
        total_data=total_iterations,
        script_status=ScriptStatus.ACTIVE
    )

def process_unicharset():
    current_progress = tracker.load_progress()
    if current_progress and current_progress.stage == Stage.PROCESS_UNICHARSET and current_progress.stage_status == StageStatus.FINISHED:
        logger.info("El procesamiento de unicharset ya está completado. Pasando a la siguiente etapa.")
        return

    logger.info("Iniciando procesamiento de unicharset")
    
    output_dir = os.path.join(OUTPUT_DIR, 'pvz_unicharset_files')
    os.makedirs(output_dir, exist_ok=True)
    
    box_dirs = [d for d in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, d))]
    
    for box_dir in tqdm(box_dirs, desc="Procesando directorios"):
        box_files = [f for f in os.listdir(os.path.join(OUTPUT_DIR, box_dir)) if f.startswith('pvz') and f.endswith('.box')]
        
        if not box_files:
            continue
        
        unicharset_file = os.path.join(output_dir, f"pvz_unicharset_{box_dir}")
        box_file_paths = [os.path.join(OUTPUT_DIR, box_dir, f) for f in box_files]
        
        command = [
            "unicharset_extractor",
            "--output_unicharset", unicharset_file,
            "--norm_mode", "1"
        ] + box_file_paths
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            logger.info(f"Unicharset generado para {box_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error al procesar {box_dir}: {e}")
    
    final_unicharset = os.path.join(OUTPUT_DIR, 'pvz_final_unicharset')
    unicharset_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.startswith('pvz_unicharset_')]
    
    command = [
        "unicharset_extractor",
        "--output_unicharset", final_unicharset,
        "--norm_mode", "1"
    ] + unicharset_files
    
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("Unicharset final generado exitosamente")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error al generar el unicharset final: {e}")
    
    logger.info("Procesamiento de unicharset completado")
    tracker.update_progress(
        stage=Stage.PROCESS_UNICHARSET,
        stage_status=StageStatus.FINISHED,
        processed_data=1,
        total_data=1,
        script_status=ScriptStatus.ACTIVE
    )


if __name__ == "__main__":
    logger.info("Iniciando el script principal")
    print("Iniciando proceso de entrenamiento...")
    main_workflow()
    print("Proceso completado.")
