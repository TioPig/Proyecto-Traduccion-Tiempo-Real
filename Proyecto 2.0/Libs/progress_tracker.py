import json
from dataclasses import dataclass, asdict
from enum import Enum
import os

class Stage(Enum):
    GENERATE_TRAINING_DATA = "Generación de datos de entrenamiento"
    PROCESS_UNICHARSET = "Procesamiento de unicharset"
    GENERATE_FONT_PROPERTIES = "Generación de archivos de propiedades de fuentes"
    CREATE_TR_FILES = "Creación de archivos .tr"
    RUN_SHAPECLUSTERING = "Ejecución de shapeclustering"
    RUN_MFTRAINING = "Ejecución de mftraining"
    RUN_CNTRAINING = "Ejecución de cntraining"
    RENAME_FILES = "Renombrado de archivos"
    COMBINE_TRAINING_DATA = "Combinación de datos de entrenamiento"

class StageStatus(Enum):
    STARTED = "Empezó"
    FINISHED = "Terminó"

class ScriptStatus(Enum):
    ACTIVE = "Activo"
    ERROR = "Error"

@dataclass
class ProgressDetail:
    processed_data: int
    total_data: int
    script_status: ScriptStatus

@dataclass
class Progress:
    stage: Stage
    stage_status: StageStatus
    detail: ProgressDetail

class ProgressTracker:
    def __init__(self, save_path='progress.json'):
        self.save_path = save_path
        self.progress_file = save_path
        self.progress = None

    def update_progress(self, stage: Stage, stage_status: StageStatus,
                        processed_data: int, total_data: int,
                        script_status: ScriptStatus):
        self.progress = Progress(
            stage=stage,
            stage_status=stage_status,
            detail=ProgressDetail(
                processed_data=processed_data,
                total_data=total_data,
                script_status=script_status
            )
        )
        self._save_progress()

    def _save_progress(self):
        with open(self.save_path, 'w') as f:
            json.dump(asdict(self.progress), f, indent=2, default=str)

    def load_progress(self):
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                data = json.load(f)
            try:
                stage_value = data['stage']
                stage = next((s for s in Stage if s.name == stage_value or s.value == stage_value), None)
                if stage is None:
                    stage = next((s for s in Stage if stage_value in s.name or stage_value in s.value), Stage.GENERATE_TRAINING_DATA)
                return Progress(
                    stage=stage,
                    stage_status=StageStatus(data['stage_status']),
                    detail=ProgressDetail(
                        processed_data=data['detail']['processed_data'],
                        total_data=data['detail']['total_data'],
                        script_status=ScriptStatus(data['detail']['script_status'])
                    )
                )
            except (KeyError, ValueError):
                return Progress(
                    stage=Stage.GENERATE_TRAINING_DATA,
                    stage_status=StageStatus.STARTED,
                    detail=ProgressDetail(
                        processed_data=0,
                        total_data=0,
                        script_status=ScriptStatus.ACTIVE
                    )
                )
        return None

    def get_progress(self):
        return self.progress
