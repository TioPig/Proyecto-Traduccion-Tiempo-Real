import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def setup_logging():
    # Crear directorio de logs si no existe
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Configurar el logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Formato de los logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Log activo (se reinicia en cada ejecución)
    active_log = logging.FileHandler(f'{log_dir}/active.log', mode='w')
    active_log.setFormatter(formatter)
    logger.addHandler(active_log)

    # Log histórico (no se reinicia)
    historical_log = RotatingFileHandler(f'{log_dir}/historical.log', maxBytes=10*1024*1024, backupCount=5)
    historical_log.setFormatter(formatter)
    logger.addHandler(historical_log)

    # Log de errores
    error_log = logging.FileHandler(f'{log_dir}/error.log')
    error_log.setLevel(logging.ERROR)
    error_log.setFormatter(formatter)
    logger.addHandler(error_log)

    return logger
