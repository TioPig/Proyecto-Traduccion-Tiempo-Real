import psutil
import time

def find_tesseract_training_process():
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        if process.info['name'] == 'python.exe':
            cmdline = process.info.get('cmdline', [])
            if cmdline and any('train_tesseract_pvz.py' in arg for arg in cmdline):
                return process.info['pid']
    return None

def get_process_output(pid):
    try:
        process = psutil.Process(pid)
        return process.cmdline()[-1]
    except psutil.NoSuchProcess:
        return None

def main():
    while True:
        pid = find_tesseract_training_process()
        if pid:
            output = get_process_output(pid)
            if output:
                print("Tesseract Training Process Output:")
                print(output)
                print("=" * 50)
            else:
                print("No output available from the Tesseract training process.")
        else:
            print("Tesseract training process not found.")
        time.sleep(5)  # Update every 5 seconds

if __name__ == "__main__":
    main()
