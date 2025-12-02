import mss
import numpy as np
import cv2
import pytesseract
import win32gui
import win32con
import win32ui
import time

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
custom_config = r'--oem 3 --psm 6 -l pvz'  # Usar nuestro modelo de lenguaje personalizado 'pvz'

def capture_game_window(window_name):
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd:
        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width = right - left
        height = bottom - top

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        img = np.frombuffer(bmpstr, dtype='uint8')
        img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    else:
        print("No se pudo encontrar la ventana del juego.")
        return None

def translate_text(text):
    translations = {
        "hello": "hola",
        "zombie": "zombi",
        # Agrega más traducciones aquí
    }
    return translations.get(text.lower(), text)

def overlay_translated_text(image, translated_texts):
    for idx, text in enumerate(translated_texts):
        position = (10, 30 + idx * 30)  # Cambia esto según donde quieras el texto
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

def main():
    window_name = "PlantsVsZombiesRH"  # Cambia esto según el nombre de tu ventana del juego

    while True:
        img = capture_game_window(window_name)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

            extracted_text = pytesseract.image_to_string(thresh, config=custom_config)
            text_lines = extracted_text.splitlines()

            translated_texts = [translate_text(line) for line in text_lines if line.strip()]

            overlay_translated_text(img, translated_texts)

            cv2.imshow("Overlay", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
