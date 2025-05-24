import cv2
import numpy as np
import pyautogui
import time
import mss
 
GAME_REGION = {
    'left': 98,     
    'top': 316,     
    'width': 987 - 98,  
    'height': 511 - 316  
}

NEAR_OFFSET_BASE = 160      
NEAR_WIDTH = 55
NEAR_HEIGHT = 40
NEAR_PIXEL_THRESHOLD = 120

FAR_OFFSET_BASE = 280    
FAR_WIDTH = 90
FAR_HEIGHT = 50
FAR_PIXEL_THRESHOLD = 150  

JUMP_COOLDOWN = 0.5       

OFFSET_SPEED_COEFF = 0.8   
PIXEL_THRESHOLD_COEFF = 1.0

def get_dynamic_offset(score, base_offset):
    offset = base_offset - int(score * OFFSET_SPEED_COEFF)
    return max(offset, 50)  

def detect_obstacle(img, threshold):
    if img.size == 0:
        return False, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    return cv2.countNonZero(binary) > threshold, 0

def jump_with_duck():
    pyautogui.press('space')
    time.sleep(0.1)
    pyautogui.keyDown('down')
    time.sleep(0.08)
    pyautogui.keyUp('down')

def main():
    sct = mss.mss()
    print("Запуск через 2 секунды...")
    time.sleep(2)
    
    score = 0
    last_jump_time = 0
    cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)

    while True:
        try:
            full_img = np.array(sct.grab(GAME_REGION))
            
            near_offset = get_dynamic_offset(score, NEAR_OFFSET_BASE)
            far_offset = get_dynamic_offset(score, FAR_OFFSET_BASE)
            
            near_x1, near_y1 = near_offset, (GAME_REGION['height'] - NEAR_HEIGHT) //2 
            near_x2, near_y2 = near_x1 + NEAR_WIDTH, near_y1 + NEAR_HEIGHT
            
            far_x1, far_y1 = far_offset, (GAME_REGION['height'] - FAR_HEIGHT) // 2
            far_x2, far_y2 = far_x1 + FAR_WIDTH, far_y1 + FAR_HEIGHT

            # Проверка границ
            if any(coord < 0 for coord in [near_x1, near_y1, far_x1, far_y1]):
                continue

            # Анализ зон
            near_zone = full_img[near_y1:near_y2, near_x1:near_x2]
            far_zone = full_img[far_y1:far_y2, far_x1:far_x2]
            
            near_detected, _ = detect_obstacle(near_zone, NEAR_PIXEL_THRESHOLD)
            far_detected, _ = detect_obstacle(far_zone, FAR_PIXEL_THRESHOLD)

            # Логика прыжка
            current_time = time.time()
            if (near_detected or far_detected) and (current_time - last_jump_time) > JUMP_COOLDOWN:
                jump_with_duck()
                last_jump_time = current_time

            debug_img = full_img.copy()
            cv2.rectangle(debug_img, (near_x1, near_y1), (near_x2, near_y2), (0, 0, 255), 1)
            cv2.rectangle(debug_img, (far_x1, far_y1), (far_x2, far_y2), (0, 255, 0), 1)
          
            cv2.imshow('Debug', debug_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.001)
            score+=1 

        except Exception as e:
            print(f"Ошибка: {e}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()