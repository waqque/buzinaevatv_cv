import cv2
import numpy as np
import os
vid = cv2.VideoCapture("output (1).avi")
# output_dir = 'out'
# os.makedirs(output_dir, exist_ok=True)
# frame_count = 0
count = 0
while True:
    ret, frame = vid.read()
    if not ret:
        break
    else:
        avg_color = np.average(np.average(np.average(frame, axis=0), axis=0))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 7 and avg_color > 30:
            # cv2.imwrite(f'{output_dir}/frame_{frame_count}_with_count.png', frame)
            # frame_count += 1
            count += 1
print(count)
