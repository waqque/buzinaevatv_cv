import cv2

video = cv2.VideoCapture("output (1).avi")

cnt_img = 0
while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 6:
        cv2.imshow("Gray", thresh)
        cnt_img += 1
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
print(cnt_img)
