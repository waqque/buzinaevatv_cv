import cv2
import numpy as np

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
capture.set(cv2.CAP_PROP_EXPOSURE, -4)


def censore(image, size=(5, 5)):
    result = np.zeros_like(image)
    stepy = result.shape[0] // size[0]
    stepx = result.shape[1] // size[1]
    for y in range(0, image.shape[0], stepy):
        for x in range(0, image.shape[1], stepx):
            for c in range(0, image.shape[2]):
                result[y:y + stepy, x:x + stepx, c] = np.mean(image[y:y + stepy, x:x + stepx, c])
    return result


face_cascade = cv2.CascadeClassifier("haarcascade-frontalface-default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade-eye.xml")
eyeglasses = cv2.imread("deal-with-it.png")
glasses, transparent = eyeglasses[:,:,:3], eyeglasses[:,:,-1]
glasses_gray = cv2.cvtColor(glasses, cv2.COLOR_BGR2GRAY)

ret, mask = cv2.threshold(glasses_gray, 0, 255, cv2.THRESH_BINARY_INV)

cv2.destroyAllWindows()
while capture.isOpened():
    ret, frame = capture.read()
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=10)

    for (fx, fy, fw, fh) in faces:
        roii = gray[fy:fy + fh, fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(roii, scaleFactor=1.2, minNeighbors=8)

        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda x: x[0])
            (ex1, ey1, ew1, eh1), (ex2, ey2, ew2, eh2) = eyes[:2]

            eye_center_x1 = fx + ex1 + ew1 + 2
            eye_center_y1 = fy + ey1 + eh1 // 2
            eye_center_x2 = fx + ex2 + ew2 + 2
            eye_center_y2 = fy + ey2 + eh2 // 2

            xof = eye_center_x1 - glasses_width // 2
            yof = eye_center_y1 - glasses_height // 2

            glasses_width = int(abs(eye_center_x2 - eye_center_x1) * 2.3)
            glasses_height = int(glasses_width * (eyeglasses.shape[0] / eyeglasses.shape[1]))

            reglass = cv2.resize(glasses, (glasses_width, glasses_height))
            remask = cv2.resize(mask, (reglass.shape[1], reglass.shape[0]))
            roi = frame[yof:yof + glasses_height, xof:xof+glasses_width]
            bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(remask))
            fg = cv2.bitwise_and(reglass, reglass, mask=remask)

            combined = cv2.add(bg, fg)
            frame[yof:yof + combined.shape[0], xof:xof + combined.shape[1]] = combined


    key = chr(cv2.waitKey(1) & 0xFF)
    if key == "q":
        break

    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()