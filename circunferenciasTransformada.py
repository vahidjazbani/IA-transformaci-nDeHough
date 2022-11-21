import cv2
import numpy as npy

originalImage = cv2.imread('imagenes/motor.png')
grey = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)


circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=400, minRadius=0, maxRadius=0)

# Draw circles
if circles is not None:
    circles = npy.uint16(npy.around(circles))
    for i in circles[0, :]:
        # external circle
        cv2.circle(originalImage, (i[0], i[1]), i[2], (0, 0, 255), 2)
        # circle middle point
        cv2.circle(originalImage, (i[0], i[1]), 2, (0, 255, 0), 3)

cv2.imshow('clean image edges', grey)
cv2.waitKey(0)
cv2.imshow('Circles detected', originalImage)
cv2.waitKey(0)

