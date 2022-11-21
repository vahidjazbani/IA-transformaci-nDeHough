import cv2
import numpy as npy

originalImage = cv2.imread('imagenes/estadioFutbol.png')
grey = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grey, 100, 300, apertureSize=3, L2gradient=False)
lines = cv2.HoughLinesP(edges, 1, npy.pi / 180, 40, minLineLength=1, maxLineGap=35)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # draw lines
        cv2.line(originalImage, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

cv2.imshow('clean image edges', edges)
cv2.waitKey()
cv2.imshow('lines detected from original image', originalImage)
cv2.waitKey()
