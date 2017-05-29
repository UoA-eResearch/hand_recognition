#!/usr/bin/env python
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
lo = np.array([0,130,101])
hi = np.array([198,155,148])

def get_largest_contour_and_children(contoursMatrix, hierarchy):
  maxA = 0
  maxCi = 0
  for i, cnt in enumerate(contoursMatrix):
    area = cv2.contourArea(cnt)
    if area > maxA:
      maxA = area
      maxCi = i
  contours = [contoursMatrix[i] for i,h in enumerate(hierarchy[0]) if h[3] == maxCi or i == maxCi]
  return contoursMatrix[maxCi], contours

def print_color_stats(img):
  ymin, ymax, yminloc, ymaxloc = cv2.minMaxLoc(img[:,:,0], mask)
  crmin, crmax, crminloc, crmaxloc = cv2.minMaxLoc(img[:,:,1], mask)
  cbmin, cbmax, cbminloc, cbmaxloc = cv2.minMaxLoc(img[:,:,2], mask)
  print("min", [ymin, crmin, cbmin])
  print("max", [ymax, crmax, cbmax])

def get_distance_between_pts(a, b):
  dx = a[0] - b[0]
  dy = a[1] - b[1]
  return (dx**2 + dy**2)**.5

while(1):
    ret, frame = cap.read()
    # Convert to Y,Cr,Cb color space
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    # Adaptive threshold
    ret, skin = cv2.threshold(img[:,:,1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Crop out skin areas
#    skin = cv2.inRange(img, lo, hi)
    contoursImage, contoursMatrix, hierarchy = cv2.findContours(skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt, largestContourWithChildren = get_largest_contour_and_children(contoursMatrix, hierarchy)
    
    rect = cv2.minAreaRect(cnt)
#    print(rect)
    
    # Draw largest contour in blue
    cv2.drawContours(frame, largestContourWithChildren, -1, (255,0,0), 2)
    
    # get skin color
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, largestContourWithChildren, -1, 255, -1)
    avgColor = cv2.mean(frame, mask)
    #print(avgColor)
    # Create a 100x100 swatch
    avgColorIm = np.array([[avgColor]*100]*100, np.uint8)
    
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)

    for i in range(defects.shape[0]):
        # Indicies of the start, end, and far points, and the distance to the far point from the hull edge
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        distance = get_distance_between_pts(start, end)
#        r = int(d / 30000.0 * 255)
        # Draw hull as a green line
        cv2.line(frame,start,end,[0,255,0],2)
        # Color a circle at the far convexity defect. Black = minimal defect, red = big defect
        cv2.circle(frame,far,5,[0,0,distance],-1)

    cv2.imshow("skin", skin)
    cv2.imshow("avg color", avgColorIm)
    cv2.imshow("detection", frame)
    k = cv2.waitKey(50) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()
