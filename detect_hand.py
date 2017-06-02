#!/usr/bin/env python
import numpy as np
import cv2
import math
import time

cap = cv2.VideoCapture(0)
lo = np.array([0,130,101])
hi = np.array([198,155,148])

version = cv2.__version__.split('.')[0]

# Wrapper function to make versions 2 and 3 behave the same
def findContours(img):
  if version is '3':
    image, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  elif version is '2':
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  return contours, hierarchy

# Wrapper function to make versions 2 and 3 behave the same
def boxPoints(rect):
  if version is '3':
    return cv2.boxPoints(rect)
  elif version is '2':
    return cv2.cv.BoxPoints(rect)

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

# Distance between two cartesian points
def dist(a, b):
  dx = a[0] - b[0]
  dy = a[1] - b[1]
  return (dx**2 + dy**2)**.5

# Translate in an angle for some distance
def move(start, theta, distance):
  x = start[0] + distance * math.cos(theta)
  y = start[1] + distance * math.sin(theta)
  return (int(round(x)), int(round(y)))

# Calculate the area of a triangle given 3 lengths
def heron(a, b, c):
  s = (a + b + c) / 2
  return (s*(s-a)*(s-b)*(s-c)) ** 0.5

def process(frame, imshow=True):
    height, width, num_channels = frame.shape
    center = (height / 2, width / 2)
    if frame is None:
      return {"error": "No image data"}
    # Convert to Y,Cr,Cb color space
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    # Adaptive threshold
#    ret, skin = cv2.threshold(img[:,:,1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Crop out skin areas
    skin = cv2.inRange(img, lo, hi)
    contoursMatrix, hierarchy = findContours(skin)
    cnt, largestContourWithChildren = get_largest_contour_and_children(contoursMatrix, hierarchy)
    
    # get skin color
    mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(mask, largestContourWithChildren, -1, 255, -1)
    avgColor = cv2.mean(frame, mask)
    
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
    # Filter to large defects
#    defects = defects[defects[:,0,3] / 256 > 3]
    
    # Find the defect closest to the center of the image
    minDistFromCenter = 9999
    centralDefect = None

    for i in range(defects.shape[0]):
        # Indicies of the start, end, and far points, and the distance to the far point from the hull edge
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        distance_from_center = dist(far, center)
        if distance_from_center < minDistFromCenter:
          minDistFromCenter = distance_from_center
          centralDefect = far
        if imshow:
          # Draw hull as a green line
          cv2.line(frame, start, end, [0,255,0], 2)
          # Color a circle at the far convexity defect. Black = minimal defect, red = big defect
          #r = float(d) / 30000 * 255
          #cv2.circle(frame, far, 5, [0,0,r], -1)
    
    # Find the palm
    left,top,w,h = cv2.boundingRect(cnt)
    palmRadius = 0
    palmCenter = None
    skip = 4
    
    for x in range(left, left+w, skip):
      for y in range(top, top+h, skip):
        pt = (x,y)
        d = cv2.pointPolygonTest(cnt, pt, True)
        dFromCentralDefect = dist(pt, centralDefect)
        if d > palmRadius and dFromCentralDefect < 130:
          palmRadius = d
          palmCenter = pt

    # Find fingers
    fingers = []
    for i in range(defects.shape[0]):
        # Indicies of the start, end, and far points, and the distance to the far point from the hull edge
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        ds = dist(start, palmCenter)
        de = dist(end, palmCenter)
        df = dist(far, palmCenter)
        d /= 256.0
        if d > palmRadius / 2:
          if ds < 130 and ds > 50:
            fingers.append({"tip": {"x": start[0], "y": start[1]}, "web": {"x": far[0], "y": far[1]}, "d": d})
            if imshow:
              cv2.circle(frame, start, 5, [0,0,255], -1)
    
    if imshow:
      # Create a 100x100 swatch
      avgColorIm = np.array([[avgColor]*100]*100, np.uint8)
      # Draw largest contour in blue
      cv2.drawContours(frame, largestContourWithChildren, -1, (255,0,0), 2)
      # Draw red circle in palm center
      cv2.circle(frame, palmCenter, int(palmRadius), [0,0,255], 2)
      text = "{} fingers".format(len(fingers))
      cv2.putText(frame, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
#     cv2.imshow("skin", skin)
      cv2.imshow("avg color", avgColorIm)
      cv2.imshow("detection", frame)

    return {
      "palm": {"x": palmCenter[0], "y": palmCenter[1], "r": palmRadius},
      "skinColor": {"b": avgColor[0], "g": avgColor[1], "r": avgColor[2]},
      "fingers": fingers
    }

if __name__ == "__main__":
  while(1):
    ret, frame = cap.read()
    s = time.time()
    details = process(frame)
    print(details)
    print(time.time() - s)
    k = cv2.waitKey(20) & 0xff
    if k == 27:
      break
  cap.release()
  cv2.destroyAllWindows()
