#!/usr/bin/env python
import detect_hand
import os
import numpy as np
import cv2
import json
from bottle import *

BaseRequest.MEMFILE_MAX = 1e8

def read_image(binary_data):
  img_array = np.asarray(binary_data, dtype=np.uint8)
  image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  if image_data is None:
    raise Exception('Unable to decode posted image!')
  return image_data

@get('/')
def default_get():
    return static_file("index.html", ".")

@post('/')
def process_image():
  try:
    if request.files.get('pic'):
      binary_data = request.files.get('pic').file.read()
    else:
      binary_data = request.body.read()
    binary_data = bytearray(binary_data)
    print("recieved image of size {}".format(len(binary_data)))
    image_data = read_image(binary_data)
    s = time.time()
    data = detect_hand.process(image_data)
    print("Processed in {}s".format(time.time() - s))
    return data
  except Exception as e:
    print("Error: {}".format(e))
    response.status = 500
    return {'error': str(e)}

port = int(os.environ.get('PORT', 8080))

if __name__ == "__main__":
  run(host='0.0.0.0', port=port, debug=True, server='gunicorn', workers=4)

app = default_app()
