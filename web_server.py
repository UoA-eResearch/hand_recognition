#!/usr/bin/env python
import detect_hand
import os
import numpy as np
import cv2
import json
from bottle import *
from gevent.pywsgi import WSGIServer
from geventwebsocket import WebSocketError
from geventwebsocket.handler import WebSocketHandler

BaseRequest.MEMFILE_MAX = 1e8
app = Bottle()

def read_image(binary_data):
  img_array = np.asarray(binary_data, dtype=np.uint8)
  image_data = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  if image_data is None:
    raise Exception('Unable to decode posted image!')
  return image_data

def read_data_uri(uri):
  encoded_data = uri.split(',')[1]
  nparr = np.fromstring(encoded_data.decode('base64'), np.uint8)
  img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return img

@app.get('/')
def default_get():
    return static_file("index.html", ".")

@app.post('/')
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

@app.route('/websocket')
def handle_websocket():
  wsock = request.environ.get('wsgi.websocket')
  if not wsock:
    abort(400, 'Expected WebSocket request.')

  while True:
    try:
      message = wsock.receive()
      if message:
        print("Got message of len {}".format(len(message)))
        if type(message) is bytearray:
          image_data = read_image(message)
        else:
          image_data = read_data_uri(message)
        s = time.time()
        data = detect_hand.process(image_data)
        print("Processed in {}s".format(time.time() - s))
        wsock.send(json.dumps(data))
    except WebSocketError:
      break
    except:
      wsock.send("error")

if __name__ == "__main__":
  port = int(os.environ.get('PORT', 8080))
  print("Starting server on http://localhost:{}".format(port))
  server = WSGIServer(("0.0.0.0", port), app,
                      handler_class=WebSocketHandler)
  server.serve_forever()
