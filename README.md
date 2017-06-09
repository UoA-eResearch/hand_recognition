# hand_recognition
OpenCV hand recognition  

### Installation

`sudo pip install -r requirements.txt`

### Running

Run `detect_hand.py` to process the primary webcam stream. Run `web_server.py` to run a REST server that can recieve images and respond with JSON data regarding detected hands. Both single request POSTs as well as websockets are supported. The server responds with a json object, with position/radius (in pixels) of the palm, as well as a list of the objects describing the positions and angles (in degrees) of the fingers. It also guesses the gesture.  

See https://handrecognition.herokuapp.com/ for a running example.  

Sample response:
```json
{
   "palm":{
      "y":340,
      "x":325,
      "r":31.304951684997057
   },
   "skinColor":{
      "r":128.51116352201257,
      "b":88.66261792452829,
      "g":92.63246855345912
   },
   "gesture":"Gun",
   "fingers":[
      {
         "distance_to_next_finger":91.285267157411553,
         "web":{
            "y":302,
            "x":320
         },
         "angle":27.62266626608903,
         "tip":{
            "y":260,
            "x":328
         },
         "d":34.859375
      },
      {
         "distance_to_next_finger":71.421285342676384,
         "web":{
            "y":319,
            "x":288
         },
         "angle":64.65494617518769,
         "tip":{
            "y":308,
            "x":242
         },
         "d":25.1484375
      }
   ]
}
```

Use  
`time curl localhost:8080 --data-binary @image.jpg -vv` to test the web server. Visit http://localhost:8080 to stream your webcam from your browser to the server, displaying the result on a canvas overlay.  

### Supported hand gestures

Flat palm, Closed Fist, Pointing, Thumbs up, Peace, Gun, Claw, Open Palm  
