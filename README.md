# hand_recognition
OpenCV hand recognition  

Run `detect_hand.py` to process the primary webcam stream. Run `web_server.py` to run a REST server that can recieve images and respond with JSON data regarding detected hands.  

Use  
`time curl localhost:8080 --data-binary @image.jpg -vv` to test the web server.
