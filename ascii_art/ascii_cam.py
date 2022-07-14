import cv2 
import os , platform 
import keyboard
import numpy as np


clearCommand = "clear"
if platform.system() == "Windows":
  clearCommand = "cls"


buffersize = 60
# Create a VideoCapture object and read from input camera 
cap = cv2.VideoCapture(0)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

ratio =1/ frame_height* frame_width 




def clump(n,m,M):
  return min(max(n, m), M)


def brightness_to_char(gray):
  brightness = " `.,:;*08#" # or   " `^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
  unit = len(brightness)/255
  char_im = ""
  for line in gray:
    char_ln=""
    for v in line :
      char_ln+=brightness[clump( int(v * unit)  , 0 , len(brightness)-1)  ]
    char_im += char_ln+"\n"
  return char_im

while (True):
    # Grab a single frame of video
  ret, frame = cap.read()

  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  gray =cv2.resize(gray , (int(buffersize*ratio),buffersize))
  gray =cv2.flip(gray ,1)

  char_im = brightness_to_char(gray)

  os.system(clearCommand)
  print(char_im) 
  if keyboard.is_pressed("q"):
    break

