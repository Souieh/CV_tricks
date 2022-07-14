import cv2 
import numpy as np
import platform , os


def clump(n,m,M):
  return min(max(n, m), M)



def brightness_to_char(gray):
  brightness = " .,:;*08# " #" `'^.,:;i!I-~+?1tfjXYUJCLQ0OZ*#MW&8%B@$"
  unit = len(brightness)/255
  char_im = ""
  for line in gray:
    char_ln=""
    for v in line :
      char_ln+=brightness[clump( int(v * unit)  , 0 , len(brightness)-1)  ]
    char_im += char_ln+"\n"
  return char_im

def to_ascii( infile , outfile = "" , width=80):
  
  im = cv2.imread(infile)

  gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
  (h, w) = im.shape[:2]

  gray =cv2.resize(gray , (width ,int(width*h/w)) )

  char_im = brightness_to_char(gray)

  if outfile=="":
    outfile = infile.split(".")[0] + "_ascii.txt"

  with open(outfile, "w") as f:
    f.write(char_im)
    f.close


  if platform.system() == "Windows":
    os.system('cls')
  else:
    os.system('clear')

  print(char_im) 

to_ascii(".\examples\python.png",outfile = ".\examples\python_ascii.txt",width=40)