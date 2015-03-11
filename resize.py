from PIL import Image
import sys


def resize(imageName, dimension):
  im = Image.open(imageName)
  if (dimension != (0,0)):
    #print("resizing "+str(dimension) + " for "+ imageName)
    im = im.resize(dimension)
    im.save(imageName)

resize(sys.argv[1], (128, 128))