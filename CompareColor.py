import pylab
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import cv
import math
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
#
# more appropriate for color images
#
def ciede2000Diff(orig, other):
    ref = cv2.imread(orig)
    lr, ar, br = cv2.split(cv2.cvtColor(ref, cv.CV_BGR2Lab))
    labref = cv2.merge((lr, ar, br))
    img = cv2.imread(other)
    li, ai, bi = cv2.split(cv2.cvtColor(img, cv.CV_BGR2Lab))
    labimg = cv2.merge((li, ai, bi))

    height, width,_ = ref.shape
    deltaE = 0
    for y in range(height):
        for x in range(width):
            lr, ar, br = labref[y,x]
            reflabcolor = LabColor(lr, ar, br)
            li, ai, bi = labimg[y,x]
            imglabcolor = LabColor(li, ai, bi)
            deltaE = deltaE +   delta_e_cie2000(reflabcolor, imglabcolor)
    return deltaE / (height * width)

def labDiff(orig, other):
    ref = cv2.imread(orig)
    lr, ar, br = cv2.split(cv2.cvtColor(ref, cv.CV_BGR2Lab))
    labref = cv2.merge((lr, ar, br))
    img = cv2.imread(other)
    li, ai, bi = cv2.split(cv2.cvtColor(img, cv.CV_BGR2Lab))
    labimg = cv2.merge((li, ai, bi))
    height, width,_ = ref.shape
    lmse = 0
    amse = 0
    bmse = 0
    for y in range(height):
        for x in range(width):
            refl,refa,refb = labref[y,x]
            imgl, imga, imgb = labimg[y,x]
            difl = refl.astype(float) - imgl.astype(float)
            difa = refa.astype(float) - imga.astype(float)
            difb = refb.astype(float) - imgb.astype(float)
            lmse = lmse + math.pow(difl , 2)
            amse = amse + math.pow(difa , 2)
            bmse = bmse + math.pow(difb , 2)
    lmse = lmse / (height * width)
    amse = amse  / (height * width)
    bmse = bmse  / (height * width)

    mse = (lmse + amse + bmse) / 3
    if (mse == 0):
        return "identical"
    psnr = 10 * math.log(pow(255, 2) / mse, 10)
    return psnr
    #return mse

def psnr(orig, other):
    ref = cv2.imread(orig,cv2.IMREAD_COLOR)
    img = cv2.imread(other,cv2.IMREAD_COLOR)
    height, width,_ = ref.shape
    rmse = 0
    gmse = 0
    bmse = 0
    for y in range(height):
        for x in range(width):
            refr,refg,refb = ref[y,x]
            imgr, imgg, imgb = img[y,x]
            rdif = refr.astype(float) - imgr.astype(float)
            gdif = refg.astype(float) - imgg.astype(float)
            bdif = refb.astype(float) - imgb.astype(float)
            rmse = rmse + math.pow(rdif , 2)
            gmse = gmse + math.pow(gdif , 2)
            bmse = bmse + math.pow(bdif , 2)
    rmse = rmse / (height * width)
    gmse = gmse  / (height * width)
    bmse = bmse  / (height * width)

    mse = (rmse + gmse + bmse) / 3
    if (mse == 0):
        return "identical"
    psnr = 10 * math.log(pow(255, 2) / mse, 10)
    return psnr
