import argparse
import sys
import os
import glob
from PIL import Image
from scipy import spatial
import leargist
import copy
import sys
import math

from more_itertools import grouper
from phash import cross_correlation, image_digest

class ImageComparator(object):

    def __init__(self):
        self.threshold = 0.0

    def __compute_gist(self, imagename):
        im = Image.open(imagename)
        im = im.convert('LA')
        g = leargist.color_gist(im)
        return g

    def __correlate_gist(silf, gist_1, gist_2):
        return spatial.distance.euclidean(gist_1, gist_2)

    def compareDirectoryImagesByGist(self, image, path2, threshold):
        failures=0
        total=0
        #files = glob.glob(image+"/*.jpg")
        #if (len(files) == 0):
         #   print("No matching jpg files under "+ image)
          #  return 0
        if (path2 == ""):
            path2 = copy.copy(image);
        files2 = glob.glob(path2+"/*.jpg")
        if (len(files2) == 0):
            print("No matching jpg files under "+ path2)
            return 0

        iterations=1
        records = len(files2)
        log_f = sys.stdout
        print('**************** %s%s%s%d%s' % ('Compare ', image, " images against ", records, path2))
        phashList = []
        #iterations=10
        complete = 1
        #complete = iterations / 10
        success_pct = 0
        #for i, img_1 in enumerate(files):
        if (total != 0):
          success_pct = (total-failures)*100/total
       # if (i % complete == 0):
        #    print('%s\t%d\t%s\t%d' % ('current iteration ', i, "success_pct", success_pct), file=sys.stdout)
        gist_1 = self.__compute_gist( image)
        for img_2 in files2:
            if (image == img_2):
                continue
            total = total + 1
            gist_2 = self.__compute_gist(img_2)
            pcc = self.__correlate_gist(gist_1, gist_2)
            phashList.append((pcc, img_2))
            if pcc <= threshold:
                status = 'pass'
                #log_f = sys.stdout
            else:
                failures += 1
                status = 'fail'
                #log_f = sys.stderr
            #                 failures += 1

                #print('%s\t%s\t%s\t%0.3f' % (status, image, img_2, pcc), file=log_f)
        success_pct = (total-failures)*100/total
        print('%s\t%s\t%d\t%s\t%d\t%s\t%d' % (image, 'total', total, "failures", failures, ' success_pct ', success_pct))
        phashList.sort(reverse = False, key = lambda pcc:pcc[0])
        #print(phashList)
        return failures, total, phashList

    def compareDirectoryImagesByPhash(self, inputFileName, path2, threshold):
        failures=0
        total=0
        files2 = glob.glob(path2+"/*.jpg")
        if (len(files2) == 0):
            print("No matching jpg files under "+ path2)
            return 0, 0, []

        iterations=1
        records = len(files2)
        log_f = sys.stdout
        print('**************** %s%s%s%d%s' % ('Compare ', inputFileName, " images against ", records, path2))

        phashList = []
        #iterations=10
        complete = 1
        #complete = iterations / 10
        success_pct = 0
        #for i, img_1 in enumerate(files):
        if (total != 0):
          success_pct = (total-failures)*100/total
        digest_1 = image_digest(inputFileName)
       # if (i % complete == 0):
        #    print('%s\t%d\t%s\t%d' % ('current iteration ', i, "success_pct", success_pct), file=sys.stdout)
        for img_2 in files2:
            if (inputFileName == img_2):
                continue
            #print("Comparing with " + img_2)
            total = total + 1
            #try:
            digest_2 = image_digest(img_2)
            pcc = cross_correlation(digest_1, digest_2)
            #except:
            #    continue
            if (math.isnan(pcc) == False):
                phashList.append((pcc, img_2))
                if pcc <= threshold:
                    status = 'pass'
                    #log_f = sys.stdout
                else:
                    status = 'fail'
                    failures += 1
                    #log_f = sys.stderr
                #                 failures += 1
                #print('%s\t%s\t%s\t%0.3f' % (status, inputFileName, img_2, pcc))
            else:
                print(" Got nan for "+img_2)
                failures += 1
            success_pct = (total-failures)*100/total
        print('%s\t%s\t%d\t%s\t%d\t%s\t%d' % (inputFileName, 'total', total, "failures", failures, ' success_pct ', success_pct))
        phashList.sort(reverse = True, key = lambda pcc:pcc[0])
        #print(phashList)
        return failures, total, phashList


if __name__ == '__main__':
    filename="../input/bwlittlegirl.jpeg"
    i = ImageComparator()
    _,_,links = i.compareDirectoryImagesByPhash(filename, "internetimages",0.7)
    links = links[0:5]
    for link in links:
        print(link)


