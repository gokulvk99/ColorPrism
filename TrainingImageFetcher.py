
from GoogleImageSearcher import GoogleImageSearcher
from ImageComparator import ImageComparator
import os, errno
import glob

import shutil
class TrainingImageFetcher(object):

    def mkdir_p(self, path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

    def __init__(self, inputImagename, numberOfImagesWanted, image_dimension, output_folder="../internetimages", image_search = True):
        print "Using Folder %s for caching images downloaded from internet. Image_Search: %s"%(output_folder, image_search)
        self.imagesFound = 0
        self.inputFileName = inputImagename
        self.numberOfImagesWanted = numberOfImagesWanted
        self.gIS = GoogleImageSearcher()
        self.imageComparator = ImageComparator()
        self.threshold = 0.7
        self.outputFolder = output_folder
        self.mkdir_p(self.outputFolder)
        filelist = glob.glob(self.outputFolder + "/*.*")
        self.image_search = image_search
        if (image_search == True):
            for f in filelist:
                os.remove(f)
        self.image_dimension = image_dimension


    def getTrainingImages(self):
        if (self.image_search == True):
            self.gIS.getImages(self.inputFileName, self.outputFolder, self.numberOfImagesWanted, self.image_dimension)
        _,_,trainingList = self.imageComparator.compareDirectoryImagesByPhash(self.inputFileName, self.outputFolder, self.threshold)
        #_,_,trainingList = self.imageComparator.compareDirectoryImagesByGist(self.inputFileName, self.outputFolder, 1-self.threshold)
        print("***************** Returning the " + str(self.numberOfImagesWanted))
        print (str(trainingList[0:self.numberOfImagesWanted]))
        return trainingList[0:self.numberOfImagesWanted]

#Make plots to test image loading, Lab conversion, and color channel quantization.
#Should probably write as a proper unit test and move to a separate file.
if __name__ == '__main__':
    filelist = glob.glob("../internetimages/*.*")
    for f in filelist:
        os.remove(f)    #os.remove("output")
    filename="../imagesearch/bwlittlegirl.jpeg"
    f = TrainingImageFetcher(filename, 50, (128,128));
    links =  f.getTrainingImages()
    links = links[0:5]
    for link in links:
        print(link)

