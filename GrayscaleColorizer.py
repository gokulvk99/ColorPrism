import cv
import cv2
import numpy as np
import os, errno
import argparse

from datetime import datetime
import time

from CompareColor import ciede2000Diff, labDiff, psnr
from TrainingImageFetcher import TrainingImageFetcher
from SvmColorizer import SvmColorizer
from AnnColorizer import AnnColorizer

class GrayscaleColorizer(object):

    def get_grayscale_from_color(self, color_file):
        L, _, _ = cv2.split(cv2.cvtColor(cv2.imread(color_file), cv.CV_BGR2Lab))
        return L

    def mkdir_p(self, path):
        try:
            os.makedirs(path)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise

    def main(self):
       parser = argparse.ArgumentParser()
       parser.add_argument('--test', default='', help="original color file")
       parser.add_argument('--repeat', dest='repeats', default=3, type=int, help="how many times to run each training file")
       parser.add_argument('--input', default='', help="input file to colorize")
       parser.add_argument('--output', default='', help="output directory")
       parser.add_argument('--internet-images', dest='internet_folder', default='../internetimages', help="folder to use for saving images downloaded from internet")
       parser.add_argument('--mode', default="svm", help="use svm or ann")
       parser.add_argument('--internet', help="Use Internet", dest='internet', action='store_true')
       parser.add_argument('--no-internet', help="Donot use internet", dest='internet', action='store_false')
       parser.add_argument('--search', help="Do image search", dest='image_search', action='store_true')
       parser.add_argument('--no-search', help="Donot do image search", dest='image_search', action='store_false')
       parser.add_argument('--self', dest='self_color', help="enable autocoloring",  action='store_true')
       parser.add_argument('--no-self', dest='self_color', help="disable autocoloring", action='store_false')
       parser.add_argument('--images', default=5, type=int, help="no of images to search for in the internet")
       # SVM Specific
       parser.add_argument('--svm-colors', default=16, type=int, help="no of colors")
       parser.add_argument('--svm-pca', default=32, type=int, help="no of principal components")
       parser.add_argument('--svm-skip', default=1, type=int, help="no of pixels to skip")
       parser.add_argument('--svm-train', default=3000, type=int, help="no of pixels to to use for training")
       parser.add_argument('--svm-window', dest='svm_window_size', default=10, type=int, help="window size for non-surf feature calculation")
       parser.add_argument('--svm-surf-window', dest="svm_surf_window_size", default=20, type=int, help="window size for surf feature calculation")
       # ANN Specific
       parser.add_argument('--ann-train-pct', dest='ann_train_pct', default=0.6, type=float, help="% of data to be used for training ")
       parser.add_argument('--ann-learning-rate', dest='ann_learning_rate', default=0.9, type=float, help="learning rate %")
       parser.add_argument('--ann-neurons', dest='ann_neurons', default=30, type=int, help="no of neuron in a hidden layer")
       parser.add_argument('--ann-layers', dest='ann_layers', default=2, type=int, help="no of hidden layers")
       parser.add_argument('--ann-train', dest='ann_train', default=0, type=int, help="no of pixels to to use for training")
       parser.add_argument('--ann-iterations', dest='ann_iterations', default=5000, type=int, help="max of iterations to perform")
       parser.add_argument('--ann-desired-error', dest='ann_desired_error', default=0.0006, type=float, help="Desired error rate to reach ")
       parser.add_argument('--ann-window', dest='ann_window_size', default=1, type=int, help="window size for ANN feature calculation")

       parser.set_defaults(internet=True,self_color=False, image_search=True)

       args = parser.parse_args()

       test_file = ""
       if (args.input == "" or args.output == ""):
           parser.error("Missing args...")
       input_file = os.path.abspath(args.input)
       mode = args.mode
       if (args.test != ""):
           test_file = args.test
       elif (args.self_color == True):
           test_file = input_file
       #else:
       #    parser.error("Missing test file args...")

       #SVM SPecifici
       ncolors = args.svm_colors
       npca = args.svm_pca
       ntrain= args.svm_train
       skip = args.svm_skip
       svmgamma = 0.25
       svmC = 0.5
       graphcut_lambda = 1
       window_size = args.svm_window_size
       surf_window_size = args.svm_surf_window_size

       image_limit = args.images
       self_color = args.self_color
       image_search = args.image_search
       internet_image_folder = args.internet_folder
       internet = args.internet
       output_dir = args.output
       self.mkdir_p(output_dir)
       print("Starting Processing  " + str(datetime.now()))
       startMillis = int(round(time.time() * 1000))
       grayscale_image = self.get_grayscale_from_color(input_file)
       cv2.imwrite(output_dir+"/"+ mode + "_grayscale_"+os.path.basename(input_file), grayscale_image)
       print(input_file + " dimension " + str(grayscale_image.shape))

       if (internet == True) :
           f = TrainingImageFetcher(input_file, image_limit, (grayscale_image.shape[1], grayscale_image.shape[0]), internet_image_folder, image_search)
           training_files =  f.getTrainingImages()
       else:
           print("Internet Search turned off ")
           training_files = [(0, test_file)]
       i = 0
       metrics = []
       repeats = args.repeats

       ann_train_pct = args.ann_train_pct
       ann_neurons =  args.ann_neurons
       ann_train = args.ann_train
       ann_layers = args.ann_layers
       ann_iterations = args.ann_iterations
       ann_desired_error = args.ann_desired_error
       ann_learning_rate = args.ann_learning_rate
       ann_window_size = args.ann_window_size

       print "Repeat each test %d"%(repeats)
       ciedeList = []
       for pcc, training_file in training_files:
           try:
               c2000diffs = []
               for k in range(repeats):
                    i += 1
                    if (mode == "svm"):
                        c = SvmColorizer(ncolors=ncolors, npca=npca, svmgamma=svmgamma, svmC=svmC, graphcut_lambda=graphcut_lambda,
                                       ntrain=ntrain, selfcolor=self_color, window_size = window_size, surf_window_size = surf_window_size)
                    else:
                        c = AnnColorizer(learning_rate=ann_learning_rate, num_neurons_hidden=ann_neurons, hidden_layers=ann_layers,
                                         max_iterations=ann_iterations, iterations_between_reports=ann_iterations/10, train_pct=ann_train_pct,
                                         desired_error=ann_desired_error, train_count=ann_train, window_size=ann_window_size);

                    training_img = self.get_grayscale_from_color(training_file)
                    print(" Repeat " + str(k+1) + " *************** [" + str(i) + "] Start training using "+training_file + " Dimension "+str(training_img.shape))
                    c.train([training_file])
                    trainMillis = int(round(time.time() * 1000))
                    print(" image training done " + str(datetime.now()))
                    #try:
                        #colorize the input image
                    if (mode == "svm"):
                        colorized_image, _ = c.colorize(grayscale_image,skip=skip)
                    else:
                        colorized_image, _ = c.colorize(grayscale_image)

                    colorMillis = int(round(time.time() * 1000))
                    print ("took (seconds) image dim"+ str(grayscale_image.shape) + " " + str((colorMillis - startMillis)/1000))
                    l, a, b = cv2.split(cv2.cvtColor(colorized_image, cv.CV_RGB2Lab))
                    newColorMap = cv2.cvtColor(cv2.merge((128*np.uint8(np.ones(np.shape(l))),a,b)), cv.CV_Lab2BGR)
                    if (self_color == True):
                        self_mode = "self_"
                    else:
                        self_mode = ""
                    colored_input_file = output_dir+"/" + self_mode + mode + "_color" + str(i) + "_" + os.path.basename(input_file)
                    cv2.imwrite(colored_input_file, cv2.cvtColor(colorized_image, cv.CV_RGB2BGR))
                    cv2.imwrite(output_dir+"/"+ self_mode + mode +'_cmap_'+str(i) +"_"+os.path.basename(input_file), newColorMap)
                    cv2.imwrite(output_dir+"/"+ self_mode + mode +'_training'+ str(i) + ".jpg", cv2.imread(training_file))
                    if (test_file != ""):
                        c2000diff =  ciede2000Diff(colored_input_file, test_file)
                        labdiff = labDiff(colored_input_file, test_file)
                        psnrDiff = psnr(colored_input_file, test_file)
                        print(" ciede200 %3f Lab %3f psnr %3f "%(c2000diff, labdiff, psnrDiff))
                        c2000diffs.append(c2000diff)
               avg = float(sum(c2000diffs))/len(c2000diffs) if len(c2000diffs) > 0 else float('nan')
               print " average CIEDE %0.2f"%(avg)
           except Exception,e :
               print "Exception when colorizing : %s" % e
               continue
           ciedeList.append((avg, training_file))
       ciedeList.sort(reverse = False, key = lambda pcc:pcc[0])
       ciede,training_file = ciedeList[0]
       print "Best CIEDE %.2f with Training File %s"%(ciede, training_file)
       print("ALL DONE " + str(datetime.now()))

if __name__ == '__main__':
    gc = GrayscaleColorizer()
    gc.main()
