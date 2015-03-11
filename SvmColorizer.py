from __future__ import division
import numpy as np
import cv
import cv2
import itertools
from sklearn.svm import SVC, LinearSVC
from sklearn import preprocessing
import pdb
import pygco
from scipy.cluster.vq import kmeans,vq
from sklearn.decomposition import PCA
import scipy.ndimage.filters
from datetime import datetime
import time
from threading import Thread
from Queue import Queue
import colorsys
from random import randint, uniform
from skimage.color import rgb2lab

#from numba import autojit

class SvmColorizer(object):

    def __init__(self, ncolors=16, probability=False, npca=32, svmgamma=0.1, svmC=1, graphcut_lambda=1, ntrain=3000, selfcolor=False,
                 window_size = 10, surf_window_size = 20):

        self.surf_window = surf_window_size
        self.window_size = window_size

        self.levels = int(np.floor(np.sqrt(ncolors)))
        self.ncolors = ncolors
        self.ntrain = ntrain

        # declare classifiers
        self.svm = [SVC(probability=probability, gamma=svmgamma, C=svmC) for i in range(self.ncolors)]
        #self.svm = [LinearSVC() for i in range(self.ncolors)]

        self.scaler = preprocessing.MinMaxScaler()                          # Scaling object -- Normalizes feature array
        self.pca = PCA(npca)

        self.centroids = []
        self.probability = probability
        self.colors_present = []
        self.surf = cv2.DescriptorExtractor_create('SURF')
        self.surf.setBool('extended', True) #use the 128-length descriptors

        self.graphcut_lambda=graphcut_lambda
        self.concurrent=200

        self.selfcolor = selfcolor

        print " ncolors: %d pca: %d ntrain:%d selfcolor:%s window: %d surf_window: %d"%(self.ncolors,npca, self.ntrain, self.selfcolor,
        self.window_size, self.surf_window)
        #self.setupQueue()
        #self.numba_init()

    def getMean(self, img, pos):
        xlim = (max(pos[0] - self.window_size,0), min(pos[0] + self.window_size,img.shape[1]))
        ylim = (max(pos[1] - self.window_size,0), min(pos[1] + self.window_size,img.shape[0]))
        return np.mean(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])

    def getVariance(self, img, pos):
        xlim = (max(pos[0] - self.window_size,0), min(pos[0] + self.window_size,img.shape[1]))
        ylim = (max(pos[1] - self.window_size,0), min(pos[1] + self.window_size,img.shape[0]))
        return np.var(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])/1000 #switched to Standard Deviation --A

    def feature_surf(self, img, pos):
        octave2 = cv2.GaussianBlur(img, (0, 0), 1)
        octave3 = cv2.GaussianBlur(img, (0, 0), 2)
        kp = cv2.KeyPoint(pos[0], pos[1], self.surf_window)
        _, des1 = self.surf.compute(img, [kp])
        _, des2 = self.surf.compute(octave2, [kp])
        _, des3 = self.surf.compute(octave3, [kp])
        return np.concatenate((des1[0], des2[0], des3[0]))

    def feature_dft(self, img, pos):
        xlim = (max(pos[0] - self.window_size,0), min(pos[0] + self.window_size,img.shape[1]))
        ylim = (max(pos[1] - self.window_size,0), min(pos[1] + self.window_size,img.shape[0]))
        patch = img[ylim[0]:ylim[1],xlim[0]:xlim[1]]
        l = (2*self.window_size + 1)**2
        #return all zeros for now if we're at the edge
        if patch.shape[0]*patch.shape[1] != l:
            return np.zeros(l)
        return np.abs(np.fft(patch.flatten()))

    #@autojit
    def get_features(self, img, pos):
        meanvar = np.array([self.getMean(img, pos), self.getVariance(img, pos)]) #variance is giving NaN
        feat = np.concatenate((meanvar, self.feature_surf(img, pos), self.feature_dft(img, pos)))
        return feat

    def train(self, files):
        features = []
        self.local_grads = []
        classes = []
        kmap_a = []
        kmap_b = []

        # compute color map
        for f in files:
            print ("Training with " + f)
            _,a,b = self.load_image(f)
            kmap_a = np.concatenate([kmap_a, a.flatten()])
            kmap_b = np.concatenate([kmap_b, b.flatten()])

        startMillis = int(round(time.time() * 1000))
        self.train_kmeans(kmap_a,kmap_b,self.ncolors)
        endMillis = int(round(time.time() * 1000))
        print (" K-Means (ms)" + str((endMillis - startMillis)))

        for f in files:
            l,a,b = self.load_image(f)
            a,b = self.quantize_kmeans(a,b)
            #dimensions of image
            m,n = l.shape
            startMillis = int(round(time.time() * 1000))
            for i in xrange(self.ntrain):
                #choose random pixel in training image
                x = int(np.random.uniform(n))
                y = int(np.random.uniform(m))
                features.append(self.get_features(l, (x,y)))
                classes.append(self.color_to_label_map[(a[y,x], b[y,x])])
            #print ("Processing DONE "  + f + " " + str(datetime.now()))
            endMillis = int(round(time.time() * 1000))
            print (" Training random pixes (ms)" + str((endMillis - startMillis)))

        # normalize features to use and use PCA to minimize their #
        self.features = self.scaler.fit_transform(np.array(features))
        classes = np.array(classes)
        self.features = self.pca.fit_transform(self.features)
        for i in range(self.ncolors):
            if len(np.where(classes==i)[0])>0:
                curr_class = (classes==i).astype(np.int32)
                #print("CURR  i " + str(i) + " " + str(curr_class))
                self.colors_present.append(i)
                self.svm[i].fit(self.features,(classes==i).astype(np.int32))
        return self

    #@autojit
    def input_image_feature_task(self, img, x, y, label_costs, skip, num_classes, count):

        #innerLoopTime = int(round(time.time() * 1000))
        if (0 == count % 10000):
           print ("Processing "+str(count) + "  " + str(datetime.now()));
        feat = self.scaler.transform(self.get_features(img, (x,y)))
        feat = self.pca.transform(feat)
        #count += 1

        # Hard-but-correct way to get g
        # self.g[y-int(skip/2):y+int(skip/2)+1,x-int(skip/2):x+int(skip/2)+1] = self.color_variation(feat)

        #get margins to estimate confidence for each class
        for i in range(num_classes):
           distance = self.svm[self.colors_present[i]].decision_function(feat)
           cost = -1*self.svm[self.colors_present[i]].decision_function(feat)[0]
           #print(" i " + str(i) + " COST "+str(cost) + " distance "+ str(distance))
           label_costs[y-int(skip/2):y+int(skip/2)+1,x-int(skip/2):x+int(skip/2)+1,i] = cost


    #def numba_init(self):
    #    self.savethread = pythonapi.PyEval_SaveThread
    #    self.savethread.argtypes = []
    #    self.savethread.restype = c_void_p

    #    self.restorethread = pythonapi.PyEval_RestoreThread
    #    self.restorethread.argtypes = [c_void_p]
    #   self.restorethread.restype = None

    def doWork(self):
        #print(" **** SETTING UP TASK " )
        while True:
            (img, x, y, label_costs, skip, num_classes, count)  = self.queue.get()
            self.input_image_feature_task(img, x, y, label_costs, skip, num_classes, count)
            self.queue.task_done()

    def setupQueue(self):
        self.queue = Queue(self.concurrent * 4)
        print("TASKS CREATED concurrent = "+str(self.concurrent))

        for i in range(self.concurrent):
            #print("TASKS CREATED "+str(i))
            t = Thread(target=self.doWork)
            t.daemon = True
            t.start()
        #print("TASKS CREATED ")

    def enqueue(self, img, x, y, label_costs, skip, num_classes, count):
        self.queue.put((img, x, y, label_costs, skip, num_classes, count))

    #@autojit
    def loop2d(self, img, n, m, skip, num_classes, label_costs):
        count = 0
        for x in xrange(0,n,skip):
           for y in xrange(0,m,skip):
               count = count+1
               #self.enqueue(img, x, y, label_costs, skip, num_classes, count)
               self.input_image_feature_task(img, x, y, label_costs, skip, num_classes, count)

    def colorize(self, img, skip=4):
        print "Skipping %d pixels" %(skip)
        m,n = img.shape
        num_classified = 0

        _,raw_output_a,raw_output_b = cv2.split(cv2.cvtColor(cv2.merge((img, img, img)), cv.CV_RGB2Lab)) #default a and b for a grayscale image

        output_a = np.zeros(raw_output_a.shape)
        output_b = np.zeros(raw_output_b.shape)

        num_classes = len(self.colors_present)
        label_costs = np.zeros((m,n,num_classes))
        self.g = np.zeros(raw_output_a.shape)

        count=0
        print("colorize() start =" + str(m) + ", n=" + str(n) + " Total iterations " +  str(m/skip * n/skip)+ "  at  " + str(datetime.now()))
        count=0
        self.loop2d(img, n,m, skip, num_classes, label_costs)
        #for x in xrange(0,n,skip):
            #print("Coloring " + str(count) + " " + str(datetime.now()))
            #fullInnerLoopTime = int(round(time.time() * 1000))
        #    for y in xrange(0,m,skip):
                #self.input_image_feature_task(img, x, y, label_costs, skip, num_classes)
        #        count = count+1
                #self.enqueue(img, x, y, label_costs, skip, num_classes, count)
        #        self.input_image_feature_task(img, x, y, label_costs, skip, num_classes, count)
            #fulllInnerLoopEndTime = int(round(time.time() * 1000))
            #print (" One Outer iteration time (secs)"+ str((fulllInnerLoopEndTime - fullInnerLoopTime)/1000))
        #self.queue.join()
        #edges = self.get_edges(img)
        #self.g = np.sqrt(edges[0]**2 + edges[1]**2)
        self.g = self.get_edges(img)
        #self.g = np.log10(self.g)

        print("input image features done for " + str(count) + " " + str(datetime.now()))
        #postprocess using graphcut optimization
        output_labels = self.graphcut(label_costs, l=self.graphcut_lambda)
        print("graphcut done " + str(datetime.now()))

        for i in range(m):
            for j in range(n):
                a,b = self.label_to_color_map[self.colors_present[output_labels[i,j]]]
                output_a[i,j] = a
                output_b[i,j] = b

        output_img = cv2.cvtColor(cv2.merge((img, np.uint8(output_a), np.uint8(output_b))), cv.CV_Lab2RGB)
        print("colors applied " + str(datetime.now()))
        return output_img, self.g

    def load_image(self, path):
        img = cv2.imread(path)
        #convert to L*a*b* space and split into channels
        l, a, b = cv2.split(cv2.cvtColor(img, cv.CV_BGR2Lab))
        if (self.selfcolor == True):
           a = l
           b = l
        return l, a, b

    def get_edges(self, img, blur_width=3):
        img_blurred = cv2.GaussianBlur(img, (0, 0), blur_width)
        vh = cv2.Sobel(img_blurred, -1, 1, 0)
        vv = cv2.Sobel(img_blurred, -1, 0, 1)
        #vh = vh/np.max(vh)
        #vv = vv/np.max(vv)
        #v = np.sqrt(vv**2 + vh**2)
        v = 0.5*vv + 0.5*vh
        return v

    def graphcut(self, label_costs, l=100):
        num_classes = len(self.colors_present)
        print(" Label costs "+str(label_costs.shape) + " num_classes "+str(num_classes))
        #calculate pariwise potiential costs (distance between color classes)
        pairwise_costs = np.zeros((num_classes, num_classes))
        for ii in range(num_classes):
            for jj in range(num_classes):
                c1 = np.array(self.label_to_color_map[ii])
                c2 = np.array(self.label_to_color_map[jj])
                pairwise_costs[ii,jj] = np.linalg.norm(c1-c2)

        label_costs_int32 = (100*label_costs).astype('int32')
        pairwise_costs_int32 = (l*pairwise_costs).astype('int32')
        vv_int32 = (self.g).astype('int32')
        vh_int32 = (self.g).astype('int32')
        new_labels = pygco.cut_simple_vh(label_costs_int32, pairwise_costs_int32, vv_int32, vh_int32, n_iter=10, algorithm='swap')
        print("NEW LABELS " + str(new_labels.shape))
        return new_labels

    # use golden ratio
    def generateSelfLabColorsFromHsv(self, count):
        golden_ratio_conjugate = 0.618033988749895
        ignore = randint(0, 20)
        for i in range(ignore):
            h   = randint(1,360) # use random start value
            s = uniform(0.01,0.9)
            v = uniform(0.01,0.9)
        print(" IGNORE "+str(ignore) +  " starting h "+ str(h) + " s "+ str(s) + " v " + str(v))

        rgbColors = np.zeros((1, count, 3), dtype=np.float)
        for i in range(count):
            r,g,b = colorsys.hsv_to_rgb(h, s, v)
            rgbColors[0, i] = [r,g,b]
            h += golden_ratio_conjugate + 1
        lab = rgb2lab(rgbColors)
        _,a,b = cv2.split(lab)
        #print("LABB " + str(lab) + "\naaa " + str(a.flatten()) + "\nbbbb " + str(b.flatten()))
        labColors =np.column_stack((a.flatten(),b.flatten()))
        #print("SHAPE " + str(labColors.shape) + "\nfinal " + str(labColors))
        return labColors

    def generateSelfLabColors(self, count):
        a = np.ones((count), dtype=float)
        b = np.ones((count), dtype=float)
        for i in range(count):
            a[i] = a[i] * randint(0,127)
            b[i] = b[i] * randint(128,255)
        labColors =np.column_stack((a.flatten(),b.flatten()))
        #print("SHAPE " + str(labColors.shape) + "\nfinal " + str(labColors))
        return labColors

    def train_kmeans(self, a, b, k):
        pixel = np.squeeze(cv2.merge((a.flatten(),b.flatten())))
        if (self.selfcolor == True):
            self.centroids = self.generateSelfLabColors(k)
        else:
            self.centroids,_ = kmeans(pixel,k) # six colors will be found
        qnt,_ = vq(pixel,self.centroids)
        #print("CENTROIDS " + str(self.centroids))

        #color-mapping lookup tables
        self.color_to_label_map = {c:i for i,c in enumerate([tuple(i) for i in self.centroids])} #this maps the color pair to the index of the color
        #print("color_to_label_map "+str(self.color_to_label_map))
        self.label_to_color_map = dict(zip(self.color_to_label_map.values(),self.color_to_label_map.keys())) #takes a label and returns a,b

    def quantize_kmeans(self, a, b):
        w,h = np.shape(a)
        # reshape matrix
        pixel = np.reshape((cv2.merge((a,b))),(w * h,2))
        # quantization
        qnt,_ = vq(pixel,self.centroids)
        # reshape the result of the quantization
        centers_idx = np.reshape(qnt,(w,h))
        clustered = self.centroids[centers_idx]

        a_quant = clustered[:,:,0]
        b_quant = clustered[:,:,1]
        return a_quant, b_quant
