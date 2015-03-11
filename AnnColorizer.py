from __future__ import division
from __future__ import print_function

import numpy as np
import cv
import cv2
import sys
from pyfann import libfann
import os
from sklearn import preprocessing
import math
from datetime import datetime
class AnnColorizer(object):
    
    def __init__(self, learning_rate=0.9, num_neurons_hidden=30, hidden_layers=2, max_iterations=5000, iterations_between_reports=100, train_pct=0.6,
                 desired_error=0.0006, train_count=0, window_size=10):
        self.learning_rate = learning_rate
        self.num_input = 3
        self.num_neurons_hidden = num_neurons_hidden
        self.num_output = 3
        self.desired_error = desired_error
        self.train_pct = train_pct
        self.train_count = train_count
        self.max_iterations = max_iterations
        self.iterations_between_reports = iterations_between_reports
        self.ann = libfann.neural_net()
        self.hidden_layers = hidden_layers
        ann_layout = []
        ann_layout.append(self.num_input)
        for  i in range(self.hidden_layers):
            ann_layout.append(self.num_neurons_hidden)
        ann_layout.append(self.num_output)
        self.ann.create_standard_array( ann_layout)
        self.ann.set_learning_rate(self.learning_rate)
        self.ann.set_training_algorithm(libfann.TRAIN_RPROP)
        self.ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
        self.ann.set_activation_function_output(libfann.LINEAR)
        self.ann.set_train_error_function(libfann.STOPFUNC_MSE)
        self.window_size = window_size

#        print "ANN Setup with learning_rate:%0.1f neurons_hidden:%d hidden_layers:%d max_iterations:%d train_pct:%0.1f train_cnt:%d" % (learning_rate, num_neurons_hidden, hidden_layers, max_iterations, train_pct, train_count)
        print (
        "ANN Setup with learning_rate:%.1f neurons_hidden:%d hidden_layers:%d max_iterations:%d train_pct:%.1f train_cnt:%d window_size:%d" % (
        learning_rate, num_neurons_hidden, hidden_layers, max_iterations, train_pct, train_count, window_size))

        #self.ann.print_parameters()

    def __trainFannV2(self, img, image_name):
        training_filename=image_name +"_train.dat"
        trainf = open(training_filename,'w')
        l, a, b = cv2.split(cv2.cvtColor(img, cv.CV_BGR2Lab))
        xdim, ydim = l.shape

        if(self.train_count == 0):
            max_train_count = int(math.floor(xdim * ydim * self.train_pct))
        else:
            max_train_count = self.train_count
        print ("Training pixels %d"%(max_train_count))
        #max_train_count=3000
        num_input = self.ann.get_num_input()
        num_output  = self.ann.get_num_output()
        dims=[max_train_count, num_input ,num_output]
        print(*dims, sep=' ', file=trainf)
        print("Image Dimensions " + str(l.shape))
        f = trainf
        count = 1

        for k in xrange(max_train_count):
                #choose random pixel in training image
                try:
                    i = int(np.random.uniform(xdim))
                    j = int(np.random.uniform(ydim))
                    features = self.__get_features(l, xdim, ydim, i, j)
                    print(*features, sep=' ', file=f)
                    #BGR values
                    output=[ float(img[i,j,0]), float(img[i,j,1]), float(img[i,j,2])]
                    print(*output, sep=' ', file=f)
                    count = count + 1
                except Exception, e:
                    print ("Exception when training %s" % (e))
                    continue

        #for i in range(xdim):
        #    for j in range(ydim):
        #        features = self.__get_features(l, xdim, ydim, i, j)
        #        print(*features, sep=' ', file=f)
        #        #BGR values
        #        output=[ float(img[i,j,0]), float(img[i,j,1]), float(img[i,j,2])]
        #        print(*output, sep=' ', file=f)
        #        count = count + 1
        trainf.close()

        data = libfann.training_data()
        data.read_train_from_file(training_filename)
        #data.shuffle_train_data()
        train_data = data
        self.ann.set_scaling_params(train_data, -1.0, 1.01, -1.0, 1.01)
        self.ann.scale_train(train_data)
        self.ann.train_on_data(train_data, self.max_iterations, self.iterations_between_reports, self.desired_error)
        print("Training ANN done ")

        self.ann.reset_MSE()
        os.remove(training_filename)

#
#            xlim = (max(pos[0] - self.window_size,0), min(pos[0] + self.window_size,img.shape[1]))
#        ylim = (max(pos[1] - self.window_size,0), min(pos[1] + self.window_size,img.shape[0]))

    def __get_features(self, img, xdim, ydim, x, y):
        l = img[x,y]
        xlim = (max(x - self.window_size,0), min(x + self.window_size,xdim))
        ylim = (max(y - self.window_size,0), min(y + self.window_size,ydim))
        #a = img[ylim[0]:ylim[1],xlim[0]:xlim[1]]
        #print ("x,y=" + str(x) + "," + str(y) + "xdim,ydim=" + str(xdim) + "," + str(ydim) +  "xlim = " + str(xlim) + " ylim=" + str(ylim))
        #print(str(a))
        #print ("*******")
        var = np.var(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])/1000 #switched to Standard Deviation --A
        avg = np.mean(img[ylim[0]:ylim[1],xlim[0]:xlim[1]])
        if (math.isnan(var) or math.isnan(avg)):
            #raise Exception("nan returned for variance ")
            var = np.var(img[xlim[0]:xlim[1],ylim[0]:ylim[1]])/1000 #switched to Standard Deviation --A
            avg = np.mean(img[xlim[0]:xlim[1],ylim[0]:ylim[1]])
        return (float(l), avg, var)

    def train(self, files):
        # compute color map
        for file in files:
            img = self.__load_image(file)
            image_name=os.path.basename(file)
            print ("Training with image  " + file)
            self.__trainFannV2(img, image_name)

        return self

    def colorize(self, img):
        m,n = img.shape
        print("COLORIZATION START m=" + str(m) + ", n=" + str(n))
        original_l,_,_= cv2.split(cv2.cvtColor(cv2.merge((img, img, img)), cv.CV_BGR2Lab)) #default a and b for a grayscale image

        output_r = np.zeros(original_l.shape)
        output_g = np.zeros(original_l.shape)
        output_b = np.zeros(original_l.shape)
        for i in range(m):
            for j in range(n):
                features = self.__get_features(img, m, n, i, j)
                op = self.ann.run(self.ann.scale_input(features))
                output_b[i,j], output_g[i,j], output_r[i,j]  = self.ann.descale_output(op)
                #output_b[i,j], output_g[i,j], output_r[i,j]  = op
        output_img = cv2.merge((np.uint8(output_r), np.uint8(output_g), np.uint8(output_b)))
        print("colors applied " + str(datetime.now()))
        return output_img,""


    def __load_image(self, path):
        img = cv2.imread(path)
        return img

#Make plots to test image loading, Lab conversion, and color channel quantization.
#Should probably write as a proper unit test and move to a separate file.
if __name__ == '__main__':
    training_files = ['images/houses/house_004.jpg' ]
    input_file = 'images/houses/house_004.jpg'
    output_file = 'output/houses/color_ann_house_004.jpg'
    c = AnnColorizer()
    c.train(training_files)
    L, _, _ = cv2.split(cv2.cvtColor(cv2.imread(input_file), cv.CV_BGR2Lab))
    img, _ = c.colorize(L)
    print("Writing the imge to "+output_file)
    cv2.imwrite(output_file, img)

