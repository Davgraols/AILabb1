'''
Created on Dec 1, 2016

@author:  
Adopted from CS231n
'''

import numpy as np
import progressbar

class NearestNeighborClass(object):
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        bar = progressbar.ProgressBar(maxval=num_test, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]).start()
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        # loop over all test rows
        for i in xrange(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            #distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))

            min_labels = []
            while(len(min_labels) <= k):
                min_index = np.argmin(distances) # get the index with smallest distance
                min_labels.append(self.ytr[min_index])
                distances[min_index] = float("inf") #set the distance to infinity so it will not be picked again

            label_count = 0
            max_label = 0
            for l in min_labels:
                if(min_labels.count(l) > label_count):
                    label_count = min_labels.count(l)
                    max_label = l


            #Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
            Ypred[i] = max_label
            bar.update(i+1)
        bar.finish()

        return Ypred

