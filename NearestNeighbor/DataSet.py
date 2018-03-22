'''
Created on Dec 1, 2016

@author: eao
'''
import os
import cPickle, gzip
import numpy
        
class DataSet(object):
    '''
    classdocs
    '''
    def test(self):

        #       dir_path = os.path.dirname(os.path.realpath(__file__))
        dir_path = os.getcwd()
        print("Test is done. DIR=", dir_path)
        return
    
    def unpickle_ciffar10(self,file_name):

        fo = open(file_name, 'rb')
        dictionary = cPickle.load(fo)
        fo.close()
        return dictionary

    def unpickle_MNIST(self,file_name):

        fo = gzip.open(file_name)
        train_set, valid_set, test_set = cPickle.load(fo)
        fo.close()
        return train_set, valid_set, test_set
    
    def load_MNIST(self):
        #set the path to dataset directory

        cwd_path = os.getcwd()
        #print (cwd_path)
        mnistdir=cwd_path+"/Datasets/MNIST_pkl/"
        print (mnistdir)
        name="mnist.pkl.gz"
        file_name=mnistdir+name
        Train_set, Valid_set, Test_set =self.unpickle_MNIST(file_name)

        
        return Train_set, Valid_set, Test_set
    
    def load_CIFAR10(self):
        #set the path to dataset directory

        cwd_path = os.getcwd()
        #print (cwd_path)
        cifar10dir=cwd_path+"/Datasets/cifar-10-batches-py/"
        #print (cifar10dir)
        name1="data_batch_1"
        name2="data_batch_2"
        name3="data_batch_3"
        name4="data_batch_4"
        name5="data_batch_5"
        name6="test_batch"
        name7="batches.meta"
        
        file_name1=cifar10dir+name1
        file_name2=cifar10dir+name2
        file_name3=cifar10dir+name3
        file_name4=cifar10dir+name4
        file_name5=cifar10dir+name5
        file_name6=cifar10dir+name6
        file_name7=cifar10dir+name7
#        print (file_name)
 
        
        X1=self.unpickle(file_name1)
        X2=self.unpickle(file_name2)
        X3=self.unpickle(file_name3)
        X4=self.unpickle(file_name4)
        X5=self.unpickle(file_name5)
        X6=self.unpickle(file_name6)
        X7=self.unpickle(file_name7)
        
        
        
#         Itr=numpy.concatenate((X1['data'], X2['data'], X3['data'], X4['data'], X5['data']), axis=0)
#         Ltr=numpy.concatenate((X1['labels'], X2['labels'], X3['labels'], X4['labels'], X5['labels']), axis=0)
        Itr=numpy.concatenate((X1['data'], X2['data']), axis=0)
        Ltr=numpy.concatenate((X1['labels'], X2['labels']), axis=0)
        Ite=X6['data']
        Lte=X6['labels']
#          Itr=0
#         Ltr=0
#         Ite=0
#         Lte=0
       
#         for k, v in X.items():
#             print(k, v)
#         
#         print(X['label_names'][0])
        
        return Itr, Ltr, Ite, Lte
    
    def __init__(self):
        '''
        Constructor
        '''
        