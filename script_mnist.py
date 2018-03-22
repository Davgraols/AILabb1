'''
Created on Dec 1, 2016

@author: eao
'''
import numpy as np
import matplotlib.pyplot as plt
from NearestNeighbor import *
from Backpropagation import *
from threading import Thread

print("Hello World:)\n")
ds=DataSet()
util=Utilities()

util.test()

Train, Valid, Test = ds.load_MNIST() 
Train_images=Train[0]
Train_labels=Train[1]
Valid_images=Valid[0]
Valid_labels=Valid[1]

def PlotSample(ARR_im, ARR_lb,num):
    #util.exit_with_error("COMPLETE THE FUNCTION ACCORDING TO LABSPEC!!\n")
    #plt.plot(ARR_im, ARR_lb)
    plt.hist(ARR_im[num], label=str(ARR_lb[num]))
    plt.show()
    return

def AnalyseData(ARR_im, num):
    #util.exit_with_error("COMPLETE THE FUNCTION ACCORDING TO LABSPEC!!\n")
    print("minimum: ", min(ARR_im[num]))
    print("maximum: ", max(ARR_im[num]))
    print("standard deviation: ", np.std(ARR_im))
    print("mean value: ", np.mean(ARR_im))
    return

def fold(train_im, train_lb, valid_im, valid_lb, k):

    nn = NearestNeighborClass()
    nn.train(train_im, train_lb)  # train the classifier on the training images and labels
    predicted_labels = nn.predict(valid_im, k)  # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    mean_accuracy = np.mean(predicted_labels == valid_lb) )
    f = open("results.txt", "a")
    f.write(k, " ", mean_accuracy, "\n")
    f.close()

def testData(train_im, train_lb, k):
    train_len = len(Train_labels)
    valid_len = len(Valid_labels)

    first_train = (train_len/3)
    second_train = 2*(train_len/3)

    first_valid = valid_len/3
    second_valid = 2 * (valid_len/3)

    for k in range(1, 15):
        fold1 = Thread(target = fold, args = (Train_images[:first_train],
                                              Train_labels[:first_train],
                                              Valid_images[:first_valid],
                                              Valid_labels[:first_valid], k, ))

        fold2 = Thread(target = fold, args = (Train_images[first_train: second_train],
                                              Train_labels[first_train: second_train],
                                              Valid_images[first_valid: second_valid],
                                              Valid_labels[first_valid: second_valid], k, ))

        fold3 = Thread(target = fold, args = (Train_images[second_train:],
                                              Train_labels[second_train:],
                                              Valid_images[second_valid:],
                                              Valid_labels[second_valid:], k, ))
        fold1.start()
        fold2.start()
        fold3.start()
        fold1.join()
        fold2.join()
        fold3.join()



#PlotSample(Train_images[:1000], Train_labels[:1000], 1)

#AnalyseData(Train_images, 1)



nn=NearestNeighborClass()

nn.train(Train_images, Train_labels) # train the classifier on the training images and labels
Labels_predict = nn.predict(Valid_images, 3) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)

print 'accuracy: %f' % ( np.mean(Labels_predict == Valid_labels) )



def prepare_for_backprop(batch_size, Train_images, Train_labels, Valid_images, Valid_labels):
    
    print "Creating data..."
    batched_train_data, batched_train_labels = util.create_batches(Train_images, Train_labels,
                                              batch_size,
                                              create_bit_vector=True)
    batched_valid_data, batched_valid_labels = util.create_batches(Valid_images, Valid_labels,
                                              batch_size,
                                              create_bit_vector=True)
    print "Done!"


    return batched_train_data, batched_train_labels,  batched_valid_data, batched_valid_labels

batch_size=100;

train_data, train_labels, valid_data, valid_labels=prepare_for_backprop(batch_size, Train_images, Train_labels, Valid_images, Valid_labels)

mlp = MultiLayerPerceptron(layer_config=[784, 100, 100, 10], batch_size=batch_size)

mlp.evaluate(train_data, train_labels, valid_data, valid_labels,
             eval_train=True)

print("Done:)\n")
