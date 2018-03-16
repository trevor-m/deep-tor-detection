import scipy.io.arff
import numpy as np
import keras.utils

class Dataset:
  def __init__(self, train_filename, testing_filename):
    # load data
    train_data, self.meta = scipy.io.arff.loadarff(train_filename)
    testing_data, self.meta = scipy.io.arff.loadarff(testing_filename)

    # shuffle
    np.random.shuffle(train_data)
    np.random.shuffle(testing_data)

    # separate input/output
    
    self.train_x = []
    self.train_y = []
    self.testing_x = []
    self.testing_y = []

    for i in train_data:
      feature = [j for j in i] 
      self.train_x.append(np.array(feature[:-1]))
      self.train_y.append(np.array(int(feature[-1] == 'TOR')))


    for i in train_data:
      feature = [j for j in i] 
      self.testing_x.append(np.array(feature[:-1]))
      self.testing_y.append(np.array(int(feature[-1] == 'TOR')))

    #convert those lists into numpy array 
    self.train_x = np.array(self.train_x)
    self.train_y = np.array(self.train_y)
    self.testing_x = np.array(self.testing_x)
    self.testing_y = np.array(self.testing_y)

    #convert the output to one hot encoding 
    self.train_y = keras.utils.to_categorical(self.train_y, num_classes =2)
    self.testing_y = keras.utils.to_categorical(self.testing_y, num_classes =2)
