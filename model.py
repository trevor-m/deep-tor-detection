from keras.models import Sequential
from keras.layers import Dense

class Model:

  def __init__(self, num_features, num_classes):
    # define our model
    self.model = Sequential()
    self.model.add(Dense(256, activation='relu', input_shape=(num_features)))
    self.model.add(Dense(num_classes, activation='softmax'))
    self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')