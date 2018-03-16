from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

class Model:
  NUM_EPOCHS = 50
  BATCH_SIZE = 32

  def __init__(self, num_features, num_classes):
    self.num_features = num_features
    self.num_classes = num_classes
    # define our model
    self.basic_mlp()

  def basic_mlp(self):
    """Single hidden layer"""
    self.model = Sequential()
    self.model.add(Dense(256, activation='relu', input_shape=(self.num_features)))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(self.num_classes, activation='softmax'))
    self.model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics='accuracy')

  def deep_mlp(self, num_layers=6):
    """More hidden layers"""
    self.model = Sequential()
    for i in range(num_layers):
      self.model.add(Dense(16, activation='relu', input_shape=(self.num_features)))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(self.num_classes, activation='softmax'))
    self.model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics='accuracy')

  def self_normalizing_network(self):
    """Klambauer et. al. 2017"""
    self.model = Sequential()
    self.model.add(Dense(256, activation='selu', kernel_initializer='lecun_normal', input_shape=(num_features)))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(self.num_classes, activation='softmax'))
    self.model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics='accuracy')

  def train(self, dataset):
    self.model.fit(x=dataset.train_x,
                   y=dataset.train_y,
                   batch_size=Model.BATCH_SIZE,
                   epochs=Model.NUM_EPOCHS,
                   validation_split=0.1,
                   shuffle=True,
                   verbose=2)

  def test(self, dataset):
    return self.model.evaluate(x=dataset.testing_x,
                              y=dataset.testing_y,
                              batch_size=Model.BATCH_SIZE)