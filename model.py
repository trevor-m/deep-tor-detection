from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import Adam
from sklearn.metrics import average_precision_score, recall_score

class Model:
  NUM_EPOCHS = 20
  BATCH_SIZE = 32

  def __init__(self, num_features, num_classes):
    self.num_features = num_features
    self.num_classes = num_classes
    # define our model
    self.batchnorm_mlp()

  def basic_mlp(self):
    """Single hidden layer"""
    self.model = Sequential()
    self.model.add(Dense(256, activation='relu', input_shape=[self.num_features]))
    self.model.add(Dropout(0.5))
    self.model.add(Dense(self.num_classes, activation='softmax'))
    self.model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

  def batchnorm_mlp(self):
    """Single hidden layer"""
    self.model = Sequential()
    self.model.add(Dense(256, input_shape=[self.num_features]))
    self.model.add(BatchNormalization())
    self.model.add(Activation('relu'))
    self.model.add(Dense(self.num_classes, activation='softmax'))
    self.model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

  def deep_mlp(self, num_layers=6):
    """Multiple hidden layers"""
    self.model = Sequential()
    self.model.add(Dense(256, input_shape=[self.num_features]))
    self.model.add(BatchNormalization())
    self.model.add(Activation('relu'))
    for i in range(num_layers):
      self.model.add(Dense(256))
      self.model.add(BatchNormalization())
    self.model.add(Activation('relu'))
    self.model.add(Dense(self.num_classes, activation='softmax'))
    self.model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

  def train(self, x, y):
    self.model.fit(x=x,
                   y=y,
                   batch_size=Model.BATCH_SIZE,
                   epochs=Model.NUM_EPOCHS,
                   validation_split=0.1,
                   shuffle=True,
                   verbose=0)

  def test(self, x, y):
    return self.model.evaluate(x=x,
                              y=y,
                              batch_size=Model.BATCH_SIZE,
                              verbose=0)

  def metrics(self, x, y):
    pred_y = self.model.predict(x=x, batch_size=Model.BATCH_SIZE)
    pred_y = pred_y.round()
    prec = average_precision_score(y_true=y, y_score=pred_y, average='weighted')
    rec = recall_score(y_true=y, y_pred=pred_y, average='weighted')
    return prec, rec