import scipy.io.arff

class Dataset:
  def __init__(self, filename):
    # load data
    self.data, self.meta = scipy.io.arff.loadarff(filename)
    # shuffle

    # split into train, testing, validation

    # separate input/output