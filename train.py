from dataset import Dataset
from model import Model
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Settings
USE_KFOLD = True

if __name__ == '__main__':
  dataset = Dataset(train_filename='TorCSV/Scenario-A/TimeBasedFeatures-60s-TOR-NonTOR-85.arff',
                    testing_filename='TorCSV/Scenario-A/TimeBasedFeatures-60s-TOR-NonTOR-15.arff')
  if USE_KFOLD:
    # k-fold cross validation
    accs = []
    precs = []
    recs = []
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    for train, test in kfold.split(dataset.x, dataset.y[:,1]):
      # get this dataset
      train_x = dataset.x[train]
      train_y = dataset.y[train]
      test_x = dataset.x[test]
      test_y = dataset.y[test]
      # train model
      model = Model(num_features=23, num_classes=2)
      model.train(train_x, train_y)
      # evaluate
      loss, acc = model.test(dataset.x[test], dataset.y[test])
      prec, rec = model.metrics(dataset.x[test], dataset.y[test])
      accs.append(acc)
      precs.append(prec)
      recs.append(rec)
      print('acc: %f - prec: %f - rec: %f' % (acc, prec, rec))
    print('\nAverage Accuracy:', np.mean(accs), '+/-', np.std(accs))
    print('Average Precision:', np.mean(precs), '+/-', np.std(precs))
    print('Average Recall:', np.mean(recs), '+/-', np.std(recs))
  else:
    model = Model(num_features=23, num_classes=2)
    model.train(dataset.train_x, dataset.train_y)
    loss, acc = model.test(dataset.testing_x, dataset.testing_y)
    print('\nFinal accuracy:', acc)
    prec, rec = model.metrics(dataset.testing_x, dataset.testing_y)
    print('Precision:', prec)
    print('Recall:', rec)