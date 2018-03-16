from dataset import Dataset
from model import Model

if __name__ == '__main__':
  dataset = Dataset(train_filename='TorCSV/Scenario-A/TimeBasedFeatures-60s-TOR-NonTOR-85.arff',
                    testing_filename='TorCSV/Scenario-A/TimeBasedFeatures-60s-TOR-NonTOR-15.arff')
  model = Model(num_features=23, num_classes=2)
  model.train(dataset)
  loss, acc = model.test(dataset)
  print('\nFinal accuracy:', acc)