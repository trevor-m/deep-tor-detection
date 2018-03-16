from dataset import Dataset
from model import Model

if __name__ == '__main__':
  dataset = Dataset(train_filename='TorCSV/Scenario-A/TimeBasedFeatures-60s-TOR-NonTOR.arff',
                    testing_filename='TorCSV/Scenario-A/TimeBasedFeatures-60s-TOR-NonTOR.arff')
  model = Model(num_features=24, num_classes=2)
  model.train(dataset)
  print(model.test(dataset))