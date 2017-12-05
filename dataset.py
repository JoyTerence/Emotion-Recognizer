import csv
import numpy as np

class dataset_reader(object):
    def __init__(self):
        self.dataset = np.ndarray(shape=(35888, 2305))
        self.train_data = np.ndarray(shape=(20000, 2304))
        self.valid_data = np.ndarray(shape=(7945, 2304))
        self.test_data = np.ndarray(shape=(7943, 2304))

    def get_dataset_before_learn(self, filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            count = -1
            print ("here")
            for row in reader:
                if count!=-1:
                    label = row[0]
                    image = np.asarray(row[1].split(' '))
                    self.dataset[count][0] = label
                    self.dataset[count][1:] = np.copy(image)
                    print (count)
                count += 1
            print ("final : ", count)
        return self.dataset

    def get_dataset_during_learn(self, train_file, valid_file, test_file):
        entire_train_data = np.ndarray(shape=(20000, 2305))
        entire_valid_data = np.ndarray(shape=(7945, 2305))
        entire_test_data = np.ndarray(shape=(7943, 2305))

        print ("Reading from train.txt file")
        with open(train_file, 'r') as f:
            entire_train_data = np.loadtxt(f, unpack=True)
        print ("Done")

        self.train_label = entire_train_data.transpose()[:,0]
        self.train_data = entire_train_data.transpose()[:,1:]

        print ("Reading from valid.txt file")
        with open(valid_file, 'r') as f:
            entire_valid_data = np.loadtxt(f, unpack=True)
        print ("Done")

        self.valid_label = entire_valid_data.transpose()[:,0]
        self.valid_data = entire_valid_data.transpose()[:,1:]

        print ("Reading from test.txt file")
        with open(test_file, 'r') as f:
            entire_test_data = np.loadtxt(f, unpack=True)
        print ("Done")

        self.test_label = entire_test_data.transpose()[:,0]
        self.test_data = entire_test_data.transpose()[:,1:]

        print (self.train_data.shape, " ", self.train_label.shape)
        print (self.valid_data.shape, " ", self.valid_label.shape)
        print (self.test_data.shape, " ", self.test_label.shape)
        return self.train_data, self.train_label, self.valid_data, self.valid_label, self.test_data, self.test_label

    def shuffle_dataset(self, data):
        return np.random.shuffle(data)

    def split_dataset(self, data):
        self.train_data = self.dataset[0:20000, 1:]
        self.train_label = self.dataset[0:20000, 0]
        self.valid_data = self.dataset[20000:27945, 1:]
        self.valid_label = self.dataset[20000:27945, 0]
        self.test_data = self.dataset[27945:35888, 1:]
        self.test_label = self.dataset[27945:35888, 0]
        return self.train_data, self.train_label, self.valid_data, self.valid_label, self.test_data, self.test_label

    def normalize_dataset(self, data):
        temp = data.copy()
        temp.fill(128)
        data = (data-temp)/128
        return data
