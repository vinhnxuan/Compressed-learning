import os 
import numpy as np
import pickle

def get_data_set(name="train", cifar=10):
    X = None
    Y = None
    L = None
    
    def dense_to_one_hot(labels_dense, num_classes=10):
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes))
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

        return labels_one_hot
    
    folder_name = "cifar_10" if cifar == 10 else "cifar_100"

    f = open('data_set/'+folder_name+'/batches.meta', 'rb')
    datadict = pickle.load(f, encoding='latin1')
    f.close()
    L = datadict['label_names']

    if name is "train":
        for i in range(5):
            f = open('./data_set/'+folder_name+'/data_batch_' + str(i + 1), 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            _X = datadict["data"]
            _Y = datadict['labels']

            _X = np.array(_X, dtype=float) / 255.0
            _X = _X.reshape([-1, 3, 32, 32])
            #_X = _X.transpose([0, 2, 3, 1])
            _X = _X.reshape(-1,3*32*32)

            if X is None:
                X = _X
                Y = _Y
            else:
                X = np.concatenate((X, _X), axis=0)
                Y = np.concatenate((Y, _Y), axis=0)

    elif name is "test":
        f = open('./data_set/'+folder_name+'/test_batch', 'rb')
        datadict = pickle.load(f, encoding='latin1')
        f.close()

        X = datadict["data"]
        Y = np.array(datadict['labels'])

        X = np.array(X, dtype=float) / 255.0
        X = X.reshape([-1, 3, 32, 32])
        #X = X.transpose([0, 2, 3, 1])
        X = X.reshape(-1, 3*32*32)
    return X,dense_to_one_hot(Y),L


def load_mnist(batch_size, is_training=True):
    path = os.path.normpath("/home/ads/gw438/vinh/MNIST_data/")
    
    
    if is_training:
        
        fd = open(os.path.join(path, 'train-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float32)
        
        trainX=trainX.reshape(60000,784)
        

        fd = open(os.path.join(path, 'train-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        trainY = loaded[8:].reshape((60000)).astype(np.int32)

        trX = trainX[:60000] / 255.
        trY = trainY[:60000]

        valX = trainX[50000:] / 255.
        valY = trainY[50000:]

        num_tr_batch = 60000 // batch_size
        num_val_batch = 10000 // batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch
    else:
        fd = open(os.path.join(path, 't10k-images.idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)
        
        teX=teX.reshape(10000,784)

        fd = open(os.path.join(path, 't10k-labels.idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.int32)

        num_te_batch = 10000 // batch_size
        return teX / 255., teY, num_te_batch