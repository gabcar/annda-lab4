import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import BernoulliRBM
from sklearn.neural_network import MLPClassifier
from keras.layers import Input, Dense
from keras.models import Model


def loadData(path):
    data = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            dat = line.split(',')
            data.append(np.asarray([int(i) for i in dat]))
    return data

def loadAll():
    train_data = np.array(loadData("binMNIST_data/bindigit_trn.csv"))
    test_data = np.array(loadData("binMNIST_data/bindigit_tst.csv"))
    target_trn = np.array(loadData("binMNIST_data/targetdigit_tst.csv"))
    target_tst = np.array(loadData("binMNIST_data/targetdigit_tst.csv"))

    data = {"X":train_data, "X_tst":test_data, "T_trn":target_trn, "T_tst":target_tst}

    return data


def trainRBM(data, n_components, n_iter, batch_size, learning_rate=0.01):

    rbm = BernoulliRBM(verbose=True, random_state=None, batch_size=batch_size)

    n_features = len(data[0])
    rbm.n_iter = n_iter
    rbm.learning_rate = learning_rate

    #initialize the weight matrix with small (normally distributed) random values with hidden and visible biases initialized to 0.
    rbm.components = np.random.randn(n_components, n_features)*0.1
    rbm.intercept_hidden_ = np.zeros((n_components,))
    rbm.intercept_visible_ = np.zeros((n_features,))

    rbm.fit(data)

    return rbm

def trainAE(data, n_components, n_iter=200, batch_size=200, learning_rate=0.01):
    n_components = 50
    image_dim = data['X'].shape[1]
    input_img = Input(shape=(image_dim,))
    print(image_dim)

    encoder_layer = Dense(n_components, activation="relu")(input_img)
    decoder_layer = Dense(image_dim, activation="sigmoid")(encoder_layer)
    # this model maps an input to its reconstruction
    ae = Model(input_img, decoder_layer)

    ae.compile(optimizer='sgd', loss='binary_crossentropy')

    ae.fit(np.array(data['X']), np.array(data['X']),
                epochs=n_iter,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(data['X_tst'], data['X_tst']))

    return ae

def assignment4_1():
    data = loadAll()

    n_components = 150
    n_iter = 20
    batch_size = 200

    ae = trainAE(data, n_components, n_iter=200, batch_size=batch_size, learning_rate=0.001)
    rbm = trainRBM(data['X'], n_components, n_iter, batch_size, learning_rate=0.01)

    test_img = data['X_tst'][0]
    print(test_img.shape)
    gibbs = rbm.gibbs(test_img)
    ae_img = ae.predict([np.array([test_img])])

    fig, ax = plt.subplots(1,3)
    plt.subplot(1,3,1)
    plt.imshow(test_img.reshape(28,28), cmap='binary')

    plt.subplot(1,3,2)
    plt.imshow(gibbs.reshape(28,28), cmap='binary')

    plt.subplot(1,3,3)
    plt.imshow(np.round(ae_img).reshape(28,28), cmap='binary')
    plt.show()


assignment4_1()
