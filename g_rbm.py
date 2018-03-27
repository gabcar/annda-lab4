import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import BernoulliRBM
from keras.callbacks import ModelCheckpoint
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
    n = 0
    plot_ims = []
    train_data = np.array(loadData("binMNIST_data/bindigit_trn.csv"))
    test_data = np.array(loadData("binMNIST_data/bindigit_tst.csv"))
    target_trn = np.array(loadData("binMNIST_data/targetdigit_tst.csv"))
    target_tst = np.array(loadData("binMNIST_data/targetdigit_tst.csv"))

    for i, im in enumerate(test_data):
        if n == target_tst[i]:
            plot_ims.append(im)
            n += 1

    data = {"X":train_data, "X_tst":test_data, "T_trn":target_trn, "T_tst":target_tst, "plot_ims":plot_ims}

    return data


def trainRBM(data, n_components, n_iter, batch_size, learning_rate=0.01):
    acc = []
    err = []

    rbm = BernoulliRBM(verbose=True, batch_size=batch_size, random_state=1)

    n_features = len(data['X'][0])
    rbm.learning_rate = learning_rate

    #initialize the weight matrix with small (normally distributed) random values with hidden and visible biases initialized to 0.
    rbm.components = np.random.randn(n_components, n_features)*0.1
    rbm.intercept_hidden_ = np.zeros((n_components,))
    rbm.intercept_visible_ = np.zeros((n_features,))

    for i in range(n_iter):
        rbm.n_iter = i+1
        rbm.fit(data['X'])

        for image in data['X_tst']:
            test = rbm.gibbs(image)
            acc.append(np.sum(np.abs(test-image)))
        err.append(np.mean(acc)/n_features)
        acc = []

    return rbm, err

def trainAE(data, n_components, n_iter=200, batch_size=200, learning_rate=0.01):
    image_dim = data['X'].shape[1]
    input_img = Input(shape=(image_dim,))
    err = []
    acc = []

    encoder_layer = Dense(n_components, activation="relu")(input_img)
    decoder_layer = Dense(image_dim, activation="hard_sigmoid")(encoder_layer)
    # this model maps an input to its reconstruction
    ae = Model(input_img, decoder_layer)

    ae.compile(optimizer='sgd', loss='binary_crossentropy')

    for i in range(n_iter):
        ae.fit(np.array(data['X']), np.array(data['X']),
                verbose=0,
                epochs=1,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(data['X_tst'], data['X_tst']))
        print("AE Epoch: {}".format(i))
        for image in data['X_tst']:
            test = ae.predict([np.array([image])]) > 0.5
            acc.append(np.sum(np.abs(test-image)))
        err.append(np.mean(acc)/image_dim)
        acc = []

    return ae, err

def assignment4_1():
    data = loadAll()
    N = [50, 75, 100, 150]
    test_img = data['plot_ims']
    for n in N:
        n_components = n
        n_iter_ae = 20
        n_iter_rbm = 20
        batch_size = 200

        ae, err_ae = trainAE(data, n_components, n_iter=n_iter_ae, batch_size=batch_size, learning_rate=0.001)

        rbm, err_rbm = trainRBM(data, n_components, n_iter=n_iter_rbm, batch_size=batch_size, learning_rate=0.01)

        gibbs = rbm.gibbs(test_img)
        ae_img = ae.predict([np.array(test_img)])

        fig, ax = plt.subplots(1,3)
        plt.subplot(1,3,1)
        plt.imshow(test_img[0].reshape(28,28), cmap='binary')

        plt.subplot(1,3,2)
        plt.imshow(gibbs[0].reshape(28,28), cmap='binary')

        plt.subplot(1,3,3)
        plt.imshow((ae_img[0] > 0.5).reshape(28,28), cmap='binary')

        fig.savefig('plots/3_1/rbm_ae_{}_nodes.png'.format(n))

        fig, ax = plt.subplots(2,5)
        fig.suptitle("RBM reconstruction error: {} epochs, {} nodes".format(n_iter_rbm, n))
        for i in range(10):

            plt.subplot(2,5,i+1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(gibbs[i].reshape(28,28), cmap='binary')
            plt.xlabel(str(i))

        fig.savefig('plots/3_1/rbm_{}_nodes.png'.format(n))

        fig, ax = plt.subplots(2,5)
        fig.suptitle("RBM reconstruction error: {} epochs, {} nodes".format(n_iter_ae, n))
        for i in range(10):

            plt.subplot(2,5,i+1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(gibbs[i].reshape(28,28), cmap='binary')
            plt.xlabel(str(i))

        fig.savefig('plots/3_1/ae_{}_nodes.png'.format(n))

        fig, ax = plt.subplots(2,1)
        plt.subplot(2,1,1)
        plt.plot(err_ae)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy [%]")

        plt.subplot(2,1,2)
        plt.plot(err_rbm)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy [%]")

        fig.savefig('plots/3_1/error_plots_ae_rbm_{}_nodes.png'.format(n))

        print("{} nodes complete.".format(n))


if __name__ == '__main__':
    assignment4_1()
