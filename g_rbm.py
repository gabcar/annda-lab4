import numpy as np
from matplotlib import pyplot as plt
import os.path
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import model_from_json
from rbm import BernoulliRBM
import pandas as pd


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
    target_trn = np.array(loadData("binMNIST_data/targetdigit_trn.csv"))
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

    rbm = BernoulliRBM(verbose=True, batch_size=batch_size, random_state=1, n_components=n_components)

    n_features = len(data['X'][0])
    rbm.learning_rate = learning_rate

    #initialize the weight matrix with small (normally distributed) random values with hidden and visible biases initialized to 0.
    rbm.components = np.random.randn(n_components, n_features)*0.1
    rbm.intercept_hidden_ = np.zeros((n_components,))
    rbm.intercept_visible_ = np.zeros((n_features,))


    rbm.n_iter = 1
    for i in range(n_iter):
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
        n_iter_ae = 200
        n_iter_rbm = 20
        batch_size = 200

        if (os.path.isfile("ae_model_{}.h5".format(n_components))):
            json_file = open('model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            ae = model_from_json(loaded_model_json)
            # load weights into new model
            ae.load_weights("ae_model_4_1_{}.h5".format(n_components))
            ae.compile(optimizer='sgd', loss='binary_crossentropy')
            print("loaded exiting model\n")
        else:
            ae, err_ae = trainAE(data, n_components, n_iter=n_iter_ae, batch_size=batch_size, learning_rate=0.01)
            df = pd.DataFrame(err_ae)
            df.to_csv("data/ae_hidden_{}.csv".format(n))
            model_json = ae.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            ae.save_weights("ae_model{}.h5".format(n_components))


        rbm, err_rbm = trainRBM(data, n_components, n_iter=n_iter_rbm, batch_size=batch_size, learning_rate=0.01)
        df = pd.DataFrame(err_rbm)
        df.to_csv("data/rbm_hidden_{}.csv".format(n))

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
        fig.suptitle("AE reconstruction error: {} epochs, {} nodes".format(n_iter_ae, n))
        for i in range(10):
            plt.subplot(2,5,i+1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(ae_img[i].reshape(28,28) >= 0.5, cmap='binary')
            plt.xlabel(str(i))
        fig.savefig('plots/3_1/ae_{}_nodes.png'.format(n))

        #fig, ax = plt.subplots(2,1, figsize=(10,5))
        fig = plt.figure()
        plt.plot(err_ae)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy [%]")
        plt.title("AE")

        fig.savefig('plots/3_1/error_plots_ae_{}_nodes.png'.format(n))

        fig = plt.figure()
        plt.plot(err_rbm)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy [%]")
        plt.title("RBM")

        fig.savefig('plots/3_1/error_plots_rbm_{}_nodes.png'.format(n))

        print("{} nodes complete.".format(n))

def assignment4_1_2():
    data = loadAll()
    n_components = 100
    n_iter_ae = 100
    n_iter_rbm = 20
    batch_size = 200

    if (os.path.isfile("ae_model_{}.h5".format(n_components))):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ae = model_from_json(loaded_model_json)
        # load weights into new model
        ae.load_weights("ae_model_{}.h5".format(n_components))
        ae.compile(optimizer='sgd', loss='binary_crossentropy')
        print("loaded exiting model\n")
    else:
        ae, err_ae = trainAE(data, n_components, n_iter=n_iter_ae, batch_size=batch_size, learning_rate=0.01)
        model_json = ae.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        ae.save_weights("ae_model{}.h5".format(n_components))

    last_layer = ae.layers[2]

    plt.subplots(2,5)
    for i in range(10):
        plt.subplot(2,5,i+1)
        plt.imshow(last_layer.get_weights()[0][i].reshape(28,28), cmap='binary')

    plt.show()
    #rbm, err_rbm = trainRBM(data, n_components, n_iter=n_iter_rbm, batch_size=batch_size, learning_rate=0.01)


def assignment4_2():
    data = loadAll()
    n_iter = 20
    no_of_layers = 3
    nodes = [200,100,50]
    rbms = []
    learning_rate = 0.01

    batch_size = 200

    pipe = []
    train = data['X']
    for i in range(no_of_layers):
        rbm = BernoulliRBM(n_components=nodes[i], verbose=True, batch_size=batch_size, random_state=1)

        rbm.learning_rate = learning_rate

        n_features = len(train[0])

        #initialize the weight matrix with small (normally distributed) random values with hidden and visible biases initialized to 0.
        rbm.components = np.random.randn(nodes[i], n_features)*0.1
        rbm.intercept_hidden_ = np.zeros((nodes[i],))
        rbm.intercept_visible_ = np.zeros((n_features,))
        rbm.n_iter = n_iter

        train = rbm.fit_transform(train)  # Enable to start pre-training

        print(rbm.get_params())
        print(rbm.get_params(deep=True))

        rbms.append(rbm)
        pipe.append(('rbm{}'.format(i), rbm))

    mlp = MLPClassifier(solver='sgd', random_state=1, learning_rate="adaptive", learning_rate_init=0.01, hidden_layer_sizes=(nodes[no_of_layers-1],10))

    pipe.append(('mlp', mlp))

    clsf = Pipeline(pipe)

    clsf.fit(data['X'], np.ravel(data['T_trn']))

    predicted_classes = clsf.predict(data['X_tst'])

    print(predicted_classes)
    print(data['T_tst'])

    acc = np.sum(data['T_tst'].T == predicted_classes)/len(data['T_tst']) * 100
    print(acc)

if __name__ == '__main__':
    assignment4_1()
    #assignment4_1_2()
    #assignment4_2()
