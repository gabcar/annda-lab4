import numpy as np
from matplotlib import pyplot as plt
import os.path
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import model_from_json, load_model
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
    err = np.zeros((n_iter,2))

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
        test = rbm.gibbs(data['X'])
        train = rbm.gibbs(data['X'])
        err[i,1] = np.sum((test-data['X'])**2)/(n_features*len(data['X']))
        err[i,0] = np.sum((train-data['X'])**2)/(n_features*len(data['X']))

    return rbm, err

def trainAE(data, n_components, n_iter=200, batch_size=200, learning_rate=0.01):
    image_dim = data['X'].shape[1]
    input_img = Input(shape=(image_dim,))
    err = []

    encoder_layer = Dense(n_components, activation="relu")(input_img)
    decoder_layer = Dense(image_dim, activation="sigmoid")(encoder_layer)
    # this model maps an input to its reconstruction
    ae = Model(input_img, decoder_layer)

    ae.compile(optimizer='adadelta', loss='mean_squared_error')

    h = ae.fit(np.array(data['X']), np.array(data['X']),
            verbose=1,
            epochs=n_iter,
            batch_size=64,
            shuffle=True,
            validation_data=(data['X_tst'], data['X_tst']))

    err = h.history
    return ae, err

def assignment4_1():
    data = loadAll()
    N = [50, 75, 100, 150]
    test_img = data['plot_ims']

    for n in N:
        n_components = n
        n_iter_ae = 150
        n_iter_rbm = 75
        batch_size = 200

        if (os.path.isfile("ae_model_{}.h5".format(n_components))):
            #json_file = open('model.json', 'r')
            #loaded_model_json = json_file.read()
            #json_file.close()
            #ae = model_from_json(loaded_model_json)
            # load weights into new model
            #ae.load_weights("ae_model_4_1_{}.h5".format(n_components))
            #ae.compile(optimizer='sgd', loss='binary_crossentropy')
            #print("loaded exiting model\n")
            ae = load_model('ae_model_{}.h5'.format(n_components))
        else:
            ae, err_ae = trainAE(data, n_components, n_iter=n_iter_ae, batch_size=batch_size, learning_rate=0.01)

            model_json = ae.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            ae.save("ae_model{}.h5".format(n_components))


        if (os.path.isfile('rbm_model_{}.pkl'.format(n_components))):
            rbm = joblib.load('rbm_model_{}.pkl'.format(n_components))
        else:
            
            rbm, err_rbm = trainRBM(data, n_components, n_iter=n_iter_rbm, batch_size=batch_size, learning_rate=0.01)
            df = pd.DataFrame(err_rbm)
            df.to_csv("data/rbm_hidden_{}e_{}.csv".format(n_iter_rbm,n), header=["tr_loss","val_loss"])
            joblib.dump(rbm, 'rbm_model_{}.pkl'.format(n_components))

        gibbs = rbm.gibbs(test_img)

        ae_img = []
        err_ae = np.vstack((err_ae['loss'], err_ae['val_loss'])).T
        df = pd.DataFrame(err_ae)
        df.to_csv("data/ae_hidden_adadelta_{}e_{}.csv".format(n_iter_ae, n), header=["tr_loss","val_loss"])

        print(test_img[1].shape)
        for i in range(10):
            ae_img.append(ae.predict([np.array([test_img[i]])]))

        fig, ax = plt.subplots(2,5)
        fig.suptitle("RBM reconstruction error: {} epochs, {} nodes".format(n_iter_rbm, n))
        for i in range(10):
            plt.subplot(2,5,i+1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(gibbs[i].reshape(28,28), cmap='binary')
            plt.xlabel(str(i))

        fig.savefig('plots/3_1/rbm_{}e_{}_nodes.png'.format(n_iter_rbm,n))

        fig, ax = plt.subplots(2,5)
        fig.suptitle("AE reconstruction error: {} epochs, {} nodes".format(n_iter_ae, n))
        for i in range(10):
            plt.subplot(2,5,i+1)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.imshow(ae_img[i].reshape(28,28), cmap='binary')
            plt.xlabel(str(i))
        fig.savefig('plots/3_1/ae_adadelta_{}e_{}_nodes.png'.format(n_iter_ae, n))

        #fig, ax = plt.subplots(2,1, figsize=(10,5))
        fig = plt.figure()
        plt.plot(err_ae)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy [%]")
        plt.title("AE")

        fig.savefig('plots/3_1/error_adadelta_plots_{}e_ae_{}_nodes.png'.format(n_iter_ae,n))

        fig = plt.figure()
        plt.plot(err_rbm)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy [%]")
        plt.title("RBM")

        fig.savefig('plots/3_1/error_plots_rbm{}_{}_nodes.png'.format(n_iter_rbm, n))

        print("{} nodes complete.".format(n))

def assignment4_1_1():
    
    test_images = loadAll()['plot_ims']
    N = [50, 75, 100, 150]
    
    for n_components in N:
        ae = load_model('ae_model{}.h5'.format(n_components))
        rbm = joblib.load('rbm_model_{}.pkl'.format(n_components))
        fig_ae, ax_ae = plt.subplots(1, 10, figsize=(20, 2))
        fig_rbm, ax_rbm = plt.subplots(1, 10, figsize=(20, 2))
        
        for i in range(10):
            recall_ae = ae.predict([np.array([test_images[i]])])  # what is this black magic
            recall_rbm = rbm.gibbs(test_images[i])
            ax_ae[i].imshow(recall_ae.reshape(28, 28), cmap='binary')
            ax_rbm[i].imshow(recall_rbm.reshape(28, 28), cmap='binary')
            ax_ae[i].xaxis.set_visible(False)
            ax_ae[i].yaxis.set_visible(False)
            ax_rbm[i].xaxis.set_visible(False)
            ax_rbm[i].yaxis.set_visible(False)
        
        fig_ae.tight_layout()
        fig_rbm.tight_layout()
        fig_ae.savefig('plots/3_1_1/recall_ae_h{}.png'.format(n_components))
        fig_rbm.savefig('plots/3_1_1/recall_rbm_h{}.png'.format(n_components))
   
    fig_orig, ax_orig = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        ax_orig[i].imshow(test_images[i].reshape(28, 28), cmap='binary')
        ax_orig[i].xaxis.set_visible(False)
        ax_orig[i].yaxis.set_visible(False)
    fig_orig.tight_layout()
    fig_orig.savefig('plots/3_1_1/orig_images.png')


def assignment4_1_2(model):
    data = loadAll()
    n_components = 100
    batch_size = 200
    
    if model == 'ae':
        n_iter = 100
        ae, err_ae = trainAE(data, n_components, n_iter=n_iter, batch_size=batch_size, learning_rate=0.01)
        last_layer = ae.layers[2]
        weights = last_layer.get_weights()[0]
    elif model == 'rbm':
        n_iter = 20
        rbm, err_rbm = trainRBM(data, n_components, n_iter, batch_size)
        weights = rbm.components

    n_cols = 10
    n_rows = int(n_components / n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 7.5 if n_rows == 5 else 15))
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(weights[i].reshape(28,28), cmap='binary')
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()
    fig.savefig("plots/3_1_2/{}_last_layer_{}_components_{}e.png".format(model, n_components, n_iter))



def assignment4_2_DBN():
    data = loadAll()
    n_iter = 2
    n_iter_mlp = 5

    nodes = [200,150]#,100,50]
    no_of_layers = len(nodes)

    learning_rate = 0.01

    batch_size = 200

    train = data['X']
    test_img = data['plot_ims']
    rbms = []
    pipe = []
    acc = []

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

        rbms.append(rbm)
        pipe.append(('rbm{}'.format(i), rbm))

        print("pre-training step {}/{} done.".format(i+1,no_of_layers))

    mlp = MLPClassifier(solver='sgd', random_state=1, learning_rate="adaptive", learning_rate_init=0.01, hidden_layer_sizes=(nodes[no_of_layers-1],10), max_iter=n_iter_mlp, verbose=True)

    print(pipe)
    pipe.append(('mlp', mlp))
    print(pipe)

    clsf = Pipeline(pipe)

    clsf.fit(data['X'], np.ravel(data['T_trn']))

    predicted_classes = clsf.predict(data['X_tst'])

    acc.append(np.sum(data['T_tst'].T == predicted_classes)/len(data['T_tst']) * 100)

    print(acc)



def assignment4_2_AE():
    data = loadAll()
    n_iter = 20
    no_of_layers = 3
    nodes = [200,100,50]
    rbms = []
    learning_rate = 0.01

if __name__ == '__main__':
    #assignment4_1()
    assignment4_1_1()
    #assignment4_1_2('rbm')
    #assignment4_2_DBN()
    #assignment4_2_AE()
