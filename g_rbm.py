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


def assignment4_1_2():
    model = 'ae'
    n_components = 100
    
    if model == 'ae':
        ae = load_model('ae_model{}.h5'.format(n_components))
        last_layer = ae.layers[1]
        weights = last_layer.get_weights()[0].T
    elif model == 'rbm':
        rbm = joblib.load('rbm_model_{}.pkl'.format(n_components))
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
    fig.savefig("plots/3_1_2/{}_last_layer_{}_components.png".format(model, n_components))


def assignment4_2_DBN():
    data = loadAll()
    n_iter = 75
    n_iter_mlp = 50

    nodes = []#[150, 100] #, 50]
    no_of_layers = len(nodes)

    learning_rate = 0.01

    batch_size = 200

    train = data['X']
    test_img = data['plot_ims']
    rbms = []
    pipe = []
    acc = []

    prev_layer_size = len(train[0])

    for i in range(no_of_layers):

        rbm = BernoulliRBM(n_components=nodes[i], verbose=True, batch_size=batch_size, random_state=1)
        rbm.learning_rate = learning_rate
        n_features = len(train[0])
        rbm.components = np.random.randn(nodes[i], prev_layer_size)*0.1
        rbm.intercept_hidden_ = np.zeros((nodes[i],))
        rbm.intercept_visible_ = np.zeros((prev_layer_size,))
        rbm.n_iter = n_iter

        #train = rbm.fit_transform(train)  # Enable to start pre-training

        rbms.append(rbm)
        pipe.append(('rbm{}'.format(i), rbm))

        prev_layer_size = nodes[i]

    mlp = MLPClassifier(
        solver='sgd',
        random_state=1,
        learning_rate="adaptive",
        learning_rate_init=0.01,
        hidden_layer_sizes=(784, 10),#(nodes[no_of_layers-1], 10),
        max_iter=n_iter_mlp,
        verbose=True
    )

    #print(pipe)
    pipe.append(('mlp', mlp))
    print(pipe)

    clsf = Pipeline(pipe)

    clsf.fit(data['X'], np.ravel(data['T_trn']))

    predicted_classes = clsf.predict(data['X_tst'])

    acc.append(np.sum(data['T_tst'].T == predicted_classes)/len(data['T_tst']) * 100)

    print(acc)

    joblib.dump(clsf, 'dbn_{}l.pkl'.format(no_of_layers))


def assignment4_2_AE():
    data = loadAll()
    n_iter = 150
    nodes = [150, 100, 50]
    n_layers = len(nodes)
    learning_rate = 0.01

    deep_input_layer = Input(shape=(784,))
    prev_layer = deep_input_layer

    training_data = data['X']
    validation_data = data['X_tst']
    prev_layer_size = 784

    for i in range(n_layers):
        input_layer = Input(shape=(prev_layer_size,))
        encoder_layer = Dense(nodes[i], activation='relu')(input_layer)
        decoder_layer = Dense(prev_layer_size, activation='sigmoid')(encoder_layer)
        ae = Model(inputs=input_layer, outputs=decoder_layer)
        ae.compile(optimizer='adadelta', loss='mean_squared_error')

        ae.fit(training_data, training_data,
               verbose=True,
               epochs=n_iter,
               batch_size=200,
               shuffle=True,
               validation_data=(validation_data, validation_data))

        encoder = Model(inputs=input_layer, output=encoder_layer)
        encoder.compile(optimizer='adadelta', loss='mean_squared_error')

        training_data = encoder.predict(training_data)
        validation_data = encoder.predict(validation_data)
        
        prev_layer = Dense(nodes[i],
                           weights=ae.layers[1].get_weights(),
                           activation='relu')(prev_layer)

        prev_layer_size = nodes[i]

    final_layer = Dense(10, activation='sigmoid')(prev_layer)
    sae = Model(inputs=deep_input_layer, output=final_layer)
    sae.compile(optimizer='adadelta', loss='mean_squared_error')
    
    vec_targets = get_vector_repr(data['T_trn'])
    vec_targets_tst = get_vector_repr(data['T_tst'])
    sae.fit(data['X'], vec_targets,
            verbose=True,
            epochs=n_iter,
            batch_size=200,
            shuffle=True,
            validation_data=(data['X_tst'], vec_targets_tst))

    predict = sae.predict(data['X_tst'])
    predict = np.argmax(predict, axis=1)
    accuracy = np.sum(data['T_tst'].T == predict) / len(data['T_tst'])
    print(accuracy)

    sae.save('sae_{}l.h5'.format(n_layers))

def get_vector_repr(data):
    vec = np.zeros(shape=(data.shape[0], 10))
    for i, t in enumerate(data):
        vec[i][t] = 1
    return vec


def eval_dbn():
    data = loadAll()
    n_layers = 3
    dbn = joblib.load('dbn_{}l.pkl'.format(n_layers))
    
    predicted = dbn.predict(data['X_tst'])
    print('Accuracy:', np.sum(data['T_tst'].T == predicted)/len(data['T_tst']))

    weights = dbn.named_steps['rbm{}'.format(n_layers - 1)].components
    n_cols = 10
    n_rows = int(weights.shape[0] / n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(int(1.5 * n_cols), int(1.5 * n_rows)))
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(weights[i].reshape(10, -1), cmap='binary')
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()
    fig.savefig("plots/3_2_1/dbn_{}l.png".format(n_layers))
    
    
def eval_sae():
    data = loadAll()
    n_layers = 3
    sae = load_model('sae_{}l.h5'.format(n_layers))
    
    predicted = sae.predict(data['X_tst'])
    predicted = np.argmax(predicted, axis=1)
    print('Accuracy:', np.sum(data['T_tst'].T == predicted)/len(data['T_tst']))

    weights = sae.layers[n_layers].get_weights()[0].T
    n_cols = 10
    n_rows = int(weights.shape[0] / n_cols)

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(int(1.5 * n_cols), int(1.5 * n_rows)))
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(weights[i].reshape(10, -1), cmap='binary')
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()
    fig.savefig("plots/3_2_1/sae_{}l.png".format(n_layers))


def check_activation_dbn():
    data = loadAll()
    digits = data['plot_ims']
    
    n_layers = 3
    dbn = joblib.load('dbn_{}l.pkl'.format(n_layers))

    digit_activation_map = {}
    for d, img_data in enumerate(digits):
        activations = []
        curr_act = img_data
        for l in range(n_layers):
            layer = dbn.named_steps['rbm{}'.format(l)]
            [curr_act] = layer.transform([curr_act])
            activations.append(curr_act)
        digit_activation_map[d] = activations

    for d in range(10):
        activation = digit_activation_map[d]
        fig, ax = plt.subplots(1, 3, figsize=(4.5, 1.5))
        for i in range(n_layers):
            ax[i].imshow(activation[i].reshape(10, -1), cmap='binary')
            ax[i].xaxis.set_visible(False)
            ax[i].yaxis.set_visible(False)
        fig.tight_layout()
        fig.savefig('plots/3_2_2/dbn_d{}_activations.png'.format(d))


def check_activation_sae():
    data = loadAll()
    digits = data['plot_ims']
    
    n_layers = 3
    sae = load_model('sae_{}l.h5'.format(n_layers))
    digit_activation_map = {}
    for d, img_data in enumerate(digits):
        activations = []
        for l in range(n_layers):
            m = Model(inputs=sae.input, outputs=sae.get_layer(index=l + 1).output)
            m.compile(optimizer='adadelta', loss='mean_squared_error')
            activations.append(m.predict([np.array([img_data])]))
        digit_activation_map[d] = activations

    for d in range(10):
        activation = digit_activation_map[d]
        fig, ax = plt.subplots(1, 3, figsize=(4.5, 1.5))
        for i in range(n_layers):
            ax[i].imshow(activation[i].reshape(10, -1), cmap='binary')
            ax[i].xaxis.set_visible(False)
            ax[i].yaxis.set_visible(False)
        fig.tight_layout()
        fig.savefig('plots/3_2_2/sae_d{}_activations.png'.format(d))


if __name__ == '__main__':
    #assignment4_1()
    #assignment4_1_1()
    #assignment4_1_2()
    #assignment4_2_DBN()
    #assignment4_2_AE()
    #eval_dbn()
    #eval_sae()
    #check_activation_dbn()
    check_activation_sae()

