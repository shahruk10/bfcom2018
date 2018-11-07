
import os
import sys
import numpy as np

from utils import getData, ensureDir, loadConfig, denormalizePrediction, deSequence
from modelLib import makeModel
from keras import backend
from keras.models import load_model
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# setting RNG seeds
tf.set_random_seed(2727)
np.random.seed(2727)


def main():

    # loading config file ...
    if len(sys.argv) > 1:
        cfgPath = sys.argv[1]
    else:
        print("ERROR : Must provide path to config.toml for model to test")
        sys.exit(1)
    cfg = loadConfig(cfgPath)
    
    try:
        modelDir = cfg['model_arch']['modelDir']
        modelName = cfg['model_arch']['modelName']
        batchSize = cfg['training_params']['batchSize']
        dataStats = cfg['data_stats']
        testCSV = cfg['database']['test']
        seqLength = cfg['model_params']['seqLength']
        stepSize = cfg['model_params']['stepSize']

    except KeyError as err:
        print("\n\nERROR: not all parameters defined in config.toml : ", err)
        print("Exiting ... \n\n")
        sys.exit(1)

    print("\n\n", "".join(['-']*100))
    print("Loading testing data ...")
    xTest, yTest, stats = getData(testCSV, seqLength=seqLength, stepSize=stepSize, stats=dataStats)
    yTest = np.expand_dims(yTest, -1)

    # setting up directories
    modelFolder = os.path.join(modelDir, modelName)
    weightsFolder = os.path.join(modelFolder, "weights")
    bestModelPath = os.path.join(weightsFolder, "best.hdf5")

    print("Loading Model")
    model = load_model(bestModelPath)

    ypred = model.predict(xTest, batch_size=batchSize, verbose=True)

    ygt = yTest.copy()
    ypred = np.array(ypred[...,0])
    ygt = np.array(ygt[...,0])
    ypred = deSequence(ypred, seqLength, stepSize)
    ygt = deSequence(ygt, seqLength, stepSize)

    ypred = denormalizePrediction(ypred, stats)
    ygt = denormalizePrediction(ygt, stats)

    err = ypred-ygt
    err2 = err ** 2
    mse = np.mean( err2 )
    rmse = np.sqrt(mse)
    print("".join(['-']*100), "\n\n")
    print("RMSE : %f" % rmse)

    # writing results to disk
    data = np.genfromtxt(testCSV, delimiter=',', names=True)

    with open( os.path.join(modelFolder,'%s_predictions.csv' % modelName), 'w') as df:
        df.write("Year, Month, Hour,Predictions, GT\n")
        
        for idx in range(len(ypred)):
            df.write("%d, %d, %d , %f, %f\n" % (data['Year'][idx],
                                                data['Month'][idx],
                                                data['Hour'][idx],
                                                ypred[idx],
                                                ygt[idx]))

    plt.figure(1, figsize=(20,5), dpi=300)
    plotRange = 1000
    plt.plot(np.arange(len(ygt[0:plotRange])), ygt[0:plotRange], ms=1, label="Actual")
    plt.plot(np.arange(len(ypred[0:plotRange])), ypred[0:plotRange], ms=1,label="Prediction")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(modelFolder, '%s_predictions.png' % modelName), dpi=300)
    plt.show()
if __name__ == '__main__':
    main()
