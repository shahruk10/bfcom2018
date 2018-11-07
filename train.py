
import os
import sys
import numpy as np

from utils import ensureDir, loadConfig, saveConfig, getData, getOptimizer
from modelLib import makeModel
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, CSVLogger, ReduceLROnPlateau

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
    cfgPath = sys.argv[1] if len(sys.argv) > 1 else './config.toml'
    cfg = loadConfig(cfgPath)
    
    try:
        # ... and unpacking variables
        dictget = lambda d, *k: [d[i] for i in k]

        dataStats = cfg['data_stats']
        modelParams = cfg['model_params']
        trainCSV, testCSV = dictget(cfg['database'], 'train', 'test')
        seqLength, stepSize = dictget(cfg['model_params'], 'seqLength', 'stepSize')
        modelArch, modelDir, modelName = dictget(cfg['model_arch'], 'modelArch', 'modelDir', 'modelName')
        optimizer, lossFunc, metricFuncs = dictget(cfg['training_params'], 'optimizer', 'lossFunc', 'metricFuncs')
        lr, epochs, batchSize, patience,= dictget(cfg['training_params'],'learningRate', 'epochs', 'batchSize','patience')
    except KeyError as err:
        print("\n\nERROR: not all parameters defined in config.toml : ", err)
        print("Exiting ... \n\n")
        sys.exit(1)

    print("Loading training data ...")
    xTrain, yTrain, stats = getData(trainCSV, seqLength=seqLength, stepSize=stepSize, stats=dataStats)
    print("Training Data Shape : ", xTrain.shape, "\n")
    
    print("Loading testing data ...")
    xTest, yTest, stats = getData(testCSV, seqLength=seqLength, stepSize=stepSize, stats=dataStats)
    print("Testing Data Shape : ", xTest.shape, "\n")
    
    yTrain = np.expand_dims(yTrain,-1)  # adding extra axis as model expects 2 axis in the output
    yTest = np.expand_dims(yTest, -1)

    print("Compiling Model")
    opt = getOptimizer(optimizer,lr)
    model = makeModel(modelArch, modelParams, verbose=True)
    model.compile(loss=lossFunc, optimizer=opt, metrics=metricFuncs)

    # setting up directories
    modelFolder = os.path.join(modelDir, modelName)
    weightsFolder = os.path.join(modelFolder, "weights")
    bestModelPath = os.path.join(weightsFolder, "best.hdf5" )
    ensureDir(bestModelPath)

    saveConfig(cfgPath, modelFolder)


     # callbacks
    monitorMetric = 'val_loss'
    check1 = ModelCheckpoint(os.path.join(weightsFolder, modelName + "_{epoch:03d}.hdf5"), monitor=monitorMetric, mode='auto')
    check2 = ModelCheckpoint(bestModelPath, monitor=monitorMetric, save_best_only=True, mode='auto')
    check3 = EarlyStopping(monitor=monitorMetric, min_delta=0.01, patience=patience, verbose=0, mode='auto')
    check4 = CSVLogger(os.path.join(modelFolder, modelName +'_trainingLog.csv'), separator=',', append=True)
    check5 = ReduceLROnPlateau(monitor=monitorMetric, factor=0.1, patience=patience//3, verbose=1, mode='auto', min_delta=0.001, cooldown=0, min_lr=1e-10)

    cb = [check2, check3, check4,check5]
    if cfg['training_params']['saveAllWeights']:
        cb.append(check1)

    print("Starting Training ...")
    model.fit(x=xTrain, y=yTrain, batch_size=batchSize, epochs=epochs, verbose=1, 
            callbacks=cb, validation_data=(xTest, yTest), shuffle=True)

if __name__ == '__main__':
    main()
