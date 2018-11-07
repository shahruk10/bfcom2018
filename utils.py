import os
import toml
import shutil
import numpy as np
from keras.optimizers import Adam, Nadam, Adamax, Adagrad, Adadelta, RMSprop, SGD

xNames = ['Date', 'Month', 'Year', 'Holiday', 'Weekend', 'Hour', 'Temperature']

def ensureDir(filePath):
    ''' checks if the folder at filePath exists. If not, it creates it. '''

    directory = os.path.dirname(filePath)

    if not os.path.exists(directory):
        os.makedirs(directory)

def loadConfig(cfgPath):
    ''' loads toml file containing configurations for model training '''

    with open(cfgPath, 'r') as df:
        tomlString = df.read()
    cfg = toml.loads(tomlString)
    return cfg

def saveConfig(cfgPath, savePath):
    ''' copies config file to model directory '''

    ensureDir(os.path.join(savePath, 'config.toml'))
    shutil.copy(cfgPath, os.path.join(savePath, 'config.toml'))

def getOptimizer(optimizer, learningRate):
    ''' returns optimizer with given learn rate '''

    if optimizer == 'adam':
        opt = Adam(lr=learningRate, beta_1=0.9,beta_2=0.999, epsilon=None, decay=0.00001)
    if optimizer == 'nadam':
        opt = Nadam(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    if optimizer == 'adamax':
        opt = Adamax(lr=learningRate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
    elif optimizer == 'sgd':
        opt = SGD(lr=learningRate, momentum=0.0, decay=1e-4, nesterov=False)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr=learningRate, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer == 'adagrad':
        opt = Adagrad(lr=learningRate, epsilon=None, decay=0.0)
    elif optimizer == 'adadelta':
        opt = Adadelta(lr=learningRate, rho=0.95, epsilon=None, decay=0.0)

    return opt

def getData(pathToCSV,  stats, seqLength = None, stepSize = 5):
    ''' loads and normalizes data, ready for training / testing '''

    X, Y = loadDataFromCSV(pathToCSV)
    X, Y, stats = normalizeData(X,Y,stats)
    X = np.moveaxis(X, 0, -1)

    if seqLength is not None:
        X, Y = getSequences(X, Y, seqLength, stepSize)

    return X,Y,stats

def loadDataFromCSV(pathToCSV):
    ''' reads data off of csv file and returns as np array '''

    data = np.genfromtxt(pathToCSV, delimiter=',', names=True, dtype=np.float32)
    Y = data['Load']   
    X = data[xNames[0]]
    for n in xNames[1:]: 
        X = np.vstack((X,data[n]))

    return X,Y

def normalizeData(X,Y, stats):
    ''' normalizes X Y data either by min-max or mean-std where appropriate '''

    lmin = stats['lmin']
    lmax = stats['lmax']
    tmin = stats['tmin']
    tmax = stats['tmax']
    yearStart = stats['yearStart']
    yearEnd = stats['yearEnd']

    lmean = np.mean(Y)
    lstd = np.std(Y)
    Y[...] = (Y[...] - lmin)/(lmax-lmin)

    for idx, n in enumerate(xNames):
        if n == 'Date':
            X[idx, :] = (X[idx, :]-1) / 31.0
        elif n == 'Month' :
            X[idx, :] = (X[idx, :]-1) / 12.0
        elif n == 'Year' :
            X[idx, :] = (X[idx, :] - yearStart) / (yearEnd - yearStart) 
        elif n == 'Hour' :
            X[idx, :] = (X[idx, :] - 1) / 24.0
        elif n == 'Temperature':
            tmean = np.mean(X[idx, :])
            tstd = np.std(X[idx, :])
            X[idx, :] = (X[idx, :] - tmin) / (tmax-tmin)

    stats['lmean'] = lmean
    stats['lstd'] = lstd
    stats['tmean'] = tmean
    stats['tstd'] = tstd

    return X,Y, stats 

def denormalizePrediction(Y, stats):
    ''' normalizes X Y data either by min-max or mean-std where appropriate '''

    lmin = stats['lmin']
    lmax = stats['lmax']
    Y[...] = (Y[...] * (lmax-lmin)) + lmin

    return Y 

def getSequences(x, y, n, stepSize=1, pad=True):
    ''' returns sequences of x and y containing n instances in each sequence. x and y
    can be any arrays of the same length. Sequences can be made to overlap with any step 
    size you fancy. Sequences are created by indexing the 0th axis. If the the number of 
    elements in x or y is not divisible by n, sequences with less than n elements are ignored. '''

    # highest index to go upto that will ensure all sequences have n elemetns
    maxIndex = x.shape[0] - n + 1
    
    xseq = [x[i:i+n, ...] for i in range(0, maxIndex, stepSize)]
    yseq = [y[i:i+n, ...] for i in range(0, maxIndex, stepSize)]

    return np.array(xseq), np.array(yseq)

def deSequence(yseq,n,stepSize):
    ''' unroll sequences and average where they overlap,
    creating a single array arrange in order of date containing
    predictions '''
    
    deSeqLength = ((yseq.shape[0] - 1) * stepSize) + n

    y = np.zeros(deSeqLength, dtype=np.float32)
    ycount = np.zeros(deSeqLength, dtype=np.float32)

    idx = 0
    for seq in yseq:
        y[idx:idx+n] = y[idx:idx+n] + seq[...]
        ycount[idx:idx+n] = ycount[idx:idx+n] + np.ones((n,))
        idx += stepSize

    for idx in range(len(y)):
        y[idx] = y[idx] / ycount[idx]

    return y
