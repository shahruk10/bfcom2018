import sys
from keras.models import Model

# core layers
from keras.layers import Input, Dense, Dropout,  BatchNormalization
from keras.layers import LSTM
from keras.layers.advanced_activations import PReLU, LeakyReLU, ReLU

# other utils
from keras.utils import plot_model


modelArch = {}												
addModel = lambda f:modelArch.setdefault(f.__name__,f)  # decorator to add defined models to dictionary

# build and return model
def makeModel(architecture, params, verbose=False):

    model = modelArch[architecture](params)
    if verbose:
        print(model.summary(line_length=150))
        plot_model(model, to_file='./model_%s.png'%architecture)

    return model

@addModel
def LSTM01(params):

    i = Input((params['seqLength'], params['inputDim']), name='input')

    # hidden layers
    h = i
    for idx, (units, actFunc) in enumerate(zip(params['denseUnits'], params['denseActivation'])):

        h = Dense(  units, 
                    activation='linear' if actFunc in ['prelu', 'leakyrelu'] else actFunc, 
                    kernel_initializer='he_normal')(h)
            
        if actFunc == 'prelu':
            h = PReLU(alpha_initializer='zeros')(h)
        elif actFunc == 'leakyrelu':
            h = LeakyReLU()(h)
        
        if params['batchnorm']:
            h = BatchNormalization()(h)

        # dropout after 2 dense layers until last two dense layers
        if idx%2 and idx < len(params['denseUnits']) - 2:   
            h = Dropout(params['dropout'])(h)
            pass
    
    # LSTM layers
    for idx, (units, actFunc) in enumerate(zip(params['lstmUnits'], params['lstmActivation'])):
        h = LSTM(units, activation=actFunc, stateful=False, return_sequences=True)(h)
    
    return Model(inputs=[i],outputs=[h])
