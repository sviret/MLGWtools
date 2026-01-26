import tensorflow as tf
import keras
#from keras import ops as K
from keras import layers, models, initializers, optimizers
import numpy as np

import os
import pickle

from ncps import wirings
from ncps.keras import CfC
from ncps.keras import LTC
#from LTC4mod import LTCCell
#from LTC4mod import ODESolver

#import tensorflow.experimental.numpy as tnp
#from tensorflow.python.ops.numpy_ops import np_config
#np_config.enable_numpy_behavior()
#np.set_printoptions(threshold=np.inf)

'''
Class defining the network

'''

class CNN_Liquid():

    '''
    Initialization: here we define the CNN LSTM encoder
    
    '''
    
    def __init__(self,training):

        self.__netinfo=training.netprop()
        self.__lr=training.lr()
        self.__kindTraining=training.kt()
        self.__epochs=training.listEpochs()
        self.__SNRs=training.listSNR() 
        self.__initializer = initializers.GlorotNormal() # Xavier initialization
        self.__opt = optimizers.Adam(learning_rate=self.__lr,clipvalue=1.)
        self.__loss=tf.keras.losses.MeanSquaredError()
        #self.__loss = custom_loss_function(0)

        self.list_chunks = []
        self.__net = []
        self.__npts = 0
        self.__weight = [] 

        nValue=len(self.__netinfo[0])

        self.__nbands=nValue  # The number of frequency bands

        self.__fs = int(self.__netinfo[1][0])

        self.__listTtot=self.__netinfo[0]
        self.__listfe=self.__netinfo[1]
        self.__Ttot=sum(self.__listTtot)
        self.__fe=max(self.__listfe)
        
        #custom_loss = custom_loss_function(d)


        # The number of data points for each band is stored
        for x in range(nValue):
            npts=self.__listTtot[x]*self.__listfe[x]
            self.list_chunks.append(npts)
            self.__npts+=int(npts)
                    
        self.singleBand=False
        if self.__nbands==1:
            self.singleBand=True
            self.__weight.append(1.)
        else:
            for i in range(self.__nbands):
                self.__weight.append(1.)
        self.__weight=np.asarray(self.__weight)

    '''
    CNN_LSTM 2/
    
    Network definition:  

    Some network examples are defined here
    '''

    '''

    '''
 

    def Liquid_denoiser(self,Liquid_neurons=10):

        self.__units = Liquid_neurons # Number of neurons (100 in the paper)
        mode='concat'
        size=self.__npts
        self.__feature=4

            
        # Define the networks (one per band)
        self.inputs=[]
        self.outputs=[]
        fc_wiring = wirings.AutoNCP(self.__units, 4)

        print("Initialize the CNN-LSTM denoising network")

        with tf.name_scope('CNN_LSTM') as scope:
            input=layers.Input(shape=(size,self.__feature,1))
            #Encoder
            x=layers.TimeDistributed(layers.Conv1D(filters=32, kernel_size=1, activation='relu', name='1er_Conv1D'))(input)
            x=layers.TimeDistributed(layers.MaxPool1D(pool_size=2, strides=2, name='MaxPool1D'))(x)
            x=layers.TimeDistributed(layers.Conv1D(filters=16, kernel_size=1, activation='relu', name='2eme_Conv1D'))(x)
            x=layers.TimeDistributed(layers.Flatten())(x)

            #x=layers.Conv1D(filters=32, kernel_size=1, activation='relu', name='1er_Conv1D')(input)
            #x=layers.Flatten()(x)
            #Decoder
            #x=layers.Bidirectional(layers.RNN(LTCCell(num_units=20, solver=ODESolver.SemiImplicit), return_sequences=True),name='1er_LTCSe')(x) 
            #x=layers.Bidirectional(layers.RNN(LTCCell(num_units=20, solver=ODESolver.SemiImplicit), return_sequences=True),name='2eme_LTCSe')(x)
            #x=layers.Bidirectional(layers.RNN(LTCCell(num_units=20, solver=ODESolver.SemiImplicit), return_sequences=True),name='3eme_LTCSe')(x)  
            #x=LTC(fc_wiring, return_sequences=True,mixed_memory=True, name='LNN_layer')(x)
            x=layers.Bidirectional(CfC(fc_wiring, return_sequences=True,mixed_memory=True), name='1er_BiCfC',
                            merge_mode=mode)(x)
            output=layers.TimeDistributed(layers.Dense(1))(x) #mean_squared_error
                
            self.inputs.append(input)
            self.outputs.append(output)

        self.model = models.Model(self.inputs, self.outputs)
        
        # Init the model
        self.model.compile(optimizer=self.__opt,metrics=[tf.keras.metrics.LogCoshError()],loss=self.__loss)

        # Print the network
        print(self.model.summary())


    '''
    CNN_LSTM 3/
    
    Getters
    '''


    def getNetSize(self):
        return self.__npts
    
    def getFeatureSize(self):
        return self.__feature
    
    def getBlockLength(self):
        return self.__Ttot
    
    def getStepSize(self):
        return self.list_chunks[0]/2
        
    def getlchunk(self,i):
        return self.list_chunks[i]

    def getfs(self):
        return self.__fs
        
    def getNband(self):
        return self.__nbands

    def model(self):
        return self.model
     
    def getListFe(self):
        return self.__listfe
 
    def getListTtot(self):
        return self.__listTtot

    # Here we save the network info

    def save(self):
        fichier='autoencoder_network-'+self.__kindTraining+'-'+str(self.__lr)+'-net-1.h5'
        fichier_js='autoencoder_network-'+self.__kindTraining+'-'+str(self.__lr)+'-net-1_fullnet.p'
        c=1
        while os.path.isfile(fichier):
            c+=1
            fichier='autoencoder_network-'+self.__kindTraining+'-'+str(self.__lr)+'-net-'+str(c)+'.h5'
            fichier_js='autoencoder_network-'+self.__kindTraining+'-'+str(self.__lr)+'-net-'+str(c)+'_fullnet.p'
        self.model.save(fichier)
        f=open(fichier_js, mode='wb')
        pickle.dump([self.__netinfo,self.model],f)
        f.close()
        self.model=None        
        
 
    
    @property
    def trainGenerator(self):
        return self.__trainGenerator

    @property
    def testGenerator(self):
        return self.__testGenerator

    @property
    def cTrainSet(self):
        return self.__cTrainSet

    @property
    def cTestSet(self):
        return self.__cTestSet

    @property
    def loss(self):
        return self.__loss
    
    @property
    def net(self):
        return self.__net
        
    @property
    def kindTraining(self):
        return self.__kindTraining
        
    @property
    def batch_size(self):
        return self.__batch_size
        
    @property
    def lr(self):
        return self.__lr

    @property
    def weight(self):
        return self.__weight

    @property
    def tabSNR(self):
        return self.__tabSNR

    @property
    def final_acc(self):
        return self.__final_acc

    @property
    def final_acc_t(self):
        return self.__final_acc_t
        
    @property
    def final_loss(self):
        return self.__final_loss
        
    @property
    def final_loss_t(self):
        return self.__final_loss_t
        

    def printinfo(self):
        print("Printing info related to the network training")
        print("Training sample properties : ")
        print("--> Data segment length(s) :",self.__netinfo[0])
        print("--> Corresp. samplings     :",self.__netinfo[1])
        print("--> PSD type               :",self.__netinfo[2])
        print("--> Template type          :",self.__netinfo[3])
        print("--> Freq range             :",self.__netinfo[5],self.__netinfo[6])
        print("Learning rate              : ",self.__lr)
        print("Epochs                     : ",self.__epochs)
        print("SNRs                       : ",self.__SNRs)
        print("SNR in intervals or scalar value: ",self.__kindTraining)
       

class custom_loss_function:
    def __init__(self, d):
        self.d = d

    def __call__(self, y_true, y_pred):
        #batch_size=100
        #axis_to_reduce = tuple(range(1, K.ndim(y_pred)))  # All axis but first (batch)
        res = ave_ft(y_true, y_pred, self.d)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        #print(rms,res)
        return mse - res
    def get_config(self):
        return {'d': self.d}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def f_tanimoto_w(y_true,y_pred,d,ds):


    
    # Default is one
    class_weights = tf.ones_like(y_true[0], dtype=tf.float32)  # Create weights of 1 for each class
    if not isinstance(class_weights, tf.Tensor):
       class_weights = tf.constant(class_weights)
    
    if ds<0: # bypass for the moment 
        # When the network start to converge we put an emphasis on small values
        wli = tf.math.reciprocal(y_true ** 1)
        # ---------------------This line is taken from niftyNet package --------------
        # ref: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py, lines:170 -- 172
        # First turn inf elements to zero, then replace that with the maximum weight value
        new_weights = tf.where(tf.math.is_inf(wli), tf.zeros_like(wli), wli)
        class_weights = tf.where(tf.math.is_inf(wli), tf.ones_like(wli) * tf.reduce_max(new_weights), wli)
        # --------------------------------------------------------------------


    axis_to_reduce = range(1, K.ndim(y_pred))
    numerator = y_true * y_pred * class_weights
    numerator = K.sum(numerator, axis=axis_to_reduce)

    denominator1 = (y_true ** 2 + y_pred ** 2) * class_weights
    denominator1 = K.sum(denominator1, axis=axis_to_reduce) * (2**d)
    denominator2 = (y_true * y_pred) * class_weights
    denominator2 = K.sum(denominator2, axis=axis_to_reduce) *((2**(d+1))-1)
    denominator = denominator1 - denominator2
    return numerator / denominator


def ave_ft(x,y,d=0):
    dstart=d
    if d==0:
        d=1
    result=0.
    for i in range(d):
        result+=f_tanimoto_w(x,y,i,dstart)
    return result/d
