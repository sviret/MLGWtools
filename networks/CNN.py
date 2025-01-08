import tensorflow as tf
from tensorflow.keras import layers, models, initializers, optimizers
import numpy as np
import os
import pickle

'''
Class defining the network
'''

class Multiple_CNN():

    '''
    Initialization: here we define the CNN neural net

    Basically we set up one net for each frequency band. Initialisation 
    requires a training config object (see utils.py)

    Once the general hyperparameters are defined, different type of networks can be instantiated     
   
    

    
    '''
    def __init__(self,training):

        self.__netinfo=training.netprop()
        self.__lr=training.lr()
        self.__kindTraining=training.kt()

        self.__initializer = initializers.GlorotNormal() # Xavier initialization
        self.__opt = optimizers.Adam(learning_rate=self.__lr)
        self.__loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

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

    #
    # Network definition scripts
    #

    '''
    This net is the standard CNN as defined in the seminal
    paper from Huerta and George. 3 CNN layers followed by a dense one
    
    https://arxiv.org/abs/1711.03121
    '''
    
    def huerta_legacy(self):

        self.inputs=[]
        self.outputs=[]

        print("Initialize the legacy Huerta-George CNN network")

        with tf.name_scope('simpleCNN') as scope:
            for i in range(self.__nbands):
                input=layers.Input(shape=(int(self.list_chunks[i]),1))
                x=layers.BatchNormalization()(input)
                x=layers.Conv1D(filters=16, kernel_size=16, kernel_initializer=self.__initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=32, kernel_size=8, kernel_initializer=self.__initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=64, kernel_size=8, kernel_initializer=self.__initializer)(x)
                
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Flatten()(x)

                output=layers.Dense(2, kernel_initializer=self.__initializer)(x)
                
                self.inputs.append(input)

                # Don't use the weighting for the moment
                #self.outputs.append(weight[i]*output)
                self.outputs.append(output)

        # The last layer is a dense one which takes as input a weighted average of all the networks outputs
        # The weighting from each layer is defined later.

        # Take note that there is no activation in the last dense layer
        # softmax activation is included in the loss function (via from_logits option)
        # enable to use the special activation function defined in https://arxiv.org/abs/2106.03741
        #

        x = layers.add(self.outputs)
        if not self.singleBand: # merge the bands if > 1
            self.out = layers.Dense(2, kernel_initializer=self.__initializer)(x)
        else:
            self.out = self.outputs
        self.model = models.Model(self.inputs, self.out)
        
        # Init the model
        self.model.compile(optimizer=self.__opt,loss=self.__loss,metrics=['accuracy'])
                
        # Print the network
        print(self.model.summary())

    '''
    This net is a version optimised for the hardware implementation
    '''

    def fpga_version(self):


        self.inputs=[]
        self.outputs=[]

        print("FPGA friendly CNN version")

        with tf.name_scope('simpleCNN') as scope:
            for i in range(self.__nbands):
                input=layers.Input(shape=(int(self.list_chunks[i]),1))

                x=input
  
                x=layers.Conv1D(filters=4, kernel_size=8, kernel_initializer=self.__initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=8, kernel_size=2, kernel_initializer=self.__initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=16, kernel_size=2, kernel_initializer=self.__initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=8, kernel_size=8, kernel_initializer=self.__initializer)(x)
                
                # Other versions
                '''
                x=layers.Conv1D(filters=10, kernel_size=8, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=12, kernel_size=2, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=8, kernel_size=2, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=8, kernel_size=8, kernel_initializer=initializer)(x)
                '''
                '''
                x=layers.Conv1D(filters=16, kernel_size=8, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=16, kernel_size=2, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=16, kernel_size=2, kernel_initializer=initializer)(x)
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Conv1D(filters=8, kernel_size=8, kernel_initializer=initializer)(x)
                '''
                x=layers.MaxPool1D(pool_size=4)(x)
                x=layers.Activation(activation='relu')(x)
                x=layers.Flatten()(x)
                #x=layers.Dense(16, activation='relu', kernel_initializer=initializer)(x)
                #x=layers.Dense(8, activation='relu', kernel_initializer=initializer)(x)
                output=layers.Dense(2, kernel_initializer=self.__initializer)(x)
                
                self.inputs.append(input)

                # Don't use the weighting for the moment
                #self.outputs.append(weight[i]*output)
                self.outputs.append(output)

        # The last layer is a dense one which takes as input a weighted average of all the networks outputs
        # The weighting from each layer is defined later.

        # Take note that there is no activation in the last dense layer
        # softmax activation is included in the loss function (via from_logits option)
        # enable to use the special activation function defined in https://arxiv.org/abs/2106.03741
        #

        x = layers.add(self.outputs)
        if not self.singleBand: # merge the bands if > 1
            self.out = layers.Dense(2, kernel_initializer=self.__initializer)(x)
        else:
            self.out = self.outputs
        self.model = models.Model(self.inputs, self.out)
        
        # Init the model
        self.model.compile(optimizer=self.__opt,loss=self.__loss,metrics=['accuracy'])
                
        # Print the network
        print(self.model.summary())

    '''
    Other methods
    '''


    def getNetSize(self):
        return self.__npts
        
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
        fichier=self.__kindTraining+'-'+str(self.__lr)+'-net-1.h5'
        fichier_js=self.__kindTraining+'-'+str(self.__lr)+'-net-1_fullnet.p'
        c=1
        while os.path.isfile(fichier):
            c+=1
            fichier=self.__kindTraining+'-'+str(self.__lr)+'-net-'+str(c)+'.h5'
            fichier_js=self.__kindTraining+'-'+str(self.__lr)+'-net-'+str(c)+'_fullnet.p'
        self.model.save(fichier)
        f=open(fichier_js, mode='wb')
        pickle.dump(self,f)
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
        
