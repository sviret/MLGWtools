import numpy as np
import os
import pickle
import csv
from MLGWtools.generators import generator as gd

'''
*** trutils ***

Class handling the network interface with data and the training 

'''

class trutils():


    '''
    Initialization
    '''
    def __init__(self,paramFile):
    
        if os.path.isfile(paramFile):
            self._readParamFile(paramFile)
        else:
            raise FileNotFoundError("Le fichier de paramètres n'existe pas")

        # Getting the datasets 
        self.__trainGenerator=gd.Generator.readGenerator(self.__tr_s)
        self.__testGenerator=gd.Generator.readGenerator(self.__te_s)

        if not self._compare_files(self.__trainGenerator,self.__testGenerator):
            raise Exception("Training and test samples have different configs: abort!!!")
  
        print("Train and test samples are compatible and loaded, go on...")


    '''
    _readparamfile: parse the csv param file
    '''
    
    def _readParamFile(self,paramFile):

        fdir=paramFile.split('/')
        self.__fname=fdir[len(fdir)-1].split('.')[0]
       
        self.__batch_size=100 # taille des batchs
        self.__lr=0.0002      # learning rate
        self.__tabEpochs=[]   # tableau des époques
        self.__kindTraining='DecrInt'
        self.__tabSNR=[]

        with open(paramFile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]

        # List of available commands
        cmdlist=['runType','batch_size','lr','kindTraining','tabEpochs',
                 'tabSNR','verbose','trainSample','testSample']

        for line in lignes:
            if len(line)==0:
                continue
            if line[0]=='\n':
                continue
            if '#' in line[0]: # Comment, skip...
                continue
            cmd=line[0]
            if cmd not in cmdlist:              
                raise Exception(f"Keyword {cmd} unknown: abort")

            if cmd=='runType': 
                self.__rType=line[1] 

            if cmd=='batch_size': 
                self.__batch_size=int(line[1]) 

            if cmd=='lr': 
                self.__lr=float(line[1]) 

            if cmd=='kindTraining': 
                self.__kindTraining=line[1] 

            if cmd=='tabSNR':
                self.__tabSNR=[]  
                for x in range(1,len(line)):
                    self.__tabSNR.append(float(line[x]))

            if cmd=='tabEpochs':
                self.__tabEpochs=[]  
                self.__tabEpochs.append(0)
                for x in range(1,len(line)):
                    self.__tabEpochs.append(int(line[x]))

            if cmd=='verbose':         
                self.__verb=True

            if cmd=='trainSample':         
                self.__tr_s=line[1]

            if cmd=='testSample':         
                self.__te_s=line[1]                


    '''
    _compare_files: check that train and test samples have the same shape
    '''
    
    def _compare_files(self,train,test):

        self._prop_tr=train.sampleprop()
        self._prop_te=test.sampleprop()

        if  self._prop_tr!=self._prop_te:
            return False

        return True
     
       
    '''
    _clear: reset everything
    '''
    
    def _clear(self):

        self.__cTrainSet=None
        self.__cTestSet=None

        
    '''
    train: this is the main training macro for a CNN-like network
    '''
    
    def train(self,net,SNRtest=7.5,results=None,verbose=True):
        self._clear()
        
        # If snr to be used is defined by SNR range, reshape tabSNR accordingly
        if 'Int' in self.__kindTraining:
            tabSNR2=[]
            for i in range(0,len(self.__tabSNR),2):
                tabSNR2.append([self.__tabSNR[i],self.__tabSNR[i+1]])
            self.__tabSNR=tabSNR2
        else:
            tabSNR2=[]
            for i in range(0,len(self.__tabSNR)):
                tabSNR2.append([self.__tabSNR[i]])
            self.__tabSNR=tabSNR2
        #print(self.__tabSNR)         

        self.__net=net
        # First we pick data in the training sample and adapt it to the required starting SNR
  
        sample=self.__trainGenerator.getDataSet(self.__tabSNR[0])
        # Training data at the initial SNR
        data=np.array(sample[0].reshape(len(sample[0]),-1,1),dtype=np.float32)
        # Expected outputs
        labels=np.array(sample[3],dtype=np.int32)
        # Sharing among frequency bands
        weight_sharing=np.array(sample[1],dtype=np.float32)

        # The test dataset will always be the same, pick it up once
        sample_t=self.__testGenerator.getDataSet([SNRtest])
        data_t=np.array(sample_t[0].reshape(len(sample_t[0]),-1,1),dtype=np.float32)
        labels_t=np.array(sample_t[3],dtype=np.int32)
        weight_sharing_t=np.array(sample_t[1],dtype=np.float32)
        
        if verbose:
            print("Shape of training set",data.shape)
            print("Shape of validation set",data_t.shape)
                
        cut_top = 0
        cut_bottom = 0
        list_inputs_val=[]
        list_inputs=[]
        
        for i in range(net.getNband()):
            cut_top += int(net.getlchunk(i))
            list_inputs_val.append(data_t[:,cut_bottom:cut_top,:])
            list_inputs.append(data[:,cut_bottom:cut_top,:])
            cut_bottom = cut_top
                              
        self.__cTrainSet=(list_inputs,labels,weight_sharing)
        self.__cTestSet=(list_inputs_val,labels_t,weight_sharing_t)

        # Put the init training properties in the results output file
        results.setMyTrainer(self,net)
        results.Fill()
        epoch=0
        
        # Loop over all the requested SNR, each of them corresponds to a certain
        # number of epochs
        
        accuracy=[]
        loss=[]
        accuracy_t=[]
        loss_t=[]
        
        for i in range(len(self.__tabSNR)):
        
            if i>0: # We start a new SNR range, need to update the training set
                del self.__cTrainSet

                # Create a dataset with the corresponding SNR
                # Starting from the initial one at SNR=1
                sample=self.__trainGenerator.getDataSet(self.__tabSNR[i])
                data=np.array(sample[0].reshape(len(sample[0]),-1,1),dtype=np.float32)
                weight_sharing=np.array(sample[1],dtype=np.float32)
                labels=np.array(sample[3],dtype=np.int32)

                cut_top = 0
                cut_bottom = 0
                list_inputs=[]
                    
                for j in range(net.getNband()):
                    cut_top += int(net.getlchunk(j))
                    list_inputs.append(data[:,cut_bottom:cut_top,:])
                    cut_bottom = cut_top
                
                self.__cTrainSet=(list_inputs,labels,weight_sharing)
            #print(self.__tabSNR[i])
                 
            # Then run for the corresponding epochs at this SNR range/value
            nepochs=self.__tabEpochs[i+1]-self.__tabEpochs[i]

            print("Training between epochs",self.__tabEpochs[i],"and",self.__tabEpochs[i+1])
            
            # Run the training over the epochs
            history=net.model.fit(self.__cTrainSet[0],labels,batch_size=self.__batch_size,epochs=nepochs, validation_data=(list_inputs_val, labels_t))
            
            acc=np.asarray(history.history['accuracy'])
            los=np.asarray(history.history['loss'])
            acc_t=np.asarray(history.history['val_accuracy'])
            los_t=np.asarray(history.history['val_loss'])
            
            for k in range(nepochs):
                accuracy.append(acc[k])
                loss.append(los[k])
                accuracy_t.append(acc_t[k])
                loss_t.append(los_t[k])

            train_acc=np.asarray(history.history['accuracy']).mean()
            test_acc=np.asarray(history.history['val_accuracy']).mean()
            train_l=np.asarray(history.history['loss']).mean()
            test_l=np.asarray(history.history['val_loss']).mean()

            epoch+=nepochs
            if verbose:
                print(f'Train perf with this SNR range: train loss {train_l:.3f}, train acc {train_acc:.3f}')
                print(f'Validation perf at this stage: test loss {test_l:.3f}, test acc {test_acc:.3f}')

            results.Fill()

        self.__final_acc=np.asarray(accuracy).flatten()
        self.__final_acc_t=np.asarray(accuracy_t).flatten()
        self.__final_loss=np.asarray(loss).flatten()
        self.__final_loss_t=np.asarray(loss_t).flatten()

        results.finishTraining()
    

    def type(self):
        return self.__rType
    
    def netprop(self):
        return self._prop_tr 

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
    def kindTraining(self):
        return self.__kindTraining
     
    def batch_size(self):
        return self.__batch_size
        
    def lr(self):
        return self.__lr
    
    def kt(self):
        return self.__kindTraining
    
    @property
    def weight(self):
        return self.__weight

    @property
    def tabSNR(self):
        return self.__tabSNR

    def listSNR(self):
        return self.__tabSNR
    
    def listEpochs(self):
        return self.__tabEpochs

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
    

        

