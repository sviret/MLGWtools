import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 1000
parameters = {'font.size': 25,'axes.labelsize': 25,'axes.titlesize': 25,'figure.titlesize': 30,'xtick.labelsize': 25,'ytick.labelsize': 25,'legend.fontsize': 25,'legend.title_fontsize': 25,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [6.4*3.5,4.8*3.5]}    
mpl.rcParams['lines.linewidth'] = 3
plt.rcParams['figure.figsize'] = (8.4,5.8)

# Slow version (loop over elements)
import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def usoftmax_f(X):
  diff=tnp.diff(X,1)
  X2=tnp.concatenate((1./(1.+tnp.exp(diff)),1./(1.+tnp.exp(-diff))),axis=1)
  return X2

def accuracy(yhat,N,seuil=0.5):
    return (sensitivity(yhat,N,seuil)+1-FAR(yhat,N,seuil))/2

def sensitivity(yhat,N,seuil=0.5):
    #superieur au seuil
    return ((yhat[::2].T[1].astype(np.float32)>=seuil)*np.ones(N//2)).mean()

def FAR(yhat,N,seuil=0.5):
    return 1-((yhat[1::2].T[0].astype(np.float32)<seuil)*np.ones(N//2)).mean()

def Threshold(yhat,N,FAR=0.005):
    l=np.sort(yhat[1::2].T[1]) # Proba d'être signal assignée au bruit
    ind=len(l)-int(np.floor(FAR*(N//2)))
    if ind==0:
        print('Sample is too small to define a threshold with FAP',FAR)
        ind=1
    seuil=l[ind-1]
    return seuil

'''
Class handling training results


'''


class Results:

    def __init__(self,SNRtest=10):
        self.__SNRtest=SNRtest # à généraliser à plusieurs SNRtests
        
        self.__listEpochs=[]
        self.__testOut=[]
        self.__listTrainLoss=[]
        self.__listTestLoss=[]
        self.__listTrainAcc=[]
        self.__listTestAcc=[]
        self.__lastOuts=[]
        self.__snrOuts=[]
        
        self.__cTrainer=None
        
    def setMyTrainer(self,mytrainer,mynet):
        self.__cTrainer=mytrainer
        self.__NsampleTrain=int(self.__cTrainer.trainGenerator.Nsample) # Limit size of the result file
        self.__NsampleTest=int(self.__cTrainer.testGenerator.Nsample)
        self.__kindTraining=self.__cTrainer.kindTraining
        self.__lr=self.__cTrainer.lr # Learning rate
        self.__minSNR=self.__cTrainer.tabSNR[-1] # Last SNR range
        self.__kindTemplate=self.__cTrainer.trainGenerator.kindTemplate
        self.__kindBank=self.__cTrainer.trainGenerator.kindBank
        self.__kindPSD=self.__cTrainer.trainGenerator.kindPSD
        self.__mInt=self.__cTrainer.testGenerator.mInt
        self.__step=self.__cTrainer.testGenerator.mStep
        self.__net=mynet

        # Use the complete dataset here
        self.__Xtest=self.__cTrainer.cTestSet[0]
    
    def setMyEncoder(self,mytrainer):
        self.__cTrainer=mytrainer
        self.__NsampleTrain=int(self.__cTrainer.trainGenerator.Nsample) # Limit size of the result file
        self.__NsampleTest=int(self.__cTrainer.testGenerator.Nsample)
        self.__tsize=self.__NsampleTrain
        self.__kindTraining=self.__cTrainer.kindTraining
        self.__batch_size=self.__cTrainer.batch_size
        self.__lr=self.__cTrainer.lr # Learning rate
        #self.__weight=self.__cTrainer.weight #
        self.__minSNR=self.__cTrainer.tabSNR[-1] # Last SNR range
        self.__kindTemplate=self.__cTrainer.trainGenerator.kindTemplate
        self.__kindBank=self.__cTrainer.trainGenerator.kindBank
        self.__kindPSD=self.__cTrainer.trainGenerator.kindPSD
        self.__mInt=self.__cTrainer.testGenerator.mInt
        self.__step=self.__cTrainer.testGenerator.mStep
            
        # Use the complete dataset here
        self.__Xtest=self.__cTrainer.cTestSet[0]
        self.__ytest=self.__cTrainer.cTestSet[1] 


    # Fill the results obtained at the end of one epoch
    def Fill(self):
        # (0,1,2,....) until the last epoch

        # Get the net output for validation sample only
        outTest = self.__net.model.predict(self.__Xtest,verbose=0)
        self.__testOut.append(usoftmax_f(outTest))
    
    def FillEncoder(self):
        # (0,1,2,....) until the last epoch
        
        # Get the net output for validation sample only
        outTest = self.__net.model.predict(self.__Xtest,verbose=0)
        self.__testOut.append(outTest)

    # Here we do the calculations for the ROC curve
    # this is called at the end
    def finishTraining(self):
    
        self.__listTrainLoss=self.__cTrainer.final_loss
        self.__listTestLoss=self.__cTrainer.final_loss_t
        self.__listTrainAcc=self.__cTrainer.final_acc
        self.__listTestAcc=self.__cTrainer.final_acc_t
        
        self.__listEpochs=np.arange(len(self.__listTrainLoss))
    
        self.__trainData=[self.__listTrainLoss,self.__listTestLoss,self.__listTrainAcc,self.__listTestAcc]

        for snr in range(0,20):
            rsnr=0.5*snr
            sample=self.__cTrainer.testGenerator.getDataSet([rsnr],size=1000)
            data=np.array(sample[0].reshape(len(sample[0]),-1,1),dtype=np.float32)
            labels=np.array(sample[3],dtype=np.int32)
            weight_sharing=np.array(sample[1],dtype=np.float32)
            TestSet=(data,labels,weight_sharing)
            
            cut_top = 0
            cut_bottom = 0
            list_inputs_val=[]
                    
            for j in range(self.__net.getNband()):
                cut_top += int(self.__net.getlchunk(j))
                list_inputs_val.append(data[:,cut_bottom:cut_top,:])
                cut_bottom = cut_top
            
            outTest = self.__net.model.predict(list_inputs_val,verbose=0)
            #self.__lastOuts.append(outTest)
            self.__lastOuts.append(usoftmax_f(outTest))
            self.__snrOuts.append(rsnr)
        del TestSet,list_inputs_val,data,sample,weight_sharing,labels
        self.__Xtest=None
        self.__cTrainer=None
        self.__net=None

     
       
    def accuracy(self,epoch,seuil=0.5):
        return accuracy(self.__testOut[epoch],self.__NsampleTrain,seuil), accuracy(self.__testOut[epoch],self.__NsampleTest,seuil)
            
    def sensitivity(self,epoch,seuil=0.5):
        return sensitivity(self.__testOut[epoch],self.__NsampleTrain,seuil), sensitivity(self.__testOut[epoch],self.__NsampleTest,seuil)
      
    def FAR(self,epoch,seuil=0.5):
        return FAR(self.__testOut[epoch],self.__NsampleTrain,seuil), FAR(self.__testOut[epoch],self.__NsampleTest,seuil)
            
    def Threshold(self,epoch,FAR=0.005):
        return Threshold(self.__testOut[epoch],self.__NsampleTrain,FAR), Threshold(self.__testOut[epoch],self.__NsampleTest,FAR)
            
    def saveResults(self):
        #if not(os.path.isdir(dossier)):
        #    raise FileNotFoundError("Le dossier de sauvegarde n'existe pas")
        fichier='train_result-'+self.__kindTraining+'-1.p'
        c=1
        while os.path.isfile(fichier):
            c+=1
            fichier='train_result-'+self.__kindTraining+'-'+str(c)+'.p'
            
        f=open(fichier, mode='wb')
        #pickle.dump(self,f)
        pickle.dump([self.__snrOuts,self.__lastOuts,self.__testOut,self.__mInt,self.mStep,self.__SNRtest,self.__trainData],f)
        f.close()
      
    @classmethod
    def readResults(cls,fichier):
        f=open(fichier, mode='rb')
        obj=pickle.load(f)
        f.close()
        return obj
      
    @property
    def SNRtest(self):
        return self.__SNRtest
    @property
    def testOut(self):
        return self.__testOut
    @property
    def lastOuts(self):
        return self.__lastOuts
    @property
    def snrOuts(self):
        return self.__snrOuts
    @property
    def NsampleTest(self):
        return self.__NsampleTest
    @property
    def mInt(self):
        return self.__mInt
    @property
    def mStep(self):
        return self.__step
    @property
    def listEpochs(self):
        return self.__listEpochs
    @property
    def TrainLoss(self):
        return self.__listTrainLoss
    @property
    def TestLoss(self):
        return self.__listTestLoss
    @property
    def TrainAcc(self):
        return self.__listTrainAcc
    @property
    def TestAcc(self):
        return self.__listTestAcc
        
    @property
    def lr(self):
        return self.__lr
    @property
    def kindTraining(self):
        return self.__kindTraining
    @property
    def kindPSD(self):
        return self.__kindPSD
    @property
    def minSNR(self):
        return self.__minSNR
    @property
    def kindTemplate(self):
        return self.__kindTemplate
    @property
    def kindBank(self):
        return self.__kindBank

'''
Class for plot printing based on a result file
'''

class Printer:
    def __init__(self):
        self.__nbDist=0
        self.__nbMapDist=0
        self.__nbROC=0
        self.__nbSens=0

    def plotResults(self,results,FAR=0.005):

        nrecorded=len(results[2])
        for i in range(nrecorded):
            self.plotDistrib(results,i,FAR=FAR)
            self.plotMapDistrib(results,i)
        self.plotROC(results,FAR=FAR)
        self.plotSensitivity(results,FAR=FAR)
        self.plotMultiSensitivity(results)



    def plotDistrib(self,result,epoch,FAR=0.005):
        self.__nbDist+=1
        data=result[2][epoch]
        SNRtest=result[5]

        distsig=data[::2].T[1].numpy()
        distnoise=data[1::2].T[1].numpy()
        seuil=Threshold(data,len(data),FAR)

        plt.figure('Distribution_epoch'+str(epoch)+'-'+str(self.__nbDist))
        plt.axvline(x=seuil,color='r',label='FAR='+str(FAR),linestyle='--')
        plt.hist(distnoise,bins=np.arange(101)/100,label='noise distrib')
        plt.hist(distsig,bins=np.arange(101)/100,label='sig distrib')
        plt.text(seuil-0.1,100,'seuil='+str(np.around(seuil,3)))
        plt.xlabel('Probability')
        plt.ylabel('Number')
        plt.yscale('log')
        plt.title(label='Sortie du reseau sur le testSet à SNR='+ str(SNRtest)+ ' à l\'epoque '+str(epoch))
        plt.legend()
            
    def plotMapDistrib(self,result,epoch):
        self.__nbMapDist+=1
        data=result[2][epoch]
        distsig=data[::2].T[1].numpy()
        mlow=int(np.floor(result[3][0]))
        mhigh=int(np.ceil(result[3][1]))
        mstep=result[4]
       
        Nbmasses=int((mhigh-mlow)/mstep)

        X, Y = np.meshgrid(np.linspace(mlow, mhigh, Nbmasses+1), np.linspace(mlow, mhigh, Nbmasses+1))
        Z=np.zeros((Nbmasses,Nbmasses))
        c=0
        for i in range(Nbmasses):
            if c==len(distsig):
                break
            for j in range(i+1):
                #print(i,j,c)
                Z[i][j]=distsig[c]
                c+=1
                if c==len(distsig):
                    break
        
        plt.figure('MapDistribution_epoch'+str(epoch)+'-'+str(self.__nbMapDist))
        plt.pcolormesh(X,Y,Z.T)
        plt.xlabel('m1')
        plt.ylabel('m2')

        plt.colorbar()
        plt.title(label='Output of softmax regression for signal sample in the plan (m1,m2)')
    


    def plotROC(self,results,FAR=0.005):
        self.__nbROC+=1
        plt.figure('ROC-'+str(self.__nbROC))
            
        trainData=results[6]

        TrainLoss=trainData[0]
        TestLoss=trainData[1]        
        TrainAcc=trainData[2]
        TestAcc=trainData[3] 
        listEpochs=np.arange(len(TrainLoss))
  
        plt.plot(listEpochs,TrainLoss,label='Training Loss')
        plt.plot(listEpochs,TestLoss,linestyle='--',label='Test Loss')
        plt.plot(listEpochs,TrainAcc,label='Training Sensitivity')
        plt.plot(listEpochs,TestAcc,linestyle='--',label='Test Sensitivity')
            
        plt.xlabel('Epochs')
        plt.legend()


    def plotSensitivity(self,results,FAR=0.005):
        self.__nbSens+=1
        plt.figure('Sensitivity_Vs_SNRtest-'+str(self.__nbSens))
        
        SNRlist=results[0]
        
        Sensitivitylist=[]
        for yhat in results[1]:
            N=len(yhat)
            seuil=Threshold(yhat,N,FAR)
            Sensitivitylist.append(100*sensitivity(yhat,N,seuil))                        

        plt.plot(SNRlist,Sensitivitylist,'.-',label='Sensitivity')
        plt.xlabel('SNROpt')
        plt.ylabel('%')
        plt.grid(True, which="both", ls="-")
        plt.title(label='Sensitivity Vs SNRopt pour un FAR='+str(FAR))
        plt.legend()
        
    def plotMultiSensitivity(self,results):
        self.__nbSens+=1
        plt.figure('Sensitivity_Vs_SNRtest-'+str(self.__nbSens))
               
        SNRlist=results[0]
        
        for i in range(4):
            Sensitivitylist=[]
            
            for yhat in results[1]:
                FAR=10**(-float(i+1))
                N=len(yhat)
                seuil=Threshold(yhat,N,FAR)
                Sensitivitylist.append(100*sensitivity(yhat,N,seuil))   
           
            plt.plot(SNRlist,Sensitivitylist,'.-',label='FAP='+str(FAR))
        plt.xlabel('SNROpt')
        plt.ylabel('%')
        plt.grid(True, which="both", ls="-")
        plt.legend()
        
    def savePrints(self,name):
            
        c_dossier=name+'-1'
        c=1
        while os.path.isdir(c_dossier):
            c+=1
            c_dossier=name+'-'+str(c)
        dossier=c_dossier
        os.mkdir(dossier)
        fichier=dossier+'/all_results.pdf'
            
        pp = PdfPages(fichier)
        
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf')
            fig.savefig(dossier+'/'+fig.get_label()+'.png',format='png')
        pp.close()
