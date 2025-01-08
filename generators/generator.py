import numpy as npy
import matplotlib.pyplot as plt
import pickle
import csv
import os
import time
import MLGWtools.generators.noises as gn
import MLGWtools.generators.signals as gt
#from gwpy.timeseries import TimeSeries
from numcompress import compress, decompress

#constantes physiques
G=6.674184e-11
Msol=1.988e30
c=299792458
MPC=3.086e22


######################################################################################################################################################################################
parameters = {'font.size': 15,'axes.labelsize': 15,'axes.titlesize': 15,'figure.titlesize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15,'legend.fontsize': 15,'legend.title_fontsize': 15,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [10,5]}
plt.rcParams.update(parameters)
  
#################################################################################################################################################################################################
'''
Class handling gravidational wave dataset production (either for training or testing)

Options:

--> mint     : Solar masses range (mmin,mmax) for the (m1,m2) mass grid to be produced
--> step     : Mass step between two samples
--> NbB      : Number of background realisations per template
--> tcint    : Range for the coalescence time within the last signal chunk (in sec, should be <Ttot)
               if the last chunk has the length T, the coalescence will be randomly placed within the range
               [tcint[0]*T,tcint[1]*T]
--> choice   : The template type (EOB or EM)
--> kindBank : The type of mass grid used (linear or optimal)
--> whitening: Frequency-domain or time-domain whitening (1 or 2) 
--> ninj     : Number of injections you want to produce (frame mode)
--> txtfile  : Text file containing a list of injection (typically a bank file)
'''


class Generator:

    def __init__(self,paramFile):
    
        if os.path.isfile(paramFile):
            self._readParamFile(paramFile)
        else:
            raise FileNotFoundError("Param file required")

    
    # Produce a data frame with injections

    def buildFrame(self):
        
        if self.__rType!='frame':
            raise Exception("Your job type is not frame, you are not supposed to use this function")
            
        start_time = time.time()
        print("Starting sim data frame generation")
    
        # Instantiate the objects

        self.initNoise()
        self.initTemplate()

        # First produces the noise sequence with the correct duration

        nitf = 1  # How many detectors ? 

        self.__Noise=[]
        self.__Signal=[]
        self.__listSNR2chunks=[[] for x in range(self.__nTtot)]
        self.__listfnames=[]
        for i in range(nitf):
            self._GenNoiseSequence(self.__length) 

        # For each detector we now have a set of 4 data stream of the total length
        # at sampling freq: 
        #
        # Noise and raw noise 
        # Signal and raw signal (empty vectors for the moment)

        # Now adds the injections
        # At fixed time intervals

        interval=self.__length/(self.__ninj+1)

        print(self.__ninj,"signals will be injected in the data stream")
        self.__injections=[]
            
        for i in range(self.__ninj):

            if len(self.__injparams)!=0 :
                ligne_aleatoire = npy.random.choice(self.__injparams).strip().split(',')
                coord=np.array(ligne_aleatoire,dtype=float)
                # !id,m1,m2,spin1x,spin1y,spin1z,spin2x,spin2y,spin2z,chieff,chip,tcoal,SNRH,SNRL
                id = int(coord[0])
                m1 = coord[1]
                m2 = coord[2]

                self.__signal.majParams(coord[1],coord[2],s1x=coord[3],s2x=coord[6],
                                        s1y=coord[4],s2y=coord[7],s1z=coord[5],s2z=coord[8])
            else :
                id = -1
                m1=npy.random.uniform(5,75)
                m2=npy.random.uniform(5,75)
                self.__signal.majParams(m1,m2)

            SNR=npy.random.uniform(5,40)

            self.__signal.getNewSample(Tsample=self.__signal.duration())
            data=self.__signal.signal()        # Whitened and normalised to SNR**2=1
            data_r=self.__signal.signal_raw()/self.__signal.norma()  # Raw and normalised

            randt=npy.random.normal((i+1)*interval,interval/5.)
            inj=[id,m1,m2,SNR,randt]
            self.__injections.append(inj)
            print("Injection",i,"(id,m1,m2,SNR,tc)=(",f'{id}',f'{m1:.1f}',f'{m2:.1f}',f'{SNR:.1f}',f'{randt:.1f}',")")

            # Where to add the first signal in the frame
            idxstart=int((randt-self.__signal.duration())*self.__listfe[0])

            if idxstart<0:
                idxstart=0
    
            # Add injection without antenna pattern
            for j in range(len(data)):
                self.__Signal[0][0][idxstart+j]+=SNR*data[j]
                self.__Signal[0][1][idxstart+j]+=SNR*data_r[j]
    
            data=[]
            data_r=[]

        # Injection process over, self.__Signal[0] now contains all the injections
        # Add them to noise to get the complete strain
                
        self.__Noise[0][0] += self.__Signal[0][0] # Whitened strain
        self.__Noise[0][1] += self.__Signal[0][1] # Raw strain
            
        npts=len(self.__Noise[0][0])
        norm=self.__length/npts
        
        plt.figure(figsize=(10,5))
        plt.xlabel('t (s)')
        plt.ylabel('h(t) (whithened)')
        plt.grid(True, which="both", ls="-")
        plt.plot(npy.arange(npts)*norm, self.__Noise[0][0])
        plt.plot(npy.arange(npts)*norm, self.__Signal[0][0])
        plt.show()

        plt.figure(figsize=(10,5))
        plt.xlabel('t (s)')
        plt.ylabel('h(t) (raw)')
        plt.grid(True, which="both", ls="-")
        plt.plot(npy.arange(npts)*norm, self.__Noise[0][1])
        plt.plot(npy.arange(npts)*norm, self.__Signal[0][1])
        plt.show()

    
    # Produce a training/test sample

    def buildSample(self):
        
        if self.__rType!='bank':
            raise Exception("Your job type is not bank, you are not supposed to use this function")
            
        start_time = time.time()
        print("Starting dataset generation")
            
        self.initNoise()
        self.initTemplate()

        self.__listSNR2evol=[]
        self.__listfnames=[]
        self.__listSNR2chunks=[[] for x in range(self.__nTtot)]
        self.__tmplist=[]

        print("1 After init --- %s seconds ---" % (time.time() - start_time))
            
        self._genGrille()   # Binary objects mass matrix
        
        print("2 After grille --- %s seconds ---" % (time.time() - start_time))
        
        self._genSigSet()   # The signals
        
        print("3 After signal --- %s seconds ---" % (time.time() - start_time))
        
        self._genNoiseSet() # The noises (one realization per template)
        
        print("4 After noise --- %s seconds ---" % (time.time() - start_time))
        
        for j in range(self.__nTtot):
            print("Chunk ",j," shape")
            print("Signal set shape:",self.__Sig[j].shape)
            print("Noise set shape:",self.__Noise[j].shape)
        
        self.__Labels=npy.concatenate((npy.ones(self.__Ntemplate*self.__NbB,dtype=int),npy.zeros(self.__Ntemplate*self.__NbB,dtype=int))) # 1 <-> signal , 0 <-> noise

        # At the end we plot a map giving the SNR**2 sharing of the different bands
        self.plotSNRmap()


    def initNoise(self):
        self.__noise = gn.Noises(Ttot=self.__listTtot,fe=self.__listfe,kindPSD=self.__kindPSD,
                                 fmin=self.__fmin,fmax=self.__fmax,whitening=self.__white,
                                 verbose=self.__verb,customPSD=self.__custPSD)
    
    def getNoise(self):
        return self.__noise

    def initTemplate(self):
        self.__signal = gt.GenTemplate(Ttot=self.__listTtot,fe=self.__listfe,kindPSD=self.__kindPSD,
                                 kindTemplate=self.__kindTemplate,
                                 fmin=self.__fmin,fmax=self.__fmax,whitening=self.__white,
                                 verbose=self.__verb,customPSD=self.__custPSD)

    def getTemplate(self):
        return self.__signal
 


    '''
    DATASET 1/
    
    Parser of the parameters csv file
    '''

    def _readParamFile(self,paramFile):


        fdir=paramFile.split('/')
        self.__fname=fdir[len(fdir)-1].split('.')[0]
       

        with open(paramFile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]

        self.__rType='none'
        self.__nTtot=-1  
        self.__Ttot=-1
        self.__fe=-1
        self.__listTtot=[]
        self.__listfe=[]
        self.__kindPSD='flat'
        self.__kindTemplate='EM'
        self.__kindBank='linear'
        self.__custPSD=[]
        self.__fmin=20.
        self.__fmax=1024.
        self.__white=1
        self.__verb=False
        self.__prop=[1.4,1.4,0,0,0,0,0,0,8.]
        self.__NbB=1
        self.__step=1.
        self.__mint=[10.,30.]
        self.__tcint=[10.,30.]        
        self.__ninj=10
        self.__length=1000.
        self.__injparams=[]

        # List of available commands
        cmdlist=['runType','Ttot','fe','kindPSD','properties',
                 'mint','tcint','NbB','kindTemplate','kindBank','step',
                 'whitening','flims','verbose','length','ninj','injlist']
        
        for line in lignes:
            cmd=line[0]
            if cmd not in cmdlist:              
                raise Exception(f"Keyword {cmd} unknown: abort")

            if cmd=='runType': 
                self.__rType=line[1] 

            if cmd=='Ttot':    
                nband=len(line)-1
                if self.__nTtot!=-1 and self.__nTtot!=nband:
                    raise Exception("Number of bands bet. Ttot and fe disagree: abort")
                self.__nTtot=nband
                if self.__nTtot > 1:  # Multi band
                    for x in range(0,nband):
                        self.__listTtot.append(float(line[x+1]))
                    self.__Ttot=sum(self.__listTtot)
                else: # Single band
                    self.__Ttot=float(line[1])
                    self.__listTtot.append(self.__Ttot)
            
            if cmd=='fe':    
                nband=len(line)-1
                if self.__nTtot!=-1 and self.__nTtot!=nband:
                    raise Exception("Number of bands bet. Ttot and fe disagree: abort")
                self.__nTtot=nband
                if self.__nTtot > 1:  # Multi band
                    for x in range(0,nband):
                        self.__listfe.append(float(line[x+1]))
                    self.__fe=max(self.__listfe)
                else: # Single band
                    self.__fe=float(line[1])
                    self.__listfe.append(self.__fe)

            if cmd=='injlist':
                self.__injlist=line[1]
                if os.path.isfile(self.__injlist):
                    self.__injparams = self._readtxtFile(self.__injlist)
                    print("Retrieving injections parameters from file",self.__injlist)
                else:
                    print("No injection file found, will inject random templates")
                

            if cmd=='kindPSD':         
                self.__kindPSD=line[1]
                if line[1]!='flat' and line[1]!='analytic' and line[1]!='realistic':
                    raise ValueError("Accepted val. for kindPSD are only 'flat','analytic', et 'realistic'")
                if (self.__kindPSD=='realistic'):
                    self.__PSDfile=line[2]

                    psd=[]
                    freq=[]
                    print('Noise info will be based on the following PSD profile:',self.__PSDfile)

                    with open(self.__PSDfile,"r") as f:
                        for dat in f:
                            value=[float(v) for v in dat.split(' ')]
                            psd.append(value[1]**2)
                            freq.append(value[0])
                    f.close()
                    self.__custPSD.append([psd,freq])  

            if cmd=='verbose':         
                self.__verb=True

            if cmd=='kindTemplate':         
                self.__kindTemplate=line[1]

            if cmd=='properties':
                for i in range(6):          
                    self.__prop[i]=float(line[i+1])
                
            if cmd=='flims':
                self.__fmin=float(line[1])
                self.__fmax=float(line[2])

            if cmd=='mint':
                self.__mint[0]=float(line[1])
                self.__mint[1]=float(line[2])

            if cmd=='NbB':
                self.__NbB=int(line[1])

            if cmd=='step':
                self.__step=float(line[1])

            if cmd=='length':
                self.__length=float(line[1])

            if cmd=='ninj':
                self.__ninj=int(line[1])

            if cmd=='tcint':
                self.__tcint[0]=float(line[1])
                self.__tcint[1]=float(line[2])



    def _readtxtFile(self,txtfile) :
        with open(txtfile, 'r') as f:
            lignes = f.readlines()

        lignes_parametres = [ligne for ligne in lignes if not ligne.startswith('#') and not ligne.startswith('!')]
        return lignes_parametres

    '''
    DATASET 2/
    
    Produce a data grid with the mass coordinates
    '''

    def _genGrille(self):

        if self.__kindBank=='linear':
            n = self.__step
            N = len(npy.arange(self.__mint[0], self.__mint[1], n))
            self.__Ntemplate=int((N*(N+1))/2)
            self.__GrilleMasses=npy.ones((self.__Ntemplate,2))
            self.__GrilleSpins=npy.zeros((self.__Ntemplate,2))
            self.__GrilleMasses.T[0:2]=self.__mint[0]
            c=0
            
            # Fill the grid (half-grid in fact)
            # Each new line starts at the diagonal, then reach the end
            # mmin,mmax
            # mmin+step,mmax
            #...
            
            for i in range(N):
                for j in range(i,N):
                
                    self.__GrilleMasses[c][0]=self.__mint[0]+i*self.__step
                    self.__GrilleMasses[c][1]=self.__mint[0]+j*self.__step
                    c+=1
                    
        else: # The optimized bank (read from a file)
            Mtmp=[]
            Sztmp=[]
            start=self.__length
            ntmps=self.__ninj
            compt=0
            print("Templates are taken into file ",self.__kindBank)
            with open(os.path.dirname(__file__)+'/params/'+self.__kindBank) as mon_fichier:
                lines=mon_fichier.readlines()
                for line in lines:
                    if '#' in line:
                        compt=0
                        continue
                    if '!' in line:
                        compt=0
                        continue
                    if compt>=start+ntmps:
                        break
                    if compt<start:
                        compt+=1
                        continue

                    data=line.strip()
                    pars=data.split(' ')
                    # Cuts on total mass
                    if (float(pars[1])+float(pars[2])<self.__mint[0] and float(pars[1])+float(pars[2])>self.__mint[0]):
                        self.__tmplist.append([float(pars[0]),float(pars[1]),float(pars[2]),0])
                        compt+=1
                        continue
                    compt+=1
                    Mtmp.append([float(pars[1]),float(pars[2])])
                    Sztmp.append([float(pars[3]),float(pars[4])])
                    self.__tmplist.append([float(pars[0]),float(pars[1]),float(pars[2]),1])
            M=npy.asarray(Mtmp)
            Spins=npy.asarray(Sztmp)
            self.__GrilleMasses=M
            self.__GrilleSpins=Spins
            self.__Ntemplate=len(self.__GrilleMasses)
        
    '''
    DATASET 3/
    
    Produce the templates
    '''

    def _genSigSet(self):

        self.__Sig=[]
        c=0
        
        # First we produce the object with the correct size
        # The size is 2*Ntemplate*NbB but we put template only in the first half
        # The rest is filled with 0, it's important for GetDataSet
        
        for j in range(self.__nTtot): # Loop over samples
            dim=int(self.__listTtot[j]*self.__listfe[j])
            self.__Sig.append(npy.zeros((self.__Ntemplate*2*self.__NbB,dim)))
            
            # The following lines are for the SNR**2 repartition (renormalized)
            self.__listSNR2chunks[j].append(npy.full((self.__Ntemplate*2*self.__NbB),(1.0/self.__nTtot)))

        self.__listSNR2chunks=npy.reshape(self.__listSNR2chunks, (self.__nTtot, self.__Ntemplate*2*self.__NbB))

        # Now fill the object
        for i in range(0,self.__Ntemplate):
            if c%100==0:
                print("Producing sample ",c,"over",self.__Ntemplate*self.__NbB)
            self.__signal.majParams(m1=self.__GrilleMasses[i][0],m2=self.__GrilleMasses[i][1],s1z=self.__GrilleSpins[i][0],s2z=self.__GrilleSpins[i][1])
            
            temp,_=self.__signal.getNewSample(tc=npy.random.uniform(self.__tcint[0],self.__tcint[1]))
            self.__listSNR2evol=npy.append(self.__listSNR2evol,self.__signal._rawSnr)

            # Fill the corresponding data
            for j in range(self.__nTtot):
                self.__Sig[j][c]=temp[j]
                self.__listSNR2chunks[j][c]=self.__signal._currentSnr[j]
            c+=1
            
            # Fill the NbB-1 additional samples (with a different tc)
            for k in range(1,self.__NbB):
                temp,_=self.__signal.getSameSample(tc=npy.random.uniform(self.__tcint[0],self.__tcint[1]))
                for j in range(self.__nTtot):
                    self.__Sig[j][c]=temp[j]
                    self.__listSNR2chunks[j][c]=self.__signal._currentSnr[j]
                c+=1   
        self.__listSNR2chunks=npy.transpose(self.__listSNR2chunks)
             
             
    '''
    DATASET 4/
    
    Produce the noises here we fill everything
    '''

    def _genNoiseSet(self):

        self.__Noise=[]
        for j in range(self.__nTtot):
            dim=int(self.__listTtot[j]*self.__listfe[j])
            self.__Noise.append(npy.zeros((self.__Ntemplate*2*self.__NbB,dim)))

        for i in range(0,self.__Ntemplate*self.__NbB*2):
            if i%1000==0:
                print("Producing sample ",i,"over",self.__Ntemplate*self.__NbB*2)

            temp=self.__noise.getNewSample()
            for j in range(self.__nTtot):
                self.__Noise[j][i]=temp[j]


    def _GenNoiseSequence(self,duration):

        # Noise is produced in chunks of length Ttot 
        # from PSD using inverse FFTs. As the total duration
        # can be slightly larger, we need to take care of 
        # continuity between the different sequences
        # We therefore add a short screening window around each chunk

        taper=0.2 # tapering time, in seconds 
        nsamples=int(duration/(self.__noise.Ttot()-taper))
        if duration/(self.__noise.Ttot()-taper)-nsamples>0:
            nsamples+=1

        npts=int(duration*self.__fe)


        white_noise=[] # Will contain the noise sequence whithened with the PSD
        raw_noise=[]   # Raw noise (no whitening)

        # Do some windowing at start/end of each chunck (200ms), in order to 
        # avoid spectral leakage
        # Try to interleave starting and ending windows in order to reduce the impact of tapering
        # but in pratice the noise in this period of length taper is underestimated 

        npts_black=2*int(taper*self.__fe)
        window=npy.blackman(npts_black)
        winsize=int(npts_black/2)         

        end_w=[]
        end_r=[]


        for i in range(nsamples): # For each chunck we instantiate a new noise sequence
            
            self.__noise.getNewSample()
            noise=self.__noise.getNoise()
            noise_raw=self.__noise.getNoise_unwhite()

            if i>0:
                noise[:winsize]=noise[:winsize]*window[:winsize]+end_w
                noise_raw[:winsize]=noise_raw[:winsize]*window[:winsize]+end_r
            else:
                noise[:winsize]=noise[:winsize]*window[:winsize]
                noise_raw[:winsize]=noise_raw[:winsize]*window[:winsize]
            noise[len(noise)-winsize:]*=window[winsize:]
            noise_raw[len(noise_raw)-winsize:]*=window[winsize:]

            white_noise.append(noise[:len(noise)-winsize])
            raw_noise.append(noise_raw[:len(noise)-winsize])

            end_w=noise[len(noise)-winsize:]
            end_r=noise_raw[len(noise_raw)-winsize:]

        white_noise = npy.ravel(npy.squeeze(white_noise))[0:npts] 
        raw_noise   = npy.ravel(npy.squeeze(raw_noise))[0:npts]

        # Finally store sequences 
        self.__Noise.append([white_noise,raw_noise]) 
        self.__Signal.append([npy.zeros(len(white_noise)),npy.zeros(len(white_noise))])

        for j in range(self.__nTtot): # Loop over samples
            
            # The following lines are for the SNR repartition
            self.__listSNR2chunks[j].append(npy.full((1),(1.0/self.__nTtot)))


        self.__listSNR2chunks=npy.reshape(self.__listSNR2chunks, (self.__nTtot, 1))
        self.__listSNR2chunks=npy.transpose(self.__listSNR2chunks)
 
             
 

    '''
    DATASET 5/
    
    Get a dataset from the noise and signal samples

    SNRopt provides the SNR of the signal to add, could be a range
    size provides the number of samples you want to pick up, 0 means the full set

    '''

    def getDataSet(self,SNRopt=[1],size=0):
        nbands=self.__nTtot
        dset=[]
        pureset=[]
        fdset = []
        fpureset = []
        finaldset=[]
        finalpset=[]
        labels=[]

        list_weights=self.__listSNR2chunks

        if self.__verb:        
            print("Getting a training set of",self.Nsample,"events based on",nbands,"frequency bands")

        # If the SNR is within a range, we define a vector containing the random SNR values
        if len(SNRopt)==2:
            randSNR=npy.random.uniform(SNRopt[1],SNRopt[0], size=self.Nsample)

        for i in range(nbands):
            if len(SNRopt)==2:
                temp=(self.__Sig[i].T*randSNR).T
                dset.append(temp+self.__Noise[i]) # Sig=0 in the second half, so this is just noise...
                pureset.append(temp) 
            else:
                dset.append(self.__Sig[i]*(SNRopt)+self.__Noise[i])
                pureset.append(self.__Sig[i]*(SNRopt))

        # Dataset has form ([Nsample][N1],[Nsample][N2],...)
        # Reorganize it as [Nsample][N1+N2]
        ntemp=size   # Can choose the size (can be handy to test quickly new architectures)
        dist=[]
        if ntemp==0 or ntemp>int(self.Nsample/2):
            ntemp=int(self.Nsample/2)
        else:
            dist=npy.random.randint(int(self.Nsample/2), size=size)
        
        for i in range(ntemp):
            if len(dist)==0:
                idx=i
            else:
                idx=dist[i]
            tempset=[] # Signal
            temppset=[] # Signal
            for j in range(nbands):
                sect=npy.asarray(dset[j][idx])
                tempset.append(sect)
                sectp=npy.asarray(pureset[j][idx])
                temppset.append(sectp)
            sec=npy.concatenate(tempset)
            labels.append(self.__Labels[idx])
            finaldset.append(sec)
            secp=npy.concatenate(temppset)
            finalpset.append(secp)

            tempset=[] # Noise
            temppset=[] # Signal
            for j in range(nbands):
                sect=npy.asarray(dset[j][self.Nsample-1-idx])
                tempset.append(sect)
                sectp=npy.asarray(pureset[j][self.Nsample-1-idx])
                temppset.append(sectp)
            sec=npy.concatenate(tempset)
            labels.append(self.__Labels[self.Nsample-1-idx])
            finaldset.append(sec)
            secp=npy.concatenate(temppset)
            finalpset.append(secp)

        fdset=npy.asarray(finaldset)
        fpureset=npy.asarray(finalpset)

        return fdset, list_weights, fpureset, labels


    def getFrame(self,det=0):
        nbands=self.__nTtot
        dset=[]
        fdset = []
        pset=[]
        fpset = []
        finaldset=[]
        
        list_weights=self.__listSNR2chunks
        
        print("Getting a frame which will be analyzed with",nbands,"frequency bands")

        #print(len(self.__Noise),len(self.__Noise[0]),len(self.__Noise[0][0][0]))

        dset.append(self.__Noise[det][0])
        pset.append(self.__Signal[det][0])

        fdset=npy.asarray(dset[0])
        fpset=npy.asarray(pset[0])

        #print("In getframe",fdset.shape,fpset.shape)

        return fdset, list_weights, fpset
    
    
    '''
    DATASET 7/
    
    Plots
    '''
            
    def plot(self,i,SNRopt=1):
        plt.plot(self.__NGenerator.T,self.getDataSet(SNRopt)[0][i],'.')
        
    def plotSNRmap(self):
        mstep = self.__step # pas des masses à faire choisir plus tard
        mlow=float(self.__mint[0])
        mhigh=float(self.__mint[1])
        
        Nbmasses=int((mhigh-mlow)/mstep)
        residual=(mhigh-mlow)/mstep-Nbmasses
        if residual>0:
            Nbmasses=Nbmasses+1

        X, Y = npy.meshgrid(npy.linspace(mlow-mstep/2., mhigh+mstep/2., Nbmasses+1), npy.linspace(mlow-mstep/2., mhigh+mstep/2., Nbmasses+1))
        
        for k in range(self.__nTtot):
            Z=npy.zeros((Nbmasses,Nbmasses))
            c=0
            for i in range(Nbmasses):
                if c==len(self.__listSNR2evol/self.__nTtot):
                    break
                for j in range(i,Nbmasses):
                    #print(i,j,self.__listSNR2evol[self.__nTtot*c+k])
                    Z[i][j] = self.__listSNR2evol[self.__nTtot*c+k]
                    if (i!=j):
                        Z[j][i] = self.__listSNR2evol[self.__nTtot*c+k]
                    c+=1
                    if c==len(self.__listSNR2evol/self.__nTtot):
                        break
            plt.figure('MapDistribution_SNR**2 : chunck N°'+str(k))
            plt.pcolormesh(X,Y,Z)
            plt.xlabel('m1')
            plt.ylabel('m2')
            plt.colorbar()
            plt.title(label='Proportion of the total collectable power collected in the plan (m1,m2) : chunk N°'+str(k))
            plt.show()

    ##
    #
    # Save training samples 
    #
    ##


    def saveGenerator(self):

        # Save the sample in an efficient way
        for k in range(self.__nTtot):
            fname=self.__fname+'_set_chunk-'+str(k)+'of'+str(self.__nTtot)+'_samples'
            npy.savez_compressed(fname,self.__Noise[k],self.__Sig[k])
            self.__listfnames.append(fname)
        
        # Save the generator object in a pickle without the samples
        self.__Sig=[]
        self.__Noise=[]
        fichier=self.__fname+'-'+str(self.__nTtot)+'chunk'+'.p'
        f=open(fichier, mode='wb')
        pickle.dump(self,f)
        f.close()
                
    ##
    #
    # Save frames 
    #
    ##
            
    def saveFrame(self):

        # Save the sample in an efficient way
        fname=self.__fname+'-'+str(self.__length)+'s'+'-frame'
        npy.savez_compressed(fname,self.__Noise,self.__Signal)
        self.__listfnames.append(fname)
        
        gpsinit=1400000000
        
        from gwpy.timeseries import TimeSeries

        t = TimeSeries(self.__Noise[0][1],channel="Noise_and_injections",sample_rate=self.__fe,unit="time",t0=self.__length+gpsinit)
        u = TimeSeries(self.__Noise[0][1]-self.__Signal[0][1],channel="Noise_alone",sample_rate=self.__fe,unit="time",t0=gpsinit)
        v = TimeSeries(self.__Noise[0][1]-self.__Signal[0][1],channel="Noise_alone",sample_rate=self.__fe,unit="time",t0=gpsinit+self.__length)
        t.write(f"{fname}_Strain.gwf")
        u.write(f"{fname}_Noise.gwf") # MBTA needs pure noise to compute the PSD
        v.write(f"{fname}_Noise2.gwf") 
        
        # Save the object without the samples (basically just the weights)
        self.__Signal=[]
        self.__Noise=[]
        fichier=fname+'-'+str(self.__nTtot)+'chunk_frame'+'-'+str(self.__length)+'s'+'.p'
        f=open(fichier, mode='wb')
        pickle.dump(self,f)
        f.close()

    @classmethod
    def readGenerator(cls,fichier):
        f=open(fichier, mode='rb')
        obj=pickle.load(f)
        
        print("We deal with a dataset containing",obj.__nTtot,"frequency bands")
        for i in range(obj.__nTtot):
            print("Band",i,"data is contained in file",obj.__listfnames[i]+'.npz')
            data=npy.load(str(obj.__listfnames[i])+'.npz')
            obj.__Sig.append(data['arr_1'])
            obj.__Noise.append(data['arr_0'])
        data=[] # Release space
        f.close()
        return obj
    
    @classmethod
    def readFrame(cls,fichier):
        f=open(fichier, mode='rb')
        obj=pickle.load(f)
        
        print("Opening the frame")

        data=npy.load(str(obj.__listfnames[0])+'.npz')
        #obj.__Sig.append(data['arr_1'])
        obj.__Noise = data['arr_0']
        obj.__Signal = data['arr_1']
        data=[] # Release space
        f.close()
        return obj
    
    def getTruth(self):
        return self.__injections
    
    def getBkParams(self):
        return self.__tmplist
    
    #def getTemplate(self,rank=0):
    #    return self.__Sig[0][rank]

    def type(self):
        return self.__rType

    def verb(self):
        return self.__verb

    def sigprop(self):
        return self.__prop

    def sampleprop(self):
        return [self.__listTtot,self.__listfe,self.__kindPSD,
                self.__kindTemplate,self.__kindBank,self.__fmin,
                self.__fmax,self.__white]

    @property
    def Ntemplate(self):
        return self.__Ntemplate
    @property
    def Nsample(self):
        return self.__Ntemplate*2*self.__NbB
    @property
    def Labels(self):
        return self.__Labels
    @property
    def mInt(self):
        return self.__mint[0],self.__mint[1]
    @property
    def mStep(self):
        return self.__step
    @property
    def kindPSD(self):
        return self.__kindPSD
    @property
    def kindTemplate(self):
        return self.__kindTemplate
    @property
    def kindBank(self):
        return self.__kindBank
