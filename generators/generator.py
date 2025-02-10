'''
*** Generator ***

Central class for data samples production. 

Based on the parameters defined in csv option files, this class
will produce either the samples for ML training, or simple frames containing 
noise and injections

'''


import numpy as npy
import matplotlib.pyplot as plt
import pickle
import csv
import os
import time
from numcompress import compress, decompress

import MLGWtools.generators.noises as gn
import MLGWtools.generators.signals as gt


######################################################################################################################################################################################
parameters = {'font.size': 15,'axes.labelsize': 15,'axes.titlesize': 15,'figure.titlesize': 15,'xtick.labelsize': 15,'ytick.labelsize': 15,'legend.fontsize': 15,'legend.title_fontsize': 15,'lines.linewidth' : 3,'lines.markersize' : 10, 'figure.figsize' : [10,5]}
plt.rcParams.update(parameters)
#################################################################################################################################################################################################



class Generator:



    '''
    GENERATOR 1/
    
    Object init. A csv file is always required
    '''

    def __init__(self,paramFile):
    
        if os.path.isfile(paramFile):
            self._readParamFile(paramFile)
        else:
            print("No init file provided, you're on your own there...")
    
    '''
    GENERATOR 2/

    Produce a data frame with injections. Injections can be randomly chosen or selected from a txt file
    provided in the job options
    '''
    
    def buildFrame(self):
        
        if self.__rType!='frame':
            raise Exception("Your job type is not frame, you are not supposed to use this function")
        
        # Instantiate the objects for noise and injection prod

        self.initNoise()
        self.initTemplate()

        # First produces the noise sequence with the correct duration

        nitf = 1  # How many detectors (keep it to one for the moment) ? 

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
                coord=npy.array(ligne_aleatoire,dtype=float)
                # !id,m1,m2,spin1x,spin1y,spin1z,spin2x,spin2y,spin2z,chieff,chip,tcoal,SNRH,SNRL
                id = int(coord[0])
                m1 = coord[1]
                m2 = coord[2]
                # Bank format
                self.__signal.majParams(coord[1],coord[2],s1x=0.,s2x=0.,
                                        s1y=0.,s2y=0.,s1z=coord[3],s2z=coord[4])
                # Inj file format (SV need to make it complient)
                #self.__signal.majParams(coord[1],coord[2],s1x=coord[3],s2x=coord[6],
                #                        s1y=coord[4],s2y=coord[7],s1z=coord[5],s2z=coord[8])
            else :
                id = -1
                m1=npy.random.uniform(5,75)
                m2=npy.random.uniform(5,75)
                self.__signal.majParams(m1,m2)

            SNR=npy.random.uniform(5,50)

            self.__signal.getNewSample(Tsample=self.__signal.gensignal())
            data=self.__signal.signal()        # Whitened and normalised to SNR**2=1
            data_r=self.__signal.signal_raw()/self.__signal.norma()  # Raw and normalised

            randt=npy.random.normal((i+1)*interval,interval/5.)

            # Where to add the first signal in the frame
            idxstart=int((randt-self.__signal.gensignal())*self.__listfe[0])
            idxend=int(randt*self.__listfe[0])
            if idxstart<0:
                idxstart=0
            if idxend>=len(self.__Signal[0][0]):
                idxend=len(self.__Signal[0][0])
            
   
            length=idxend-idxstart
            idxmax=npy.argmax(npy.abs(data[-length:]))
            tpeak=float(idxmax+idxstart)/self.__listfe[0]
            
            inj=[id,m1,m2,SNR,tpeak]
            self.__injections.append(inj)

            print("Injection",i,"(id,m1,m2,SNR,tc)=(",f'{id}',f'{m1:.1f}',f'{m2:.1f}',f'{SNR:.1f}',f'{tpeak:.1f}',")")

            self.__Signal[0][0][idxstart:idxend]+=SNR*data[-length:]
            self.__Signal[0][1][idxstart:idxend]+=SNR*data_r[-length:]
            data=[]
            data_r=[]

        # Injection process over, __Signal[0][0] now contains all the injections whitened, 
        # and raw in __Signal[0][1]
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

    '''
    GENERATOR 3/

    Produce a bank of data for network tests
    '''

    def buildSample(self):
        
        if self.__rType!='bank' and self.__rType!='ffactor' :
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
        
        if self.__rType=='bank':
            self._genSigSet()   # The signals
        else:
            self._genSigSet_ff() # Signals without noise
            print("3 After signal --- %s seconds ---" % (time.time() - start_time))
            return 

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


    '''
    GENERATOR 4/

    Interfaces with the classes handling noise and signal generation
    '''


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

    def initTemplate_TD(self,length):
        self.__signal = gt.GenTemplate(Ttot=length,fe=self.__listfe,kindPSD=self.__kindPSD,
                                 kindTemplate=self.__kindTemplate,
                                 fmin=self.__fmin,fmax=self.__fmax,whitening=self.__white,
                                 verbose=self.__verb,customPSD=self.__custPSD)

    def getTemplate(self):
        return self.__signal
 


    '''
    GENERATOR 5/
    
    Parser of the parameters csv file
    '''

    def _readParamFile(self,paramFile):


        fdir=paramFile.split('/')
        self.__fname=fdir[len(fdir)-1].split('.')[0]
       
        with open(paramFile) as mon_fichier:
              mon_fichier_reader = csv.reader(mon_fichier, delimiter=',')
              lignes = [x for x in mon_fichier_reader]

        # Default parameters
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
        self.__tcint=[0.7,0.95]        
        self.__ninj=10
        self.__length=1000.
        self.__injparams=[]
        self.__start=0
        self.__stop=100    

        # List of available commands
        cmdlist=['runType','Ttot','fe','kindPSD','properties',
                 'mint','tcint','NbB','kindTemplate','kindBank','step',
                 'whitening','flims','verbose','length','ninj','injlist','start','stop']
        
        # Retieving the info in the job option
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
                    self.__kindBank='inputfile'
                    print("Retrieving injections/bank parameters from file",self.__injlist)
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

            if cmd=='start':
                self.__start=int(line[1])

            if cmd=='stop':
                self.__stop=int(line[1])

            if cmd=='step':
                self.__step=float(line[1])

            if cmd=='length':
                self.__length=float(line[1])

            if cmd=='ninj':
                self.__ninj=int(line[1])

            if cmd=='tcint':
                self.__tcint[0]=float(line[1])
                self.__tcint[1]=float(line[2])


    # Read the injection list provided
    def _readtxtFile(self,txtfile) :
        with open(txtfile, 'r') as f:
            lignes = f.readlines()

        lignes_parametres = [ligne for ligne in lignes if not ligne.startswith('#') and not ligne.startswith('!')]
        return lignes_parametres

    '''
    GENERATOR 6/
    
    Bank production
    Produce a data grid with the mass coordinates
    '''

    def _genGrille(self):

        # Default is a regular grid in the m1/m2 plan (linear)

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
                    
        else: # Read from a text file, not used for the moment
            Mtmp=[]
            Sztmp=[]
            start=self.__start
            ntmps=self.__stop
            compt=0
            print("Templates are taken into file ",self.__injlist)
            with open(self.__injlist) as mon_fichier:
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
    GENERATOR 7/
    
    Bank production
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
    
    # This one produces pure signals, and is used for fitting factor calculation only

    def _genSigSet_ff(self):

        self.__Sig=[]
        self.__Noise=[]
        c=0
        
        # First we produce the object with the correct size
        # The size is Ntemplate
        for j in range(self.__nTtot): # Loop over samples

            self.__Sig.append(npy.zeros(self.__Ntemplate,dtype=object))
            self.__Noise.append(npy.zeros(self.__Ntemplate,dtype=object))


        # Now fill the object
        for i in range(0,self.__Ntemplate):

            if c%10==0:
                print("Producing sample ",c,"over",self.__Ntemplate)
            self.__signal.majParams(fast=True,m1=self.__GrilleMasses[i][0],m2=self.__GrilleMasses[i][1],s1z=self.__GrilleSpins[i][0],s2z=self.__GrilleSpins[i][1])
        
            # Create the template            
            temp,freqs=self.__signal.getNewSample(tc=npy.random.uniform(self.__tcint[0],self.__tcint[1]))

            # Fill the corresponding data
            for j in range(self.__nTtot):

                if (not isinstance(freqs[j],int)):

                    trunc=0
                    if (len(temp[j])>len(freqs[j])):
                        trunc=len(temp[j])-len(freqs[j])

                    # Compress the data
                    self.__Sig[j][c]=compress(list(temp[j][trunc:]), precision=10)
                    self.__Noise[j][c]=compress(list(freqs[j]), precision=5)
                else:
                    self.__Sig[j][c]=0
                    self.__Noise[j][c]=0

            del temp,freqs            
            c+=1

             
    '''
    GENERATOR 8/
    
    Bank production
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

    '''
    GENERATOR 9/
    
    Frame production
        
    Noise is produced in chunks of length Ttot 
    from PSD using inverse FFTs. As the total duration
    can be slightly larger, we need to take care of 
    continuity between the different sequences
    We therefore add a short screening window around each chunk

    '''

    def _GenNoiseSequence(self,duration):



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
    GENERATOR 10/
    
    Get a dataset from a bank 

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
                dset.append(self.__Sig[i]*(SNRopt[0])+self.__Noise[i])
                pureset.append(self.__Sig[i]*(SNRopt[0]))

        # Dataset has form ([Nsample][N1],[Nsample][N2],...)
        # Reorganize it as [Nsample][N1+N2]
        ntemp=size   # Can choose the size (can be handy to test quickly new architectures)
        dist=[]
        if ntemp==0 or ntemp>int(self.Nsample/2):
            ntemp=int(self.Nsample/2)
        else:
            #print('here')
            dist=npy.random.randint(int(self.Nsample/2), size=size)
        
        for i in range(ntemp):
            if len(dist)==0:
                idx=i
            else:
                idx=dist[i]

            #print(idx)
            tempset=[] # Signal
            temppset=[] # Signal
            for j in range(nbands):
                sect=npy.asarray(dset[j][idx])
                tempset.append(sect)
                sectp=npy.asarray(pureset[j][idx])
                temppset.append(sectp)
            sec=npy.concatenate(tempset)
            #print("Signal",self.__Labels[idx])
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
            #print("Noise",self.__Labels[self.Nsample-1-idx])
            labels.append(self.__Labels[self.Nsample-1-idx])
            finaldset.append(sec)
            secp=npy.concatenate(temppset)
            finalpset.append(secp)

        fdset=npy.asarray(finaldset)
        fpureset=npy.asarray(finalpset)

        return fdset, list_weights, fpureset, labels


    '''
    GENERATOR 10/
    
    Retrieve a frame 
    '''

    def getFrame(self,det=0):
        nbands=self.__nTtot
        dset=[]
        fdset = []
        pset=[]
        fpset = []
        
        list_weights=self.__listSNR2chunks
        
        print("Getting a frame which will be analyzed with",nbands,"frequency bands")

        dset.append(self.__Noise[det][0])
        pset.append(self.__Signal[det][0])

        fdset=npy.asarray(dset[0])
        fpset=npy.asarray(pset[0])

        return fdset, list_weights, fpset


    def getrawFrame(self,det=0):
        nbands=self.__nTtot
        dset=[]
        fdset = []
        pset=[]
        fpset = []
        
        list_weights=self.__listSNR2chunks
        
        print("Getting a raw frame which will be analyzed with",nbands,"frequency bands")

        dset.append(self.__Noise[det][1])
        pset.append(self.__Signal[det][1])

        fdset=npy.asarray(dset[0])
        fpset=npy.asarray(pset[0])

        return fdset, list_weights, fpset
    
    '''
    GENERATOR 11/
    
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
    
    '''
    GENERATOR 12/
    
    Macros saving and retrieving files in different formats 
    '''

    # For the banks
    def saveGenerator(self):

        # Save the sample in an efficient way
        for k in range(self.__nTtot):
            fname=self.__fname+'_set_chunk-'+str(k)+'of'+str(self.__nTtot)+'_samples'
            npy.savez_compressed(fname,self.__Noise[k],self.__Sig[k])
            self.__listfnames.append(fname)
        
        # Save the generator object in a pickle without the samples
        self.__Sig=[]
        self.__Noise=[]
        if self.__rType=='ffactor':
            self.__signal=[]
            self.__noise=[]
        fichier=self.__fname+'-'+str(self.__nTtot)+'chunk'+'.p'
        f=open(fichier, mode='wb')
        pickle.dump(self,f)
        f.close()
                

    # For the frames   
    def saveFrame(self):

        # Save the sample in an efficient way
        fname=self.__fname+'-'+str(self.__length)+'s'+'-frame'
        npy.savez_compressed(fname,self.__Noise,self.__Signal)
        self.__listfnames.append(fname)
        '''
        gpsinit=1400000000
        
        # Also save into the canonical GWF format, to test data in MBTA for example
        from gwpy.timeseries import TimeSeries

        t = TimeSeries(self.__Noise[0][1],channel="Noise_and_injections",sample_rate=self.__fe,unit="time",t0=self.__length+gpsinit)
        u = TimeSeries(self.__Noise[0][1]-self.__Signal[0][1],channel="Noise_alone",sample_rate=self.__fe,unit="time",t0=gpsinit)
        v = TimeSeries(self.__Noise[0][1]-self.__Signal[0][1],channel="Noise_alone",sample_rate=self.__fe,unit="time",t0=gpsinit+self.__length)
        t.write(f"{fname}_Strain.gwf")
        u.write(f"{fname}_Noise.gwf") # MBTA needs pure noise to compute the PSD
        v.write(f"{fname}_Noise2.gwf") 
        '''

        # Save the object without the samples (basically just the weights)
        self.__Signal=[]
        self.__Noise=[]
        #self.__injections=[]
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
        obj.__Noise = data['arr_0']
        obj.__Signal = data['arr_1']
        data=[] # Release space
        f.close()
        return obj
    
    '''
    GENERATOR 13/
    
    Getters
    '''


    def getTruth(self):
        return self.__injections
    
    def getBkParams(self):
        return self.__tmplist

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
