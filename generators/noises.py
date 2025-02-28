import numpy as npy
import scipy
import matplotlib.pyplot as plt
import random
from scipy import signal
import scipy.fft

'''
Class handling noise generation

Option:

->Ttot     : noise samples duration, in seconds (Default is 1)
->fe       : sampling frequencies, in Hz (Default is 2048)
->kindPSD  : noise type: 'flat', 'analytic', or 'realistic' (Default is g. flat).
             Noise is stationary in both cases (see below)
->fmin     : minimal frequency for noise definition (Def = 20Hz)
->fmax     : maximal frequency for noise definition (Def = 1500Hz)
->whitening: type of signal whitening:
                0: No whitening
                1: Frequency-domain whitening (Default standard procedure)
                2: Time-domain whithening (Zero latency, as described in https://dcc.ligo.org/public/0141/P1700094/005/main_v5.pdf)
            

flat noise is just a gaussian noise with constant sigma over all frequency.
analytic is a bit more evolved, and takes into account the different theoretical contribution
            still it's not based on real data, ie not including glitches for example.
            So that's gaussian colored noise
            Reference used for analytic noise is cited below
realistic is based on ever more realistic PSD, taken a input file (see generator.py)

Noise is produced over a given frequency range.

Indeed there is no reason to produce noise well outside
detector acceptance

'''

class Noises:

    def __init__(self,Ttot=1,fe=2048,kindPSD='flat',fmin=20,fmax=1500,whitening=1,customPSD=None,verbose=False):

        if not((isinstance(Ttot,int) or isinstance(Ttot,float) or isinstance(Ttot,list)) and (isinstance(fe,int) or isinstance(fe,float) or isinstance(fe,list))):
            raise TypeError("Ttot et fe doivent être des ints, des floats, ou des list")
        if not(isinstance(kindPSD,str)):
            raise TypeError("kindPSD doit être de type str")
        if kindPSD!='flat' and kindPSD!='analytic' and kindPSD!='realistic':
            raise ValueError("Les seules valeurs autorisées pour kindPSD sont 'flat','analytic', et 'realistic'")

        if verbose:
            print("GEN NOISE: VERBOSE MODE ON")

        # Deal with the fact that we can sample the frame with different frequencies
        if isinstance(Ttot,list):
            if not isinstance(fe,list):
                raise TypeError("Ttot et fe doivent être du même type")
            elif not len(Ttot)==len(fe):
                raise ValueError("Les list Ttot et fe doivent faire la même taille")
            else:
                self.__listTtot=Ttot           # List of chunk lengths
                self.__listfe=fe               # List of corresponding sampling freqs
                self.__Ttot=sum(Ttot)          # Total sample length
                self.__fe=max(fe)              # Max sampling freq
                self.__nTtot=len(Ttot)         # Total number of subsamples
        else:
            self.__Ttot=Ttot                   # Total sample length
            self.__fe=fe                       # Sampling freq
            self.__nTtot=1
        
        # We will generate a sample with the total length and the max sampling freq, and resample
        # only at the end
        
        self.__whiten=whitening
        self.__verb=verbose
        self.__fmin=fmin
        self.__fmax=fmax
        self.__N=int(self.__Ttot*self.__fe)    # The total number of time steps produced
        self.__delta_t=1/self.__fe             # Time step
        self.__delta_f=self.__fe/self.__N      # Frequency step
        self.__kindPSD=kindPSD                 # PSD type
        self.__customPSD=customPSD             # For realistic case a PSD vector is provided

        # N being defined we can generate all the necessary vectors

        self.__T=npy.arange(self.__N)*self.__delta_t  # Time values
        
        # Frequencies (FFT-friendly)
        self.__F=npy.concatenate((npy.arange(self.__N//2+1),npy.arange(-self.__N//2+1,0)))*self.__delta_f
        self.__Fnorm=self.__F/(self.__fe/2.)
        
        self.__PSD=npy.ones(self.__N, dtype=npy.float64) # Setting the PSD to one means infinite noise
        self.__invPSD=npy.ones(self.__N, dtype=npy.float64)

        # Then we produce the PSD which will be use to generate the noise in freq range.

        self.__Nf=npy.zeros(self.__N,dtype=complex)          # Noise FFT
        self.__Nf2=npy.zeros(self.__N,dtype=complex)         # Noise FFT (whitened if required)
        self.__Nfr=npy.zeros(self.__N, dtype=npy.float64)    # Noise FFT real part
        self.__Nfi=npy.zeros(self.__N, dtype=npy.float64)    # Noise FFT imaginary part
        self.__PSDloc=npy.ones(self.__N, dtype=npy.float64)    # Noise FFT imaginary part
        self.__realPSD = npy.zeros(self.__N, dtype=npy.float64) # For the realistic case

        if kindPSD=='realistic' and len(self.__customPSD)>0:
            self._extractPSD(self.__customPSD)

        self._genPSD()
            
        if self.__verb:
            print("____________________________")
            print("Noise generator initialised")
            print("____________________________")

        
    '''
    NOISE 1/9
    
    Noise generation for analytic option, account for shot, thermal, quantum and seismic noises
    This is the one sided PSD, as defined in part IV.A of:
    https://arxiv.org/pdf/gr-qc/9301003.pdf
    '''

    def Sh(self,f):

        ##Shot noise (Eq 4.1)
        hbar=1.05457182e-34 #m2kg/s
        lamda=5139e-10 #m
        etaI0=60 #W
        Asq=2e-5
        L=4e3 #m
        fc=100 #Hz
        
        Sshot=(hbar*lamda/etaI0)*(Asq/L)*fc*(1+(f/fc)**2)
        
        ##Thermal Noise (Eq 4.2 to 4.4)
        kb=1.380649e-23 #J/K
        T=300 #K
        f0=1 #Hz
        m=1000 #kg
        Q0=1e9
        Lsq=L**2
        fint=5e3 #Hz
        Qint=1e5
        Spend=kb*T*f0/(2*(npy.pi**3)*m*Q0*Lsq*((f**2-f0**2)**2+(f*f0/Q0)**2))
        Sint=2*kb*T*fint/((npy.pi**3)*m*Qint*Lsq*((f**2-fint**2)**2+(f*fint/Qint)**2))
    
        Sthermal=4*Spend+Sint
        
        #Seismic Noise (Eq 4.6)
        S0prime=1e-20 #Hz**23
        f0=1 #Hz
        with npy.errstate(divide='ignore'):
            Sseismic=npy.where((f!=f0) & (f!=0),S0prime*npy.power(f,-4)/(f**2-f0**2)**10,(1e-11)**2)
            
        #Quantum noise (Eq 4.8)
        with npy.errstate(divide='ignore'):
            Squant=npy.where((f!=0),8*hbar/(m*Lsq*(2*npy.pi*f)**2),(1e-11)**2)
        
        return (Squant+Sshot+Sthermal+Sseismic)

    '''
    NOISE 2/9
    
    Produce the PSD, so in the noise power in frequency domain
    We don't normalize it, so be carefuf if the signal is not whitened
    We start from single-sided PSD distribution but express them on 
    double-sided arrays, thus there is a factor 0.5 introduced.
    For time domain we stay double sided
    '''

    def _genPSD(self):

        # Frequency range for the PSD
        ifmax=int(min(self.__fmax,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fmin/self.__delta_f)
        
        # Generate the function
        if self.__kindPSD=='flat':
            sigma=2e-23
            self.__PSD[ifmin:ifmax]=0.5*sigma**2
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2] # Double sided
        elif self.__kindPSD=='analytic':
            self.__PSD[ifmin:ifmax]=0.5*self.Sh(abs(self.__F[ifmin:ifmax]))
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2]
        elif self.__kindPSD=='realistic':
            self.__PSD[ifmin:ifmax]=0.5*self.__realPSD[ifmin:ifmax]
            self.__PSD[-1:-self.__N//2:-1]=self.__PSD[1:self.__N//2]

        # Prepare the time-domain whitening filters (FIR-filters), EXPERIMENTAL

        self.__invPSD=npy.sqrt(1./npy.abs(self.__PSD)) # Stay single sided here

        freqs=self.__Fnorm[0:self.__N//2]
        ampl=self.__invPSD[0:self.__N//2]
        for i in range(len(ampl)):
            if ampl[i]<=1.5: # Safety check to avoid rounding issues
                ampl[i]=0.
        freqs[self.__N//2-1]=1.
        ampl[self.__N//2-1]=0.
        self.__nf=ampl.max()
        normamp=ampl/ampl.max()
        self.__whitener = signal.firwin2(self.__N//2, freqs, normamp)
        self.__whitener_MP = signal.minimum_phase(signal.firwin2(self.__N//2, freqs, normamp**2), method='hilbert',n_fft=10*self.__N)


        if self.__verb:

            tmp = npy.zeros(len(self.__whitener))
            tmp[:len(self.__whitener_MP)]=self.__whitener_MP
        
            w, h = signal.freqz(self.__whitener)
            w2, h2 = signal.freqz(self.__whitener_MP)

            plt.title('Digital filter amplitude (FD) ')
            plt.plot(w/npy.pi, npy.abs(h),label="Basic FIR filter")
            plt.plot(freqs, normamp,label="Reference (Inv. PSD)")
            plt.plot(w2/npy.pi, npy.abs(h2),label="Zero-latency FIR filter")
            plt.title('Digital filter frequency response')
            plt.ylabel('Normalized amplitude')
            plt.xlabel('Normalized frequency')
            plt.legend()
            plt.grid()
            plt.show()
            plt.title('Digital filter phase (FD)')
            plt.plot(w, npy.unwrap(npy.angle(h,deg=True)),label="Basic FIR filter")
            plt.plot(w2, npy.unwrap(npy.angle(h2,deg=True)),label="Zero-latency FIR filter")
            plt.title('Digital filter frequency response')
            plt.ylabel('Phase shift')
            plt.xlabel('Normalized frequency')
            plt.grid()
            plt.show()
        
        
    '''
    NOISE 3/9
    
    PSD type change
    '''

    def _changePSD(self,kindPSD):

        del self.__PSD
        self.__PSD=npy.ones(self.__N, dtype=npy.float64)
        if kindPSD!='flat' and kindPSD!='analytic' and kindPSD!='realistic':
            raise ValueError("Les seules valeurs autorisées sont 'flat' et 'analytic'")
        self.__kindPSD=kindPSD
        
        if kindPSD=='realistic' and len(self.__custom)>0:
            self._extractPSD(self.__customPSD)
    
        self._genPSD()


    '''
    NOISE 4/9
    
    Retrieving PSD info from a text file 
    '''

    def _extractPSD(self,PSD):

        self.__custom=PSD
        nPSDs=len(PSD)
        rk=random.randint(0,nPSDs-1)
            
        self._brutePSD=PSD[rk][0]
        freqs=PSD[rk][1]
        ifmax=int(min(self.__fmax,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fmin/self.__delta_f)
        idxprev=ifmin

        count=0
        for i in range(len(freqs)):
            idx=int(freqs[i]/self.__delta_f)
                
            if idx<ifmin or idx>ifmax:
                continue
                
            if idx!=idxprev:
                # Special case (initialization)
                if count==0:
                    self.__realPSD[idxprev:idx]=self._brutePSD[i]
                else:    
                    self.__realPSD[idxprev]/=count
                    self.__realPSD[idxprev:idx]=self.__realPSD[idxprev]
             
                    count=0
                idxprev=idx

            self.__realPSD[idxprev]+=self._brutePSD[i]
            count+=1
                
        self.__realPSD[idxprev:ifmax]=self.__realPSD[idxprev-1]


    '''
    NOISE 5/9
    
    Create the noise in freq domain from the PSD

    We start from a PSD which provides us the power of the noise at a given frequency
    In order to create a noise ralisation, we need first to generate a random noise realisation of the noise
    in the frequency domain

    For each frequency we choose a random value of the power centered on the PSD value, we consider
    that power distribution is gaussian with a width equal to 5% of the central value (rule of thumb, could be improved)

    Then when the power is chosen we choose a random starting phase Phi0 in order to make sure that the
    frequency component is fully randomized.

    Nf (ASD) is filled like that:

    a[0] should contain the zero frequency term (aka DC),
    a[1:n//2] should contain the positive-frequency terms,
    a[n//2+1:] should contain the negative-frequency terms,
    
    PSD and Nf(f)**2 are centered around PSD(f)

    '''
 
    def _genNfFromPSD(self):

        # The power at a given frequency is taken around the corresponding PSD value
        # We produce over the required frequency range

        ifmax=int(min(self.__fmax,self.__fe/2)/self.__delta_f)
        ifmin=int(self.__fmin/self.__delta_f)

        # Add some 5% smearing to the central PSD amplitude
        self.__PSDloc[ifmin:ifmax]=self.__PSD[ifmin:ifmax]*npy.random.normal(1,0.05,ifmax-ifmin)
        self.__PSDloc[-1:-self.__N//2:-1]=self.__PSDloc[1:self.__N//2]

        self.__Nfr[0:self.__N//2+1]=npy.sqrt(self.__PSDloc[0:self.__N//2+1])
        self.__Nfi[0:self.__N//2+1]=self.__Nfr[0:self.__N//2+1]

        # The initial phase is randomized
        # randn provides a nuber following a normal law centered on 0 with sigma=1, so increase a
        # bit to make sure you cover all angular values (thus the factor 100)
        
        phi0=100*npy.random.randn(len(self.__Nfr))
        self.__Nfr*=npy.cos(phi0)
        self.__Nfi*=npy.sin(phi0)
        
        # Brutal filter
        self.__Nfr[ifmax:]=0.
        self.__Nfi[ifmax:]=0.
        self.__Nfr[:ifmin]=0.
        self.__Nfi[:ifmin]=0.
 
        # Then we can define the components
        self.__Nf[0:self.__N//2+1].real=self.__Nfr[0:self.__N//2+1]
        self.__Nf[0:self.__N//2+1].imag=self.__Nfi[0:self.__N//2+1]
        self.__Nf[-1:-self.__N//2:-1]=npy.conjugate(self.__Nf[1:self.__N//2])


    '''
    NOISE 6/9
    
    Get noise signal in time domain from signal in frequency domain (inverse FFT)

    https://numpy.org/doc/stable/reference/generated/numpy.fft.ifft.html

    If whitening option is set to true signal is normalized. The whitened noise should be a gaussian centered on 0 and of width 1.

    '''

    def _genNtFromNf(self):

        self.__Nt=[]                       # Noise in time domain
        self.__Ntnow=[]                    # Non whitened noise (h(t))

        for j in range(self.__nTtot):      # one array per chunk
            self.__Nt.append([])
        
        # Inverse FFT over the total length (0/1)

        # Normalization factor for the FFT
        # In forward mode the norm is 1/N for fft and nothing for ifft
        #
        # So the only remaining normalisation is linked to the PSD binning
        # The power of one bin is equal to PSD*df (size of the bin)
        # For the ASD we take the square root, so sqrt(PSD*df)

        # Raw h(t) is always produced
        self.__Ntnow = scipy.fft.ifft(self.__Nf[:]*npy.sqrt(self.__delta_f),norm='forward').real
        
        if self.__whiten==0:     # No whitening
            self.__Nt[0] = self.__Ntnow
        elif self.__whiten==1:   # Whitening, so we normalise by the ASD
            self.__Nt[0] = scipy.fft.ifft(self.__Nf[:]/npy.sqrt(self.__PSDloc),norm='ortho').real
        elif self.__whiten==2:   #Run the Zero-L FIR filter (EXPERIMENTAL)
            self.__Nt[0] = self.__nf*signal.lfilter(self.__whitener_MP,1,self.__Ntnow)*npy.sqrt(2*self.__delta_t)
      
        
        self.__Ntraw = self.__Nt[0].copy() # Raw data without resampling

        self.__Nf2=scipy.fft.fft(self.__Nt[0],norm='ortho') # Control
  
        #Run the main FIR filter
        #self.__tfilt = self.__nf*signal.lfilter(self.__whitener,1,npy.fft.ifft(self.__Nf[:],norm='ortho').real)


    '''
    NOISE 7/9
    
    Signal resampling (if one asked for multibands)
    '''

    def _resample(self):

        Ntref=self.__Nt[0] # The temporal realization at max sampling
                    
        if self.__verb:
            print("Signal Nt has frequency",self.__fe,"and duration",self.__Ttot,"second(s)")

        for j in range(self.__nTtot):
            if self.__verb:
                print("Chunk",j,"has frequency",self.__listfe[j],"and covers",self.__listTtot[j],"second(s)")
            
            #Pick up the data chunk
            ndatapts=int(self.__listTtot[j]*self.__fe)
            nttt=len(Ntref)
            Nt=Ntref[-ndatapts:]
            Ntref=Ntref[:nttt-ndatapts]
            decimate=int(self.__fe/self.__listfe[j])
            self.__Nt[j]=Nt[::int(decimate)]
    
    
    '''
    NOISE 8/9
    
    The full procedure to produce a noise sample once the noise object has been instantiated
    '''

    def getNewSample(self):

        self._genNfFromPSD()               # Noise realisation in frequency domain
        self._genNtFromNf()                # Noise realisation in time domain
        if self.__nTtot > 1:               # If requested, resample the data
            self._resample()
        return self.__Nt.copy()
   
   
    '''
    NOISE 9/9
    
    Plot macros and getters
    '''

    # The main plot (noise in time domain)
    def plotNoise(self):

        listT=[] # Time of the samples accounting for the diffrent freqs
        if self.__nTtot > 1:
            maxT=self.__Ttot
            for j in range(self.__nTtot):
                delta_t=1/self.__listfe[j]
                N=int(self.__listTtot[j]*self.__listfe[j])
                listT.append(npy.arange((maxT-self.__listTtot[j])/delta_t,maxT/delta_t)*delta_t)
                maxT-=self.__listTtot[j]
        else:
            listT.append(self.__T)
            
        
        for j in range(self.__nTtot):
            plt.plot(listT[j], self.__Nt[j],'-',label=f"noise at {self.__listfe[j]} Hz")

        plt.xlabel('t (s)')
        plt.ylabel('h(t)')
        plt.grid(True, which="both", ls="-")
        plt.legend()
       

    def plotNoiseTW(self):

        npts=len(self.__whitener)
        npts2=len(self.__whitener_MP)

        middle=int(npts/2)

        listT  = npy.arange(npts)*self.__delta_t-(npts/2)*self.__delta_t
        listT2 = npy.arange(npts2)*self.__delta_t
        
        plot1 = plt.subplot2grid((2, 2), (0, 0))
        plot2 = plt.subplot2grid((2, 2), (0, 1))
        plot3 = plt.subplot2grid((2, 2), (1, 0))
        plot4 = plt.subplot2grid((2, 2), (1, 1))

        plot1.plot(listT, self.__whitener,'-')
        plot1.set_title('FIR whitening filter')
        plot2.plot(listT2, self.__whitener_MP,'-')
        plot2.set_title('Zero-L FIR whitening filter')
        plot3.plot(listT[middle-500:middle+500], self.__whitener[middle-500:middle+500],'-')
        plot4.plot(listT2[0:500], self.__whitener_MP[0:500],'-')

        plt.tight_layout()
       


    # The 1D projection (useful to check that noise has been correctly whitened
    def plotNoise1D(self,band):

        print("Freq band:",self.__listfe[band],"Hz")
        
        _, bins, _ = plt.hist(self.__Nt[band],bins=100, density=1)
        mu, sigma = scipy.stats.norm.fit(self.__Nt[band])
        print("Noise properties:")
        print(f"Width: {sigma}")
        print(f"Mean: {mu}")
        
        best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
        plt.plot(bins, best_fit_line)
        plt.title(f'Time domain noise 1D projection for band at {self.__listfe[band]}Hz')

    # Frequency domain
    def plotTF(self):
        
        fmin=self.__fmin
        fmax=self.__fmax

        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf[ifmin:ifmax]),'.',label='n_tilde(f)')
        plt.title('Noise realisation in frequency domain')
        plt.xlabel('f (Hz)')
        plt.ylabel('n_tilde(f) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
             
    # Frequency domain whithened

    def plotTF2(self):
        
        fmin=self.__fmin
        fmax=self.__fmax
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf2[ifmin:ifmax]),'-',label='n_tilde(f)')
        #plt.plot(self.__F[ifmin:ifmax],npy.abs(self.__Nf2_ZL[ifmin:ifmax]),'--',label='n_tilde(f)')
        plt.title('Noise realisation in frequency domain (normalized to ASD)')
        plt.xlabel('f (Hz)')
        plt.ylabel('n_tilde(f) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")

    # PSD
    def plotPSD(self):
        plt.figure(figsize=(10,5))
        fmin=self.__fmin
        fmax=self.__fmax
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        # First create some toy data:

        plt.plot(self.__F[ifmin:ifmax],npy.sqrt(self.__PSD[ifmin:ifmax]),'-',label='Sn(f)')
        plt.plot([1.],[1.])
        plt.title('ASD')
        plt.xlabel('f (Hz)')
        plt.ylabel('Sn(f)^(1/2) (1/sqrt(Hz))')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")
    

    def plotinvPSD(self):
        
        fmin=self.__fmin
        fmax=self.__fmax
        ifmax=self.__N//2 if fmax is None else min(int(fmax/self.__delta_f),self.__N//2)-1
        ifmin=0 if fmin is None else max(int(fmin/self.__delta_f),0)+1
        plt.plot(self.__F[ifmin:ifmax],npy.sqrt(self.__invPSD[ifmin:ifmax]),'-',label='Sn(f)')
        plt.title('inverse PSD')
        plt.xlabel('f (Hz)')
        plt.yscale('log')
        plt.xscale('log')
        plt.grid(True, which="both", ls="-")

    
    def getNf(self):
        return self.__Nf

    @property
    def PSDloc(self):
        return self.__PSDloc

    @property
    def kindPSD(self):
        return self.__kindPSD

    @property
    def PSD(self):
        return self.__PSD

    def getNoise(self):
        return self.__Ntraw
    
    def getNoise_unwhite(self):
        return self.__Ntnow
    
    @property
    def nf(self):
        return self.__nf

    @property
    def whitener_MP(self):
        return self.__whitener_MP

    @property
    def whitener(self):
        return self.__whitener
        
    @property
    def length(self):
        return self.__N
        
    @property
    def T(self):
        return self.__T
        
    def Ttot(self):
        return self.__Ttot

