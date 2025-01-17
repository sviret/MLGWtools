#
# Produce a small sample, either pure noise or signal
#

import MLGWtools.generators.generator as gd
import argparse
import matplotlib.pyplot as plt

def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument("param",help="Fichier csv contenant les options du job",default='generators/samples/noise.csv')

    args = parser.parse_args()
    return args


# Main macro starts here

# Get the job params stored in the csv file
args = parse_cmd_line()

# Instantiate the generator object 
gen=gd.Generator(paramFile=args.param)

if gen.type() =='noise': # Noise-only generation

    gen.initNoise()
    noise = gen.getNoise()
    noise.getNewSample()
    
    plt.figure(figsize=(10,5))
    noise.plotNoise()
    
    if gen.verb():
        plt.figure(figsize=(10,5))
        noise.plotNoiseTW()
        
        plt.figure(figsize=(10,5))
        noise.plotPSD()
    
        plt.figure(figsize=(10,5))
        noise.plotNoise1D(0)

        plt.figure(figsize=(10,5))
        noise.plotinvPSD()
        
    plt.show()

if gen.type() =='template': # Get some signal

    gen.initTemplate()
    sig=gen.getTemplate()
    info=gen.sigprop()

    sig.majParams(m1=info[0],m2=info[1])
    sig.getNewSample()

    plt.figure(figsize=(10,5))
    sig.plot(Tsample=sig.duration(),tc=0.95,SNR=7)
  
    if gen.verb():
        plt.figure(figsize=(10,5))
        sig.plotTF()
            
        plt.figure(figsize=(10,5))
        sig.plotTFn()
    
        plt.figure(figsize=(10,5))
        sig.plotSignal1D()
        
        plt.figure(figsize=(10,5))
        sig.plotSNRevol()
       
    plt.show()