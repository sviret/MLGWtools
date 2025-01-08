from MLGWtools.generators import generator as gd
import argparse
import matplotlib.pyplot as plt


def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument("param",help="Fichier csv contenant les options du job",default='test/noise.csv')

    args = parser.parse_args()
    return args


args = parse_cmd_line()
gen=gd.Generator(paramFile=args.param)

if gen.type() =='template': # Noise-only generation

    gen.initTemplate()
    sig=gen.getTemplate()
    info=gen.sigprop()

    gen.initNoise()
    noise = gen.getNoise()
    noise.getNewSample()

    sig.majParams(m1=info[0],m2=info[1])
    sig.getNewSample()
        
    plt.figure(figsize=(10,5))
    noise.plotNoise()
    sig.plot(Tsample=sig.duration(),tc=0.95,SNR=7)
  
    plt.figure(figsize=(10,5))
    sig.plotTF()
            
    plt.figure(figsize=(10,5))
    sig.plotTFn()
    
    plt.figure(figsize=(10,5))
    sig.plotSignal1D()
        
    plt.figure(figsize=(10,5))
    sig.plotSNRevol()
       
    plt.show()
    