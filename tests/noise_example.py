import MLGWtools.generators.generator as gd
import argparse
import matplotlib.pyplot as plt

def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument("param",help="Fichier csv contenant les options du job",default='test/noise.csv')

    args = parser.parse_args()
    return args


    
args = parse_cmd_line()
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
    