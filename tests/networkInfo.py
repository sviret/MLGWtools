import numpy as npy
import pickle
from tensorflow.keras.utils import plot_model
npy.set_printoptions(threshold=npy.inf)


'''
Print some info about the network and its training
'''


'''
Command line parser
'''

def parse_cmd_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--network","-n", help="Fichier pickle contenant le reseau entrainé",default=None)
   
    args = parser.parse_args()
    return args

'''
The main training macro starts here
'''

def main():
    
    # 1. Start by parsing the input options
    args = parse_cmd_line()
                
    # 2. Then retrieve the network and initialize it
    f=open(args.network,mode='rb')
    net=pickle.load(f)
    f.close()

    

    # get the file name
    netname=((args.network).split("/")[-1]).split(".p")[0]

    # and the network object
    model=net[1].model

    n_mult=0
    n_add=0

    f = open(f'output_{netname}.txt','w')

    f.write(f"Print network information for input net {netname}\n")
    f.write("\n")

    layers=model.layers
    compt=0

    f.write(f"This network contains {len(layers)} layers\n")
    f.write("\n")    

    for layer in layers:
      

        data=model.get_layer(layer.name).get_weights()
        f.write(f"-> Layer {compt}\n")
        f.write(f"Name: {layer.name}\n")

        compt+=1

        if compt==1: # Just input vector as first entry 
            continue

        f.write(f"Input size {layer.input.shape}\n")
        f.write(f"Output size {layer.output.shape}\n")
        f.write(f"Parameters:\n")

        if len(data)==0:
            continue
        for wgh in data:
            wght=npy.asarray(wgh)
            f.write(f"   ->Param container size {wght.shape}\n")
            f.write(f"   {wght}\n")

            if 'conv1d' in layer.name and len(wght.shape)==3:
                
                mult=layer.output.shape[1]*wght.shape[0]*wght.shape[1]*wght.shape[2]
                add=mult
                n_mult+=mult
                n_add+=add

            if 'batch_norm' in layer.name:
                #https://keras.io/api/layers/normalization_layers/batch_normalization/
                
                # Batch normalization operation 
                # x_norm = gamma*(x - mean)/sqrt(var) + beta
                # gamma = wght[0]
                # beta  = wght[1]
                # mean  = wght[2]
                # var   = wght[3]

                # In inference mode mean and var are fixed 
                # So 4 params
                # And 1 mult and 2 addition per sample
                mult=layer.input.shape[1]
                add=layer.input.shape[1]*2
                n_mult+=mult
                n_add+=add

            if 'dense' in layer.name and len(wght.shape)==2:
       
                mult=wght.shape[0]*wght.shape[1]
                add=wght.shape[0]
                n_mult+=mult
                n_add+=add

        f.write(f"\n")


    f.write(f"Summary     :\n")
    f.write(f"-> Total number of additions      : {n_add}\n")
    f.write(f"-> Total number of multiplications: {n_mult}\n")
    f.close()
    plot_model(model, to_file=f'output_{netname}.png', show_shapes=True)

    
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
