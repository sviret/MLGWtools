'''
Macro showing the training of the CNN standard network
'''

from MLGWtools.networks import utils as tn
from MLGWtools.networks import CNN as cnn
from MLGWtools.utils import useResults as ur
import argparse

def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument("param",help="Fichier csv contenant les options du job",default='test/trainHuerta.csv')

    args = parser.parse_args()
    return args


'''
The main training macro starts here
'''

# Initialisation
args = parse_cmd_line()
trainer=tn.trutils(paramFile=args.param)
net=cnn.Multiple_CNN(trainer)

# Instantiate the network architecture

# The main network of the Huerta and George paper (the small one)
#net.huerta_legacy()
net.fpga_version()

# Here you can define your custom autoencoder
# Providing the number of CONV layers, along with their properties:
# kernel size (ksize)
# filter size (fsize)
# pooling layer size (poolsize)
# Then give the number of intermediate dense layers (MLP)
# Along with the number of neurons at the end
#net.custom_net(nCNN=1,ksize=[16],fsize=[10],poolsize=[4],nDense=1,densesize=[32])

# Run the training
results=ur.Results(SNRtest=8) # This file will contain the results of the training
trainer.train(net,results=results)

# Save the results
net.save()
results.saveResults()
