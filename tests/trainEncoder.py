'''
Macro showing the training of the CNN-LSTM standard autoencoder
'''

from MLGWtools.networks import utils as tn
from MLGWtools.networks import LSTM as encoder
from MLGWtools.utils import useResults as ur
import argparse

def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument("param",help="Fichier csv contenant les options du job",default='networks/trainings/encoder_training.csv')
    args = parser.parse_args()
    return args


'''
The main training macro starts here
'''

# Initialisation
args = parse_cmd_line()
trainer=tn.trutils(paramFile=args.param)
net=encoder.CNN_LSTM(trainer)

# Instantiate the network architecture

# The main network of the Chatterjee et al paper
# https://arxiv.org/abs/2105.03073

net.LSTM_denoiser(LSTM_layers=2,LSTM_neurons=50)

# Note that you can modify you can define LSTM decoding architecture
# LSTM_layers : number of bidirectional LSTM layers (3 in the paper)
# LSTM_neurons: number of neurons per LSTM cell  (100 in the paper)

# Run the training
results=ur.Results(SNRtest=10) # This file will contain the results of the training (n)ot used for the moment
trainer.train_denoiser(net,results=results)

# Save the results
net.save()
#results.saveResults()
