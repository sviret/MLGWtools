'''
Macro showing how to train the Huerta and George network
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

args = parse_cmd_line()
trainer=tn.trutils(paramFile=args.param)
net=cnn.Multiple_CNN(trainer)
#net.huerta_legacy()
net.fpga_version()

results=ur.Results(SNRtest=8)
trainer.train(net,results=results)

net.save()
results.saveResults()
