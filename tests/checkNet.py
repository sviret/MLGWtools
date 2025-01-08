'''
Macro showing how to train the Huerta and George network
'''

#from MLGWtools.networks import utils as tn
#from MLGWtools.networks import CNN as cnn
from MLGWtools.utils import useResults as ur
import argparse

def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument("param",help="Pickle file containing training results")

    args = parser.parse_args()
    return args


'''
The main testing macro starts here
'''

args = parse_cmd_line()
res = ur.Results.readResults(args.param)
printer=ur.Printer()
printer.plotResults(res)
printer.savePrints('bebert')

