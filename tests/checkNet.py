'''
Opening the result file and do some printout
'''

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
printer.savePrints('trainres')

