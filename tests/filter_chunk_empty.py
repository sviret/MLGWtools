import matplotlib.pyplot as pp
import scipy.fft
import numpy as npy
from MLGWtools.generators import generator as gd


'''
Command line parser
'''

def parse_cmd_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--framefile","-f", help="Fichier pickle contenant les données filtrer",default=None)
    parser.add_argument("-tmin", help="Starting time for filter",default=None)
    parser.add_argument("-tmax", help="Ending time for filter",default=None)
    parser.add_argument("-m1", help="m1 (in solar mass unit)",default=None)
    parser.add_argument("-m2", help="m2 (in solar mass unit)",default=None)
    args = parser.parse_args()
    return args

'''
The main training macro starts here
'''
    
# 1. Start by parsing the input options
args = parse_cmd_line()

# Retrieve the input data in the time range required
test=gd.Generator('nothing')
inputFrame=test.readFrame(args.framefile)
sample=inputFrame.getFrame()[0]
tin=float(args.tmin)
tend=float(args.tmax)

# Then the template properties  
template=gd.Generator(paramFile="MLGWtools/generators/samples/template.csv")
template.initTemplate_TD([tend-tin]) # Same time range
sig=template.getTemplate()
info=template.sigprop()
fs=template.sampleprop()[1][0]
sig.majParams(m1=float(args.m1),m2=float(args.m2))
sig.getNewSample()

# chunk is the bit of data to filter
chunk=sample[int(tin*fs):int(tend*fs)]

# tmpl is the pure template, whitened and normalized
tmpl=sig.signal()

# This can happen if we look at a chunk of data shorter than the template
if len(chunk)<len(tmpl):
    tmpl=tmpl[-len(chunk):]

# Now we can do the filtering
# tmpl and chunk contain the template and input data over the 
# studied period 
#
# You just have to complete this with the matched filter 

# 1 Pass in the frequency domain


# 2 Compute the match filter

# 3 Get the maximum and do some plots


pp.plot()
pp.ylabel('signal-to-noise ratio')
pp.xlabel('time (s)')
pp.show()