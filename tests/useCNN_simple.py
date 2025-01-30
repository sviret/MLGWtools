'''
Macro showing the usage of a CNN in simple case (one frequency band)
'''

import numpy as npy
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def usoftmax_f(X):
  diff=tnp.diff(X,1)
  X2=tnp.concatenate((1./(1.+tnp.exp(diff)),1./(1.+tnp.exp(-diff))),axis=1)
  return X2

'''
Command line parser
'''

def parse_cmd_line():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--framefile","-f", help="Fichier pickle contenant les données à étudier",default=None)
    parser.add_argument("--network","-n", help="Fichier pickle contenant le reseau entrainé",default=None)
    parser.add_argument("--step","-s", help="Temps entre chaque inférence",default=None)
    args = parser.parse_args()
    return args

'''
The main training macro starts here
'''

from MLGWtools.generators import generator as gd

    
# 1. Start by parsing the input options
args = parse_cmd_line()
                
# 2. Then retrieve the network and initialize it
f=open(args.network,mode='rb')
net=pickle.load(f)
f.close()
struct=net[0]
model=net[1]

fs=float(max(struct[1]))   # Sampling freq
nTot=int(len(struct[0]))   # Number of bands
listFe=struct[1]           # List of sampling freqs
listTtot=struct[0]         # List of frame sizes
tTot=sum(listTtot)         # Total length of a block
nptsHF=int(tTot*fs)        # Size of a block in the original frame

#
# Below we consider the simple case of single band: nTot=1
# See useCNN.py for the general case
#


npts=int(listTtot[0]*listFe[0]) # Number of points fed to the network

step=int(npts/2.) # Default step between two inferences is half this size

# Retrieve the parameter
step=int(float(args.step)*fs)

# 3. Trained network is loaded, now retrieve the data

test=gd.Generator('nothing')
inputFrame=test.readFrame(args.framefile)
    
sample=inputFrame.getFrame()
input=sample[0]                   # Input data strain
injections=inputFrame.getTruth()  # Injections


ninf=int(len(sample[0])/step)  # How many inferences we will do? 

output_sig=npy.zeros(ninf)   # The output of the signal category before activation
output_noise=npy.zeros(ninf) # The output of the noise category before activation
squashed_sig=npy.zeros(ninf) # The output of the signal category after activation

# 4. Loop over subframes to perform the inference

# In order to increase the speed we will pass batch of subframes to the model
batch_size=5000 

for i in range(0,ninf,batch_size):

    if i+batch_size>ninf:
        batch_size=int(ninf%batch_size-1) # The last batch can be smaller
    if i%10000==0:
        print("Dealing with inference ",i," of ",ninf)

    data=[]
    # create the batch
    for j in range(batch_size):
        data.append(input[(i+j)*step:(i+j)*step+nptsHF])
        if len(data[-1])!=nptsHF: # We are at the end of the frame, stop
            data.pop()
            break

    data=npy.asarray(data)            
    # We feed the network and recover the raw net output for this batch
    res  = model.predict(data.reshape(len(data),-1,1),verbose=0)
    sres = usoftmax_f(res) # Normalize it

    # Transform the keras tensor in human readable info
    output_sig[i:i+len(data)]=tf.keras.backend.get_value(res.T[1])[:]
    output_noise[i:i+len(data)]=tf.keras.backend.get_value(res.T[0])[:]
    squashed_sig[i:i+len(data)]=tf.keras.backend.get_value(sres.T[1])[:]

# Inference is completed, output of the network is stored in vectors 
# Save the results

f=open("network_output.p", mode='wb')        
pickle.dump([squashed_sig,injections,nptsHF,step,output_sig,output_noise],f)
f.close()

plt.plot(npy.arange(len(output_sig))*step/fs+nptsHF/(2*fs), output_sig,'.')  # Network output
plt.plot(npy.arange(len(output_sig))*step/fs+nptsHF/(2*fs), squashed_sig,'.')  # Network output
plt.plot(npy.arange(len(output_noise))*step/fs+nptsHF/(2*fs), output_noise,'.')  # Network output
plt.show()
