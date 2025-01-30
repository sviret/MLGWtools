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

fs=float(max(struct[1]))      # Sampling freq
nTot=int(len(struct[0]))   # Number of bands
listFe=struct[1]     # List of sampling freqs
listTtot=struct[0] # List of frame sizes
tTot=sum(listTtot)  # Total length of a block
nptsHF=int(tTot*fs)        # Size of a block in the original frame

# The number of data points for each band is retrieved

npts=0      # Number of points fed to the network for each block
for x in range(nTot):
    npts+=int(listTtot[x]*listFe[x])
step=npts # Default step between two blocks

print(fs,tTot,nptsHF)

# 3. Trained network is loaded, now load the data data

test=gd.Generator('nothing')
inputFrame=test.readFrame(args.framefile)
    
sample=inputFrame.getFrame()
#sample=inputFrame.getrawFrame()
injections=inputFrame.getTruth()


step=int(float(args.step)*fs) 
print(len(sample[0]),step)
nblocks=int(len(sample[0])/step) # Number of steps necessary to analyze the frame

output=[]
Frameref=[]

for j in range(nTot):
    Frameref.append([]) 

# 4. Loop over frames to perform the inference

for i in range(nblocks):

    if i%100==0:
        print("Dealing with step ",i," of ",nblocks)
    tmpfrm=[]
    finalFrame=[]
    Frameref[0]=sample[0][i*step:i*step+nptsHF] # The temporal realization at max sampling
    ref=Frameref[0]
    # ref contains the chunck of data we will provide to the network
    # we need to reformat it a bit 
     
    for j in range(nTot): # Resample the block (particularly true for multibanding)
                            
        #Pick up the number of samples in chunk j

        ndatapts=int(listTtot[j]*fs)
        nttt=len(ref)
            
        # Nt contains chunk j
        Nt=ref[-ndatapts:]
        # We remove the chunk from ref
        ref=ref[:nttt-ndatapts]
        # We resample Nt to the correct sampling frequency
        decimate=int(fs/listFe[j])
        Frameref[j]=Nt[::int(decimate)]
        # Add it to a temporary frame
        tmpfrm.append(npy.asarray(Frameref[j]))
    
    # resampledFrame is a 1D vector containing the data points at the correct 
    # sampling freq for ech chunk. This is what we will feed to the network
    resampledFrame=npy.concatenate(tmpfrm)
    # Some formatting steps for TFlow compliance
    finalFrame.append(resampledFrame)
    fFrame=npy.asarray(finalFrame)
    data=npy.array(fFrame.reshape(1,-1,1),dtype=npy.float32)

    if data.shape[1]<npts: # Safety cut at the end
        break
        
    cut_top = 0
    cut_bottom = 0
    list_inputs_val=[]
                    
    for k in range(nTot):
        cut_top += int(listTtot[k]*fs)
        list_inputs_val.append(data[:,cut_bottom:cut_top,:])
        cut_bottom = cut_top
        
    # We feed the network
    res = usoftmax_f(model.predict(list_inputs_val,verbose=0))

    # Recover the probability to get a signal
    out=tf.keras.backend.get_value(res.T[1])[0]
    output.append(out)
        
finalres=npy.array(output)

# Inference is completed, output of the network is stored in vector output
# Save the results

f=open("network_output.p", mode='wb')        
pickle.dump([finalres,injections,nptsHF,step],f)
f.close()
 

plot1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
plot2 = plt.subplot2grid((3, 1), (2, 0))
    
plot1.plot(npy.arange(len(sample[0]))/fs, sample[0],'.')   # Input data
plot2.plot(npy.arange(len(finalres))*step/fs, finalres,'.')  # Network output
plt.tight_layout()
plt.show()