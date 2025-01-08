import numpy as npy
import matplotlib.pyplot as plt
import pickle
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
    parser.add_argument("--network","-n", help="Fichier h5 contenant le reseau entrainé",default=None)
    
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

npts=net.getNetSize()      # Number of points fed to the network for each block
step=int(net.getStepSize())# Step between two blocks
fs=float(net.getfs())      # Sampling freq
nTot=int(net.getNband())   # Number of bands
listFe=net.getListFe()     # List of sampling freqs
listTtot=net.getListTtot() # List of frame sizes
tTot=net.getBlockLength()  # Total length of a block
nptsHF=int(tTot*fs)        # Size of a block in the original frame

step=int(0.1*fs) 


# 3. Trained network is loaded, one can use it on data

frame_cfg = "MLGWtools/tests/samples/Frame.csv"

test=gd.Generator(paramFile=frame_cfg)

inputFrame=test.readFrame(args.framefile)
    
sample=inputFrame.getFrame()
injections=inputFrame.getTruth()
nblocks=int(len(sample[0])/step) # Number of steps to analyze the frame
weight_sharing=npy.array(sample[1],dtype=npy.float32)
output=[]
output_S=[]
Frameref=[]

for j in range(nTot):
    Frameref.append([])

# 4. Loop over frames to perform the inference

for i in range(nblocks):

    tmpfrm=[]
    finalFrame=[]
    Frameref[0]=sample[0][i*step:i*step+nptsHF] # The temporal realization at max sampling
    ref=Frameref[0]
                         
    for j in range(nTot): # Resample the block
                            
        #Pick up the data chunk
        ndatapts=int(listTtot[j]*fs)
        nttt=len(ref)
            
        Nt=ref[-ndatapts:]
        ref=ref[:nttt-ndatapts]
        decimate=int(fs/listFe[j])
            
        Frameref[j]=Nt[::int(decimate)]
        tmpfrm.append(npy.asarray(Frameref[j]))
                        
    resampledFrame=npy.concatenate(tmpfrm)
    finalFrame.append(resampledFrame)
    fFrame=npy.asarray(finalFrame)
    
    data=npy.array(fFrame.reshape(1,-1,1),dtype=npy.float32)

    if data.shape[1]<npts: # Safety cut at the end
        break

    TestSet=(data,weight_sharing)
        
    #res=usoftmax_f(net(TestSet[0].as_in_ctx(device),TestSet[1].as_in_ctx(device))).asnumpy()
        
    cut_top = 0
    cut_bottom = 0
    list_inputs_val=[]
                    
    for k in range(nTot):
        cut_top += int(listTtot[k]*fs)
        list_inputs_val.append(data[:,cut_bottom:cut_top,:])
        cut_bottom = cut_top
        #print(cut_bottom,cut_top,)
        
    res = usoftmax_f(net.model.predict(list_inputs_val,verbose=0))
    #print((i*step+nptsHF)*(1/fs),net.model.predict(list_inputs_val,verbose=0))

    out=tf.keras.backend.get_value(res.T[1])[0]

    output.append(out)
    if (out>0.999):
        output_S.append(out)
        t_hit=(i*step+nptsHF)*(1/fs)
        #print("Potential signal at t=",t_hit,out)
        #for inj in injections:
        #    if npy.abs(inj[4]-t_hit)<1.:
        #        print("!Match injection:",inj)
        #        break
    else:
        output_S.append(0.)
        
#print(net.model.summary())
finalres=npy.array(output)
finalres_S=npy.array(output_S)

# Inference is done, output of the network is stored in vector output
         
i=0
highSigs=[]

# Loop over the output to form clusters

for netres in output:

    #
    # Look if this value can belong to a cluster
    #

    nclust=len(highSigs)
    missflag=0
    if (netres>0.995): # Interesting value, does it belong to a cluster
        if nclust==0: # No cluster yet, create one
            highSigs.append([i])
        else:         # Clusters exist, check is we are in or not
            curr_clust=highSigs[nclust-1]
            sclust=len(curr_clust)
                
            # End of the last cluster is the previous block, we add the new hit to the cluster
            if (i-curr_clust[sclust-1]==1):
                curr_clust.append(i)
                highSigs.pop()
                highSigs.append(curr_clust)
            # End of the last cluster is the next to previous block, we add the new hit to the cluster
            # As we accept one missed hit (just one)
            elif (i-curr_clust[sclust-1]==2 and missflag==0):
                #curr_clust.append(i-1)
                curr_clust.append(i)
                highSigs.pop()
                highSigs.append(curr_clust)
                missflag=1
            # Last cluster is too far away, create a new cluster
            else:
                if sclust==1:
                    highSigs.pop() # The last recorded cluster was one block only, remove the artefact
                highSigs.append([i]) # New cluster
                missflag=0
    i+=1
                    
# End of cluster building stage    
nclust=len(highSigs)
if nclust==0:
    print("No clus!!")
    sys.exit()
    
# Remove the last cluster if only one block long
if len(highSigs[len(highSigs)-1])==1:
    highSigs.pop()

# Now determine the cluster coordinates
#
# Center, average network output value, sigma of this average

clust_truth=[]  # Clus is matched to an injections
clust_vals=[]   # Cluster properties
    
for clust in highSigs:

    clust_truth.append(-1)
        
    clust_val=0
    clust_cen=0
    clust_sd=0
        
    for val in clust:
        res=output[val]
        clust_val+=float(res)
        clust_cen+=float(val)
    clust_val/=len(clust)
    clust_cen/=len(clust)
    
    for val in clust:
        res=float(output[val])
        clust_sd+=(res-clust_val)*(res-clust_val)
            
    clust_sd=npy.sqrt(clust_sd/len(clust))
    clust_vals.append([clust_val,clust_cen,(clust_cen*step+nptsHF)*(1/fs),clust_sd,len(clust)])


# Now establish the correspondence between clusters and injection

found=0
idx_inj=0
    
# Look at which injections have led to a cluster
for inj in injections:
    
    inj.append(0)
    tcoal=inj[4]
        
    # Check if it's into one cluster
   
    idx_clus=0
    for clust in highSigs:
        
        tstart=(clust[0]*step+nptsHF)*(1/fs)
        tend=(clust[len(clust)-1]*step+nptsHF)*(1/fs)
        
        #if (len(clust)>10):
        #    print(tcoal,tstart,tend,len(clust))

        if (tcoal>tstart-0.1 and tcoal<tend): # Injection is in the cluster
            found+=1
            inj[5]=len(clust)
            clust_truth[idx_clus]=idx_inj
            inj.append(clust_vals[idx_clus])
            break
        idx_clus+=1
    idx_inj+=1
    print(inj)

idx_clus=0
for clus in clust_truth:
    print(clus,highSigs[idx_clus])
    idx_clus+=1


print("Found",found,"injections out of",len(injections))
print("Among",len(highSigs),"clusters in total")




plot1 = plt.subplot2grid((3, 1), (0, 0), rowspan = 2)
plot2 = plt.subplot2grid((3, 1), (2, 0))
    
plot1.plot(npy.arange(len(sample[0]))/fs, sample[0],'.')
plot1.plot(npy.arange(len(sample[0]))/fs, sample[2],'.')
plot2.plot(npy.arange(len(finalres))*step/fs, finalres,'.')
plot2.plot(npy.arange(len(finalres))*step/fs, finalres_S,'.')
plt.tight_layout()
plt.show()
 


