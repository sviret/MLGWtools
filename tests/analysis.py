import numpy as npy
import matplotlib.pyplot as plt
import pickle
import argparse
import sys

'''
Command line parser
'''
def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument("param",help="Fichier pickle contenant les results",default=None)

    args = parser.parse_args()
    return args

'''
The main training macro starts here
'''


    
# 1. Start by parsing the input options
args = parse_cmd_line()
                
# 2. Then retrieve the network and initialize it
f=open(args.param,mode='rb')
results=pickle.load(f)
f.close()

output=results[0]
injections=results[1]
nptsHF=results[2]
step=results[3]
fs=2048
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
    if (netres>0.9999): # Interesting value, does it belong to a cluster
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
matched_inj=[] 
# Look at which injections have led to a cluster
for inj in injections:
    
    inj.append(0)
    tcoal=inj[4]
        
    # Check if it's into one cluster
    #print("Looking for an injection at t=",tcoal)
    idx_clus=0
    for clust in highSigs:
        
        tstart=(clust[0]*step+nptsHF)*(1/fs)
        tend=(clust[len(clust)-1]*step+nptsHF)*(1/fs)
        
        #if (len(clust)>10):
       

        if (tcoal>tstart-0.5 and tcoal<tend+0.5): # Injection is in the cluster
            #print("It's a match")
            #print(tcoal,clust_vals[idx_clus])
            found+=1
            inj[5]=len(clust)
            clust_truth[idx_clus]=idx_inj
            inj.append(clust_vals[idx_clus])
            matched_inj.append(inj)
            break
        idx_clus+=1
    idx_inj+=1
    if inj[5]==0:
        print("Missed !",inj)

idx_clus=0
for clus in clust_truth:
    #print(clus,highSigs[idx_clus])
    idx_clus+=1

snr=[]
width=[]
dt=[]
mchirp=[]

for inj in matched_inj:
    snr.append(inj[3])
    width.append(inj[5])
    dt.append(inj[4]-inj[6][2])
    M=inj[1]+inj[2]                        # Total mass
    eta=inj[1]*inj[2]/(M**2)        # Reduced mass
    mchirp.append(npy.power(eta,3./5.)*M) 


print("Found",found,"injections out of",len(injections))
print("Among",len(highSigs),"clusters in total")



plt.figure(figsize=(10,5))
plt.grid(True, which="both", ls="-")
plt.plot(snr,width,'.')
plt.xlabel('SNR')
plt.ylabel('Cluster size')
plt.legend()       
plt.show()
 
plt.figure(figsize=(10,5))
plt.grid(True, which="both", ls="-")
plt.plot(mchirp,width,'.')
plt.xlabel('Mchirp')
plt.ylabel('Cluster size')
plt.legend()       
plt.show()

plt.figure(figsize=(10,5))
plt.hist(dt)
#plt.xlabel('SNR')
#plt.ylabel('Cluster size')
#plt.legend()       
plt.show()

