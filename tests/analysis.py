'''
Macro analyzing the output of useCNN 
'''

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
    parser.add_argument("param",help="Name of the output pickle file of useCNN",default=None)

    args = parser.parse_args()
    return args


'''
The main analysis macro starts here
'''


# 1. Start by parsing the input options
args = parse_cmd_line()
                
# 2. Then retrieve the network output
f=open(args.param,mode='rb')
results=pickle.load(f)
f.close()

output=results[0]
output_free=results[4]-results[5]
#output_free=results[4]
injections=results[1]
nptsHF=results[2]
step=results[3]
fs=2048
# Inference is done, output of the network is stored in vector output
         
i=0
highSigs=[]

# Loop over the output to form clusters

for netres in output_free:

    #
    # Look if this value can belong to a cluster
    #

    nclust=len(highSigs)
    missflag=0
    if (netres>1. and output[i]>0.999 ): # Interesting value, does it belong to a cluster
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
highSigs_filt=[]
for clust in highSigs:
    if len(clust)>=5:
        highSigs_filt.append(clust)
        

for clust in highSigs_filt:
    
    clust_truth.append(-1)
        
    clust_val=0
    clust_wgh=0
    clust_cen=0
    clust_sd=0
        
    for val in clust:
        res=output[val]
        wght=output_free[val]
        clust_wgh+=float(wght)
        clust_val+=float(res)
        clust_cen+=float(val*wght)
    clust_val/=len(clust)
    clust_cen/=(clust_wgh)
    
    for val in clust:
        res=float(output[val])
        clust_sd+=(res-clust_val)*(res-clust_val)
            
    clust_sd=npy.sqrt(clust_sd/len(clust))
    clust_vals.append([clust_wgh,clust_val,clust_cen,(clust_cen*step+nptsHF)*(1/fs),clust_sd,len(clust)])
    #print(clust_vals[-1:])

# Now establish the correspondence between clusters and injection
#sys.exit()
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
    for clust in highSigs_filt:
        
        tstart=(clust[0]*step+nptsHF)*(1/fs)
        tend=(clust[len(clust)-1]*step+nptsHF)*(1/fs)
        

        if (tcoal>tstart-1. and tcoal<tend+1.): # Injection is in the cluster
            #print("It's a match")
            #print(tcoal,clust_vals[idx_clus])
            found+=1
            inj[5]=len(clust)
            clust_truth[idx_clus]=idx_inj
            inj.append(clust_vals[idx_clus])
            matched_inj.append(inj)
            #break
        idx_clus+=1
    idx_inj+=1
    if inj[5]==0:
        print("Missed !",inj)

idx_clus=0
width_n=[]
weight_n=[]
sigma_n=[]
for clus in clust_truth:
    if clus==-1 and clust_vals[idx_clus][0]>1000.:
        weight_n.append(clust_vals[idx_clus][0])
        width_n.append(clust_vals[idx_clus][5])
        sigma_n.append(clust_vals[idx_clus][0]/clust_vals[idx_clus][5])
        print(clust_vals[idx_clus])
    idx_clus+=1

snr=[]
width=[]
weight=[]
dt=[]
mchirp=[]
sigma=[]

for inj in matched_inj:
    snr.append(inj[3])
    width.append(inj[5])
    weight.append(inj[6][0])
    sigma.append(inj[6][0]/inj[6][5])
    dt.append(inj[4]-inj[6][3])
    M=inj[1]+inj[2]                        # Total mass
    eta=inj[1]*inj[2]/(M**2)        # Reduced mass
    mchirp.append(npy.power(eta,3./5.)*M) 
    print(inj)


print("Found",found,"injections out of",len(injections))
print("Among",len(highSigs_filt),"clusters in total")


plt.plot(npy.arange(len(results[4]))*step/fs+nptsHF/(2*fs), results[4],'.')  # Network output
plt.plot(npy.arange(len(results[5]))*step/fs+nptsHF/(2*fs), results[5],'.')  # Network output

plt.show()

plt.figure(figsize=(10,5))
plt.grid(True, which="both", ls="-")
plt.plot(snr,weight,'.')
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
plt.grid(True, which="both", ls="-")
plt.plot(weight,sigma,'.')
plt.plot(weight_n,sigma_n,'.')
plt.xlabel('Mchirp')
plt.ylabel('Cluster size')
plt.legend()       
plt.show()

plt.figure(figsize=(10,5))
plt.hist(sigma_n, histtype='step',bins=100)
plt.hist(sigma, histtype='step',bins=100)
#plt.xlabel('SNR')
#plt.ylabel('Cluster size')
#plt.legend()       
plt.show()

