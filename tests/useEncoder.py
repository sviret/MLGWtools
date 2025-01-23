import numpy as npy
import matplotlib.pyplot as plt
import pickle

'''
Command line parser
'''
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


def split_sequence(array, n_steps):
    ## split a univariate sequence into samples
    # Dimension of input array: [nsample][npts][1]
    # Dimension of output array: [nsample][npts][n_steps]

    splitted=[]
    for data in array:
        # Zero padding
        seq = npy.concatenate((npy.zeros(int(n_steps/2)), data.reshape(-1), npy.zeros(int(n_steps/2))))
        # Splitting
        ssequence = npy.array([npy.array(seq[i:i+n_steps]) for i in range(len(seq)-n_steps)])
        splitted.append(ssequence)

    final=npy.asarray(splitted)
    return npy.expand_dims(final,axis=-1)

'''
The main training macro starts here
'''



def main():
    from MLGWtools.generators import generator as gd
    from tqdm import tqdm
    
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

    print(model.summary())

    feature=4
    step=int(0.2*fs) 

    limit=int(0.05*nptsHF) # Don't account for the first points

    # 3. Trained network is loaded, one can use it on data

    test=gd.Generator('nothing')
    inputFrame=test.readFrame(args.framefile)
        
    sample=inputFrame.getFrame()

    nblocks=int(len(sample[0])/step) # Number of steps to analyze the frame
    weight_sharing=npy.array(sample[1],dtype=npy.float32)
    output=[]
    output_S=[]
    Frameref=[]
    Truthref=[]
    idxref=[]

    for j in range(nTot):
        Frameref.append([])
        Truthref.append([])

    # 4. Loop over frames to perform the inference
    nresults=npy.zeros(len(sample[0]))
    finaloutput=npy.zeros(len(sample[0]))
    finalpure=npy.zeros(len(sample[0]))

    for i in tqdm(range(nblocks)):

        tmpfrm=[]
        finalFrame=[]
        finalTruth=[]
        tmptru=[]
        time=npy.arange(i*step,i*step+nptsHF)
        nresults[i*step+limit:i*step+nptsHF-limit]+=1
        Frameref[0]=sample[0][i*step:i*step+nptsHF] # The temporal realization at max sampling
        Truthref[0]=sample[2][i*step:i*step+nptsHF] # The truth (if available)
        ref=Frameref[0]
        truth=Truthref[0]

        #print(ref.shape,truth.shape)

        for j in range(nTot): # Resample the block
                            
            #Pick up the data chunk
            ndatapts=int(listTtot[j]*fs)
            nttt=len(ref)
            decimate=int(fs/listFe[j])
                       
            Nt=ref[-ndatapts:]
            ref=ref[:nttt-ndatapts]
            Frameref[j]=Nt[::int(decimate)]
            tmpfrm.append(npy.asarray(Frameref[j]))

            Nt=truth[-ndatapts:]
            truth=truth[:nttt-ndatapts]
            Truthref[j]=Nt[::int(decimate)]
            tmptru.append(npy.asarray(Truthref[j]))

        resampledFrame=npy.concatenate(tmpfrm)
        finalFrame.append(resampledFrame)
        fFrame=npy.asarray(finalFrame)

        resampledTruth=npy.concatenate(tmptru)
        finalTruth.append(resampledTruth)
        fTruth=npy.asarray(finalTruth)
        
        data=npy.array(fFrame.reshape(1,-1,1),dtype=npy.float32)
        truth=npy.array(fTruth.reshape(1,-1,1),dtype=npy.float32)

        if data.shape[1]<npts: # Safety cut at the end
            break
        
        data=split_sequence(data, feature)

        TestSet=(data,truth)
                
        cut_top = 0
        cut_bottom = 0
        list_inputs_val=[]
                    
        for k in range(nTot):
            cut_top += int(listTtot[k]*fs)
            list_inputs_val.append(data[:,cut_bottom:cut_top,:])
            cut_bottom = cut_top
        
        res = model.predict(list_inputs_val,verbose=0)
        res=npy.squeeze(res)

        idxref.append(time)
        finaloutput[i*step+limit:i*step+nptsHF-limit]+=res[limit:-limit]
        finalpure[i*step+limit:i*step+nptsHF-limit]+=npy.squeeze(fTruth)[limit:-limit]
        output.append(res)
        output_S.append(truth)
        
    finaloutput=finaloutput/nresults
    finalpure=finalpure/nresults

    finalres=npy.array(finaloutput).flatten()
    finalres_S=npy.array(finalpure).flatten()
    print(len(finalres),len(finalres_S))

    f=open("encoder_output.p", mode='wb')        
    pickle.dump([sample[0],finalres,finalres_S,nptsHF,step,fs],f)
    f.close()
    plot1 = plt.subplot2grid((4, 1), (0, 0), rowspan = 2)
    plot2 = plt.subplot2grid((4, 1), (2, 0))
    plot3 = plt.subplot2grid((4, 1), (3, 0))
    
    plot1.plot(npy.arange(len(sample[0]))/fs, sample[0],'.')
    plot2.plot(npy.arange(len(finalres))/fs, finalres,'.')
    plot3.plot(npy.arange(len(finalres))/fs, finalres_S,'.')
    plt.tight_layout()
    plt.show()
    plt.savefig('encode_output.png')

    
############################################################################################################################################################################################
if __name__ == "__main__":
    main()
