import time
import pickle
import heapq
import math
import multiprocessing

import numpy as np
from scipy.linalg import norm
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import linear_kernel

def thread_cos_diag(top_nbrs,dataM,job_ranges,c_offset,
                    n_nbr=100,verbose=False):
    
    ''' (cos,idx) 
        Note in the min-heap, the first one is the smallest.
    '''

    for job_bd in job_ranges:
        crossV = linear_kernel(dataM[job_bd[0]:job_bd[1],:],dataM)
        n_doc1, n_doc2 = crossV.shape
        
        for i_doc in range(n_doc1):
            i_offset = i_doc + job_bd[0]
            L = top_nbrs[i_offset]
            for j in range(n_doc2):            
                if i_offset == j+c_offset:
                    continue

                if len(L)<n_nbr:
                    heapq.heappush(L, (crossV[i_doc,j],j+c_offset))
                elif crossV[i_doc,j] > L[0][0]:
                    heapq.heapreplace(L, (crossV[i_doc,j],j+c_offset))
        
            top_nbrs[i_offset] = L

        if verbose:
            print('process range (%d,%d)'%(job_bd[0],job_bd[1]))

def thread_diag_block(top_nbrs,dataM,job_ranges,r_offset, c_offset,
                    n_nbr=100,verbose=False):
    
    ''' (cos,idx) 
        Note in the min-heap, the first one is the smallest.
    '''

    for job_bd in job_ranges:
        crossV = linear_kernel(dataM[job_bd[0]:job_bd[1],:],dataM)
        n_doc1, n_doc2 = crossV.shape
        
        for i_doc in range(n_doc1):
            i_offset = i_doc + job_bd[0] + r_offset
            L = top_nbrs[i_offset]
            for j in range(n_doc2):            
                if i_offset == j+c_offset:
                    continue

                if len(L)<n_nbr:
                    heapq.heappush(L, (crossV[i_doc,j],j+c_offset))
                elif crossV[i_doc,j] > L[0][0]:
                    heapq.heapreplace(L, (crossV[i_doc,j],j+c_offset))
        
            top_nbrs[i_offset] = L

        if verbose:
            print('process range (%d,%d)'%(job_bd[0],job_bd[1]))

def diag_block_launcher(top_nbrs, dataM,r_offset, c_offset,n_nbr=100,batch_size=500,n_thread=4):


    n_doc = dataM.shape[0]
  
    # prepare the nbr data structure
    '''
    manager = multiprocessing.Manager()
    top_nbrs = manager.list()
    for i in range(n_doc):
        top_nbrs.append([])
    '''

    # prepare the job ranges
    job_queue = [[i*batch_size,(i+1)*batch_size] \
                 for i in range(int(math.ceil(float(n_doc)/batch_size)))]

    if job_queue[len(job_queue)-1][1]>n_doc:
        job_queue[len(job_queue)-1][1] = n_doc    

    n_job = len(job_queue)
    n_job_per_thread = int(math.ceil(float(n_job)/n_thread))
    job_collections = []
    for i in range(n_thread):
        if (i+1)*n_job_per_thread < n_job:
            job_collections.append(job_queue[i*n_job_per_thread:(i+1)*n_job_per_thread])
        else:
            job_collections.append(job_queue[i*n_job_per_thread:n_job])
    
    # start the work
    time_s = time.time()
    workers = []
    for i in range(n_thread):
        workers.append(multiprocessing.Process(target=thread_diag_block,
            args=(top_nbrs, dataM, job_collections[i],r_offset,c_offset,100,True)))
        print('Prepare to start worker %d'%i)

    for worker in workers:
        worker.start()        

    for worker in workers:
        worker.join()
    print('time spent: %f sec'%(time.time()-time_s))

def compute_tfidf(fn_in,fn_out,n_doc,vocabulary,normalize=True):

    # Get the vobaculary set and indexing
    if isinstance(vocabulary,str):
        voc = load_voc(vocabulary,include_df=True)
    elif isinstance(vocabulary,dict):
        voc = vocabulary
    else:
        print('Please provide the vocabulary indexing')
        return None

    n_doc = np.double(n_doc)

    n_line_read = 0
    fin = open(fn_in)
    fout = open(fn_out,'w')
    for line in fin:
        words = []
        tfidf = []
        
        segments = (line.strip()).split('\t')
        sp = segments[1].split(',')

        for i in range(0,len(sp),2):
            if sp[i] in voc:
                words.append(voc[sp[i]][0])
                tfidf.append( (np.double(sp[i+1])) \
                             *(np.log(n_doc/np.double(voc[sp[i]][1]))) )

        tfidf = np.array(tfidf)
        if normalize:
            tfidf = tfidf/norm(tfidf)

        fout.write(segments[0] + '\t' + \
        ','.join(['%d,%f'%(words[i],tfidf[i]) for i in range(len(words))]) +'\n')

        n_line_read += 1
        if n_line_read%2000 == 0:
            print('processed %d lines'%n_line_read)
        
    fin.close()
    fout.close()


def load_tfidf(fn_in,line_start=1,line_end=None,
             n_line=None,vocabulary=None,verbose=True):
    ''' 
        Load doc from fn and store into a doc-term sparse matrix (in csr_matrix
        form). 
    
        If vocabulary is a number, assume the words in the input file are
        indices and the number is the voc size.


        'line_start' and 'line_end' are the range of lines to read (inclusive). If
        'line_end' is None, read until end of the file. If 'n_line' and line_end 
        are both given, then the smaller n_line will be chosen.        

    '''
    # Get the vobaculary set and indexing
    if isinstance(vocabulary,str):
        voc = load_voc(vocabulary)
        size_voc = len(voc)
    elif isinstance(vocabulary,dict):
        voc = vocabulary
        size_voc = len(voc)
    elif isinstance(vocabulary,int):
        voc = None
        size_voc = vocabulary
    else:
        print('Please provide the vocabulary indexing')
        return None
    
    # Decide what lines to read
    if (line_end is None) and (n_line is None):
        # need to knwo number lines to read so that we can create the sparse
        # matrix
        print('Please provide the number of lines to read')
        return None
    elif line_end is not None:
        n_line_ask = line_end-line_start+1
        if (n_line is not None) and (n_line_ask > n_line):
            n_line_ask = n_line
    elif n_line is not None:
        n_line_ask = n_line

    # Skip lines if line_start is not 1
    if line_start > 1:
        n_line_skip = line_start-1
    else:
        n_line_skip = 0
        
    # Loading data
    dokM = dok_matrix((n_line_ask,size_voc),dtype=np.double)
    i_line = 0
    n_line_read = 0

    print('start processing data')

    fin = open(fn_in)
    for line in fin:
        # skip some lines as requested
        if i_line < n_line_skip:
            i_line += 1
            continue

        segments = line.split('\t')

        sp = segments[1].split(',')
        wid_counts = dict()
        for i in range(0,len(sp),2):
            if voc is None:
                wid_counts[int(sp[i])] = np.double(sp[i+1])
            elif (sp[i] in voc):
                wid_counts[voc[sp[i]]] = np.double(sp[i+1])
       
        dokM[n_line_read,wid_counts.keys()] = wid_counts.values()

        n_line_read += 1
        i_line += 1

        if n_line_read >= n_line_ask:
            break

        if verbose and (n_line_read%1000==0):            
            print('read %d lines'%n_line_read)
    
    if verbose:
        print('read total %d lines'%n_line_read)

    fin.close()
    #return (dokM.tocoo()).tocsr(),id_doc_list
    return dokM.tocsr()



#-------- Utility Functions ------------#

def txt2pickle(fn,n_line,vocabulary=None):

    if vocabulary is None:
        vocabulary = 6423670

    dataM = load_tfidf(fn,n_line=n_line,vocabulary=vocabulary,verbose=False)
    fout = open(fn+'.pickle','w')
    pickle.dump(dataM,fout)
    fout.close()
    print('convert %s to %s'%(fn,fn+'.pcikle'))

def load_voc(vocabulary,include_df=False):

    voc = None

    if isinstance(vocabulary,str):
        try:
            fin = open(vocabulary)
        except:
            print('failed to read the vocabulary set')
            return None

        voc = dict()
        idx = 0
        for line in fin:
            if not include_df:
                voc[(line.split())[0]] = idx
            else:
                sp = (line.strip()).split('\t')
                voc[sp[0]] = (idx,int(sp[1]))

            idx+=1
        fin.close()
        print('Read vocabulary indexing from %s'%vocabulary)

    # ToDo: convert a list to dict

    return voc
