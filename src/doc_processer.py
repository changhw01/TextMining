import numpy as np
from scipy.linalg import norm
from scipy.sparse import dok_matrix
from sklearn.metrics.pairwise import linear_kernel
import threading
import time
import pickle

class LKComputer(threading.Thread):

    def __init__(self,dataM,doc_id_list,tgt_range,fn_out,topN=200,batch_size=1000):

        ''' tgt_range is in the format of usual python indexing. 
            That is, starts from 0 and not inclusive the end.
        '''
        threading.Thread.__init__(self)
        self.dataM = dataM
        self.doc_id_list = doc_id_list
        self.tgt_range = tgt_range
        self.fn_out = fn_out
        self.topN = topN
        self.batch_size = batch_size
    
    def run(self):
        fout = open(self.fn_out,'w')

        n_doc_work = self.tgt_range[1]-self.tgt_range[0]
        n_batch = int(np.ceil(float(n_doc_work)/self.batch_size))

        for i in range(n_batch):
            work_range = [self.tgt_range[0]+i*self.batch_size, self.tgt_range[0]+(i+1)*self.batch_size]
            print(work_range)

            if work_range[1]>self.tgt_range[1]:
                work_range[1] = self.tgt_range[1]
        
            crossD = 1-linear_kernel(self.dataM[work_range[0]:work_range[1]],self.dataM)
            for j in range(crossD.shape[0]):
                I_sorting = np.argsort(crossD[j,:])
                outdata = []
                for k in range(1,self.topN+1):
                    idx = I_sorting[k]
                    outdata.append('%s,%f'%(self.doc_id_list[idx],crossD[j,idx]))
                fout.write(self.doc_id_list[work_range[0]+j]+'\t'+','.join(outdata)+'\n')

        fout.close()



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

def txt2pickle(fn,n_line,vocabulary):
    dataM = load_tfidf(fn,n_line=n_line,vocabulary=6423670,verbose=False)
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
