import numpy as np
import pickle
import multiprocessing
import doc_processer
reload(doc_processer)
import time

def get_manager_data(fn):

    fin = open(fn)
    data = pickle.load(fin)
    fin.close()

    manager = multiprocessing.Manager()
    manager_list = manager.list(data)
    data = None
    return manager_list


n_docs_total = 2153744

# filenames
fns_data = ['data/tfidf_n12_%02d.pickle'%i for i in range(11)]
fns_nbrs_diag = ['data/top_nbrs_diag/top_nbrs_%02d.pickle'%i for i in range(11)]
fns_nbrs_out = ['data/top_nbrs_final_%02d.pickle'%i for i in range(11)]

inc_n_doc = 200000
for i in range(len(fns_data)):

    time_s = time.time()
    top_nbrs = get_manager_data(fns_nbrs_diag[i])
    print('Load top_nbrs %02d: %f secs'%(i,time.time()-time_s))

    time_s = time.time()
    fin = open(fns_data[i])
    dataM1 = pickle.load(fin)
    fin.close()
    r_offset = i*inc_n_doc
    print('Load data %02d: %f secs'%(i,time.time()-time_s))

    for j in range(len(fns_data)):
        if i==j:
            continue

        time_s = time.time()
        fin = open(fns_data[j])
        dataM2 = pickle.load(fin)
        fin.close()
        c_offset = j*inc_n_doc
        print('Load data %02d: %f secs'%(j,time.time()-time_s))
    
        time_s = time.time()
        doc_processer.offdiag_block_launcher(top_nbrs,dataM1,dataM2,
                      r_offset,c_offset,batch_size=500,n_thread=10,verbose=False)
        print('Update top_nbrs (%2d,%2d): %f secs'%(i,j,time.time()-time_s))

        break

    # dump top_nbrs back to the disc

    time_s = time.time()
    fout = open(fns_nbrs_out[i],'w')
    pickle.dump(list(top_nbrs),fout)
    fout.close()
    print('dump top_nbrs %s: %f secs'%(fns_nbrs_out[i],time.time()-time_s))
    break
