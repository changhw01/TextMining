import pickle
import multiprocessing
import doc_processer
reload(doc_processer)
import time

n_docs_total = 2153744

# filenames
fns = ['data/tfidf_n12_%02d.pickle'%i for i in range(11)]

# prepare the nbr data structure
time_s = time.time()
manager = multiprocessing.Manager()
top_nbrs = manager.list()
for i in range(n_docs_total):
    top_nbrs.append([])
print('Done allocating top_nbr: %f'%(time.time()-time_s))


inc_n_doc = 200000
for i in range(len(fns)):
    fin = open(fns[i])
    dataM = pickle.load(fin)
    fin.close()

    r_offset = i*inc_n_doc
    c_offset = i*inc_n_doc
    doc_processer.diag_block_launcher(top_nbrs,dataM,r_offset,c_offset,batch_size=1000,n_thread=10)

# output the top_nbrs
fout = open('data/top_nbrs','w')
for i in range(n_docs_total):
    L = top_nbrs[i]
    fout.write(','.join(['%8f,%i'%(pair[0],pair[1]) for pair in L])+'\n')
fout.close()
