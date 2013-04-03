import numpy as np
import pickle
import multiprocessing
import heapq
import doc_processer
reload(doc_processer)
import time

n_docs_total = 2153744

# filenames
fns = ['data/tfidf_n12_%02d.pickle'%i for i in range(11)]

# load top_nbrs_diag
manager = multiprocessing.Manager()
top_nbrs = manager.list()

n_line_want = n_docs_total
i_line = 0
fin = open('data/top_nbrs_diag')

time_s = time.time()
for line in fin:
    sp = (line.strip()).split(',')
    len_sp = len(sp)
    if (len_sp==0) or (len_sp%2 != 0):
        print('Sth wrong at line %d'%i_line)        

    L = [(np.double(sp[i]),int(sp[i+1])) for i in range(0,len_sp,2)]
    heapq.heapify(L)

    top_nbrs.append(L)
    i_line+=1

    if (n_line_want is not None) and (i_line >= n_line_want):
        break
fin.close()

print('Done loading top_nbr: %f'%(time.time()-time_s))


inc_n_doc = 200000
for i in range(len(fns)):
    fin = open(fns[i])
    dataM1 = pickle.load(fin)
    fin.close()
    r_offset = i*inc_n_doc

    for j in range(len(fns)):
        if i==j:
            continue

        fin = open(fns[j])
        dataM2 = pickle.load(fin)
        fin.close()
        c_offset = j*inc_n_doc

        doc_processer.offdiag_block_launcher(top_nbrs,dataM1,dataM2,
                      r_offset,c_offset,batch_size=500,n_thread=3)
        break

    break


# output the top_nbrs
'''
fout = open('data/top_nbrs_all','w')
for i in range(n_docs_total):
    L = top_nbrs[i]
    fout.write(','.join(['%8f,%i'%(pair[0],pair[1]) for pair in L])+'\n')
fout.close()
'''
