import time
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import HashingVectorizer

def extract_ngram(fn_in,fn_out,ngram_range=(1,1),stopwords=None,verbose=False,n_doc=None):
    
    '''
        Assume each line has the three parts: (1) id, (2) title, (3) content,
        and are seperated by tab.

        stop_word = string {'english'}, list, or None (default)

        Implementation Notes: 
        N-gram is just a n-length sliding window feature extraction from unigram
        in scikit-learn. Because we want to do stemming, we create unigrams and
        build other n-grams from them instead of using the function in sklearn.
    '''

    if isinstance(ngram_range,int):
        ngram_range = (ngram_range,ngram_range)

    if isinstance(stopwords,str):
        try:
            fin_sw = open(stopwords)
        except:
            print('Failed to open stop word file: %s'%stopwords)

        stopwords = [line.strip() for line in fin_sw]
        fin_sw.close()
        print('got stop word list from file')


    tokenizer_hash = HashingVectorizer(ngram_range=(1,1),
    stop_words=stopwords).build_analyzer()

    eng_stemmer = SnowballStemmer('english')

    idx_line = 0
    
    fin = open(fn_in)
    fout = open(fn_out,'w')

    t_global = time.time()
    t_local = time.time()
    for line in fin:

        sp = (line.strip()).split('\t')
        if len(sp)<3:
            print('Missing pmid, title or content: %d'%(idx_line+1))
            continue

        # for title    
        tf_local = {}
        unigrams = [eng_stemmer.stem(w) \
                   for w in tokenizer_hash(sp[1]) if not w.isdigit()]
        n_unigrams = len(unigrams)
       
        for n in range(ngram_range[0]-1,ngram_range[1]):
            for i in range(0,n_unigrams-n):
                tkn = ' '.join(unigrams[i:i+n+1])
                if tkn not in tf_local:
                    tf_local[tkn] = 1
                else:
                    tf_local[tkn] += 1

        title_output = ','.join([k+','+str(v) for k,v in tf_local.iteritems()])

        # for abstract
        tf_local = {}
        unigrams = [eng_stemmer.stem(w) \
                   for w in tokenizer_hash(sp[2]) if not w.isdigit()]
        n_unigrams = len(unigrams)
        
        for n in range(ngram_range[0]-1,ngram_range[1]):
            for i in range(0,n_unigrams-n):
                tkn = ' '.join(unigrams[i:i+n+1])
                if tkn not in tf_local:
                    tf_local[tkn] = 1
                else:
                    tf_local[tkn] += 1

        abs_output = ','.join([k+','+str(v) for k,v in tf_local.iteritems()])

        try:
            fout.write('\t'.join([sp[0],title_output,abs_output])+'\n')
        except:
            print('failed to write line %d'%idx_line)
            idx_line+=1
            continue

        idx_line+=1

        if verbose and (idx_line+1)%2000==0:
            t_now = time.time()
            print('Processed %d lines, time %f secs'%(idx_line+1,t_now-t_local))
            t_local = t_now

        if (n_doc is not None) and (n_doc<=idx_line):
            break

    fin.close()
    fout.close()

    if verbose:
        print('Total time span: %f secs'%(time.time()-t_global))

def merge_title_content(fn_in,fn_out,title_weight=1,ngram_range=None):
    ''' 
        The output of the tokenization function seperate the title and
        content(abstract). The function will merge them and weight the title
        differently.

        Also, we will consider only n-grams within the given range (inclusive).

        Here we assume title weight is int for storage concern.
    '''

    fin = open(fn_in)
    fout = open(fn_out,'w')

    i_line = 0
    for line in fin:
        segment = (line.strip()).split('\t')
        if len(segment)<3:
            print('Sth wrong with line %d'%i_line)
            continue

        # title
        tf = dict()
        termcount = segment[1].split(',')
        len_tc = len(termcount)
        if len_tc==0 or len_tc%2 != 0:
            print('Sth wrong with line %d'%i_line)
            continue

        for i in range(0,len_tc,2):
            if (ngram_range is not None):
                n = len(termcount[i].split(' '))
                if n<ngram_range[0] or n>ngram_range[1]:
                    continue
        
            tf[termcount[i]] = int(termcount[i+1])*title_weight

        # abstract
        termcount = segment[2].split(',')        
        len_tc = len(termcount)
        if len_tc==0 or len_tc%2 != 0:
            print('Sth wrong with line %d'%i_line)
            continue

        for i in range(0,len_tc,2):
            if (ngram_range is not None):
                n = len(termcount[i].split(' '))
                if n<ngram_range[0] or n>ngram_range[1]:
                    continue

            if termcount[i] in tf:
                tf[termcount[i]] += int(termcount[i+1])
            else:
                tf[termcount[i]] = int(termcount[i+1])

        # output
        fout.write(segment[0]+'\t'+ \
            ','.join(['%s,%d'%(t,w) for t,w in tf.iteritems()])+'\n')

        i_line+=1

    fin.close()
    fout.close()

def get_df(fns_in,fn_out,thre=None, sort_term=True):
    
    ''' Get the df from a collection of files'''

    if isinstance(fns_in,str):
        fns_in = [fns_in]
    
    df = dict()
    for fn in fns_in:
        fin = open(fn)
        for line in fin:
            sp = (line.strip()).split('\t')
            termcount = sp[1].split(',')
            for i in range(0,len(termcount),2):
                if termcount[i] in df:
                    df[termcount[i]] += 1
                else:
                    df[termcount[i]] = 1                    
        fin.close()

    # sort and output by df
    count_term = {}
    for k,c in df.iteritems():
        if c in count_term:
            count_term[c].append(k)
        else:
            count_term[c] = [k]

    fout = open(fn_out,'w')
    for c in sorted(count_term.keys(),reverse=True):
        if (thre is not None) and (c<thre):
            continue
        if sort_term:
            fout.write('\n'.join([term+'\t'+str(c) for term in sorted(count_term[c])]))
        else:
            fout.write('\n'.join([term+'\t'+str(c) for term in count_term[c]]))

        fout.write('\n')
    fout.close()


