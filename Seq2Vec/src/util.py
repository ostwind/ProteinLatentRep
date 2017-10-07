
import pickle
import os
import numpy as np
import pandas as pd
from model import *
'''
embedding ============write_pred_target===> matrix.p, target.p =train_test_split===> X_train, X_test, y_train, y_test
target_catalog_file ======================>
'''

def write_pred_target(embedding, target_catalog_file = '../data/meta_files/family_classification_metadata.tab', ):
    if os.path.isfile("./matrix.p") and os.path.isfile('./target.p'):
        print('X and y already extracted ')
        return

    df = pd.read_table(target_catalog_file)
    model = Seq2Vec(embedding)
    realization = model.all_ids()

    # write sample vector and label sample-wise
    target = []
    matrix = []
    i = 0
    for proteinID in realization:
        AccessionID = proteinID.split('|')[1]
        if AccessionID in df['SwissProtAccessionID'].values:
            matrix.append( model.vect_rep(proteinID) )
            target.append( df[df['SwissProtAccessionID'] == AccessionID]['FamilyID'].values[0] )
            i += 1
        if i % 1000 == 0:
            print(i/50000)
            if i == 50000:
                break
    matrix= np.array(matrix)

    with open('matrix.p', 'wb') as x:
        pickle.dump(matrix, x)
    with open('target.p', 'wb') as y:
        pickle.dump(target, y)

def load_x_y(x = 'matrix.p', y = 'target.p'):
    return pickle.load( open(x, 'rb') ), pickle.load( open(y, 'rb') )

def train_test_split():
    matrix, target = load_x_y()
    #careful, also does columnwise normalization
    matrix_normed = (matrix - matrix.min(0)) / matrix.ptp(0)

    X_train, X_test, y_train, y_test = train_test_split(
    matrix_normed, target, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

''' functions from biovec, an older paper that applies word2vec to sequences
    Not used in model.py, but may be useful for data preprocessing.
'''

def split_ngrams(seq, n):
    """
    'AGAMQSASM' => [['AGA', 'MQS', 'ASM'], ['GAM','QSA'], ['AMQ', 'SAS']]
    """
    a, b, c = zip(*[iter(seq)]*n), zip(*[iter(seq[1:])]*n), zip(*[iter(seq[2:])]*n)
    str_ngrams = []
    for ngrams in [a,b,c]:
        x = []
        for ngram in ngrams:
            x.append("".join(ngram))
        str_ngrams.append(x)
    return str_ngrams

def generate_corpusfile(fname, n, out):
    '''
    Args:
        fname: corpus file name
        n: the number of chunks to split. In other words, "n" for "n-gram"
        out: output corpus file path
    Description:
        Protvec uses word2vec inside, and it requires to load corpus file
        to generate corpus.
    This function was found in the original implementation of BioVec, not Seq2Vec
    '''
    f = open(out, "w")
    for r in SeqIO.parse(fname, "fasta"):
        ngram_patterns = split_ngrams(r.seq, n)
        f.write(str(r.id) + " ")
        for ngram_pattern in ngram_patterns:
            f.write(" ".join(ngram_pattern) + " ")
        f.write("\n")
    f.close()
