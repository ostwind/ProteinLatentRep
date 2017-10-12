from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import doc2vec, Doc2Vec

from Bio import SeqIO
import os
import pickle
from util import *

class Seq2Vec(doc2vec.Doc2Vec):
    def __init__(self,embedding_path = None,
     fasta_path = None, data_dir = None,
    window_size=3, size=100, window=5,
    min_count=2, workers=4, epochs = 5):

        self.embedding_path = embedding_path
        if not embedding_path:
            self.embedding_path = 'ws%s_size%s_win%s_e%s.d2v' %(window_size, size, window, epochs)

        if os.path.isfile(self.embedding_path):
            print('loading model at %s' %(self.embedding_path))
            self.model = Doc2Vec.load(self.embedding_path)
            return

        print('generating new model at %s' %(self.embedding_path))
        self.fasta_path = fasta_path
        self.data_dir = data_dir

        # embedding parameters, see doc2vec documentation for parameter explanation
        self.window_size = window_size
        self.size = size # dimensionality of the distributed representation
        self.min_count = min_count
        self.workers = workers
        self.epochs = epochs
        self.window = window

        # first unpack fasta file into a directory of .txt, then learn embedding
        if not os.path.isfile('./dir_key.p'):
            self._preprocess_fasta()
        self._learn_embedding()
        self.model = Doc2Vec.load(self.embedding_path)

    def _preprocess_fasta(self): # existence of pickle to tell this class no need for generating data dir
        assert os.path.isfile(self.fasta_path), "fasta not found at %s, and something wrong with data dir" %(self.fasta_path)
        print('Generating corpus files from fasta file...')

        dir_key = gen_dir( filepath = self.fasta_path, n = self.window_size, out_dir = self.data_dir )
        pickle.dump(dir_key, open("dir_key.p", "wb"))

    def _learn_embedding(self):
        assert os.path.isfile("./dir_key.p"), "directory key does not exist or data dir not created"
        print('Loading directory key and initializing model, model initialization takes a long time')
        dir_key = pickle.load(open('dir_key.p', "rb"))
        sentences = LabeledLineSentence(dir_key)

        self.model = Doc2Vec( sentences, dm = 0,
        size=self.size, window=self.window, #min_count=50,
        negative = 2, hs = 1,
        sample = 1e-3, # downsampling
        workers= self.workers, alpha=.025)
        self.model.save(self.embedding_path)

        print('Training model at %s for %s epochs and %s total samples' %(self.embedding_path, self.epochs, self.model.corpus_count))
        self.model.train(sentences, epochs = self.epochs, total_examples = self.model.corpus_count)
        self.model.save(self.embedding_path)
        print('Done!')

    def vect_rep(self, string):
        assert string in self.model.docvecs.doctags.keys(), '%s not found in %s' %(string, self.embedding_path)
        return self.model.docvecs[string]

    def all_ids(self):
        return self.model.docvecs.doctags.keys()

'''
word2vec + biological sequence = BioVec (original paper)
word2vec + amino-acid sequences = Protvec
DNA2Vec (?) is an extension w/ sequence lengths varying from 3 to 5 (I think)
doc2vec + amino-acid sequences = Seq2Vec (current implementation)

SEQ2VEC PAPER @ http://dtmbio.net/dtmbio2016/pdf/11.pdf says:
dbow (faster, preferred for large corpus) vs distributed memory
context window size
vector space dimension, hierarchical softmax and/or
negative sampling for learning procedure, and the threshold
for sub-sampling high-frequency words are all selected using
cross-validation.
'''

def gen_dir(filepath,n, out_dir):
    '''
    filepath: input fasta file to be made into dir
    n: n for n-mer (or n-gram)
    out_dir: output directory into which .txt files are written
    '''
    ids = dict()
    for r in SeqIO.parse(filepath, "fasta"):
        print(r)
        ngram_patterns = split_ngrams(r.seq, n)
        f = open(out_dir+str(r.id)+'.txt', 'w+')
        ids[out_dir + str(r.id)+'.txt'] = str(r.id)

        for ngram_pattern in ngram_patterns:
            f.write(" ".join(ngram_pattern))
        f.close()
    return ids

# I don't know how this works, except it instantiates nice labeled input to doc2vec
class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources
        flipped = {}
        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    #print(utils.to_unicode(line).split())
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    # def to_array(self):
    #     self.sentences = []
    #     for source, prefix in self.sources.items():
    #         with utils.smart_open(source) as fin:
    #             for item_no, line in enumerate(fin):
    #                 self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
    #     return self.sentences
    #
    # def sentences_perm(self):
    #     shuffle(self.sentences)
    #     return self.sentences