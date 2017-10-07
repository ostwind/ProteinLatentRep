import os
from model import Seq2Vec

dir = os.path.dirname(__file__)
fasta_input = os.path.join(dir, '../data/uniprot.fasta')
data_dir = os.path.join(dir, '../data/dis_prot_processed/')

if __name__ == '__main__':

    model = Seq2Vec(None, fasta_input, data_dir)

    # example latent rep for a sequence
    example_vect = model.vect_rep( 'sp|P04637|P53_HUMAN_0' )
    print(example_vect)

    # all ids
    #print(model.all_ids())
