from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein

def protein_rna_map(lookup_file_path = './data/protein_rna_map.txt'):
      dictionary = dict()
      with open(lookup_file_path) as text:
            for rrm_rna_pair in text:
                  rrm, rna = rrm_rna_pair.split()
                  dictionary[rrm] = rna
      return dictionary

def write_fasta(list_of_sequences, list_of_ids, 
fasta_name,description = None):
      record = []
      for sequence, name in zip(list_of_sequences, list_of_ids):
            record.append(
                  SeqRecord(Seq(sequence, generic_protein),
                  id=name,
                  ))

      from Bio import SeqIO
      SeqIO.write(record, fasta_name, "fasta")

def write_to_csv(name_ordering, proteins, path):
    import numpy as np 
    import pandas as pd
    proteins = np.array(proteins)
    print('writing latent representation of shape: ', proteins.shape)
    df = pd.DataFrame(data=proteins,
              index=name_ordering,)    
    df.to_csv(path , sep='\t')

def OneHot(tensor):
    batch_size, seq_len =  tensor.shape
    tensor = torch.unsqueeze(tensor, 2)
    OneHot = torch.LongTensor(batch_size, seq_len, 23).zero_()
    #print(OneHot.shape, tensor.shape)
    OneHot = OneHot.scatter_(2, tensor , 1)
    return OneHot
