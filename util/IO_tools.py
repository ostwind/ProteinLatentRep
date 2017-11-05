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
