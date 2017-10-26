from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import generic_protein

def write_fasta(list_of_sequences, list_of_ids, 
fasta_name = 'RRM_55.fasta',description = None):
      
      print(list_of_ids)
      record = []
      for sequence, name in zip(list_of_sequences, list_of_ids):
            record.append(
                  SeqRecord(Seq(sequence, generic_protein),
                  id=name,
                  ))

      from Bio import SeqIO
      SeqIO.write(record, fasta_name, "fasta")
