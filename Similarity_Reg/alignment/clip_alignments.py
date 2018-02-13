# file to clip new alignments from last 20 amino acids
# python clip_alignments.py -i ../../../comineddata_nolinegaps.fasta 

import numpy as np
import argparse


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input_file', '-i', type=str, default="",
        help="Input file of sequences to trim.")
    parser.add_argument('--pos', '-p', type=int, default=20, help="position of tailing amino acids to trim to")
    parser.add_argument('--output_file', '-o', type=str,default="output.fasta", help="where to save trimmed sequences")
    return parser





list_of_amino_acids = ["A","B","C","D","E","F","G","H","I","J","K","L","M",'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
# get average position where tail of sequence hits 20 amino acids
def find_position(sequences, amino_acids=list_of_amino_acids):
    final_pos = []
    for seq in sequences:
        seq=seq[1]
        i = 1
        aa_count = 0
        while aa_count < 20:
            if i >= len(seq):
                break
            last_pos = seq[-1* i]
            # print(last_pos)
            if last_pos in amino_acids:
                aa_count +=1
            # print(aa_count)
            i+=1
        final_pos.append(i)
    avg_pos = np.mean(final_pos)
    return np.floor(avg_pos)


def clip_alignments(sequences):
    print(sequences.shape)
    pos = find_position(sequences).astype(int)
    seqs = []
    for s in sequences:
        new_s1 = s[1][:-1*pos]
        seqs.append([s[0], new_s1])
    return np.array(seqs)#sequences[:,:-1*pos]


# if input sequences of variable length, append " " to end
def fill_nas(sequence, max_length):
    if len(sequence) < max_length:
        new_sequence = sequence + " "* (max_length - len(sequence))
    else:
        return sequence
    return new_sequence


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    infile = args.input_file
    pos_to_trim = args.pos
    outfile = args.output_file
    
    seqs = []
    with open(infile,'r') as f:
        thisitem = []
        for line in f:
            if ">" in line:
                thisitem.append(line)
            else:
                thisitem.append(line)
                seqs.append(thisitem)
                thisitem=[]
    print(len(seqs))
    print(np.mean([len(x) for x in seqs]))
    sequences = np.array(seqs)
    #print(find_position(sequences))
    new_sequences = clip_alignments(sequences)
    print(sequences.shape, new_sequences.shape)
    with open(outfile, 'w+') as f:
        for s in new_sequences:
            f.write(s[0])
            f.write(s[1]+"\n")
    