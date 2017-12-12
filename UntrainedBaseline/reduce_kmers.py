import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main(args):
    # Load the kmer representations
    print('Loading kmers...')
    kmers = pd.read_csv(args.kmers, sep=' ', index_col=0, header=[0,1])
    kmers = kmers.transpose()
    names = [name[0] + '||' + name[1] for name in kmers.index.values]

    # Center and scale
    print('Standardizing...')
    kmers = StandardScaler().fit_transform(kmers)

    # Compute PCA
    print('Starting PCA...')
    pca = PCA(n_components=args.dim, whiten=args.whiten)
    reduced_kmers = pca.fit_transform(kmers)

    # Format to match learned reps from trained models
    reduced_kmers = pd.DataFrame(reduced_kmers)
    reduced_kmers['name'] = names

    # Save to csv
    print('Saving output...')
    
    if args.dim == 64:
        model = 'encoder'
    elif args.dim == 128:
        model = 'decoder'
    else:
        model = args.dim
    
    output = args.save_dir+model+'_reps_whiten'+str(args.whiten)+'_kmers.csv'
    reduced_kmers.to_csv(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmers', type=str, default='Proteindomains_complete_4-1mers_4mer_features.txt',
                       help='path to txt file with kmers')    
    parser.add_argument('--dim', type=int, default=64,
                       help='number of principal components')    
    parser.add_argument('--whiten', action='store_true', default=False,
                       help='whether to whiten (see sklearn.decomposition.PCA for more)')    
    parser.add_argument('--save_dir', type=str, default='./',
                       help='directory to save reduced kmer representations after PCA')
    args = parser.parse_args()
    main(args)
    