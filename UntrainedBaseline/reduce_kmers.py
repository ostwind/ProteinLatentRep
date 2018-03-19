import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main(args):
    # Load the kmer representations
    kmers = pd.read_csv(args.kmers, sep=' ', index_col=0, header=[0,1])
    kmers = kmers.transpose()
    names = [name[0] + '||' + name[1] for name in kmers.index.values]
    #print(kmers[kmers['ALH'] > 0]['ALH'])
    
    # # Center and scale
    # kmers = StandardScaler().fit_transform(kmers)
    
    # # Compute PCA
    
    for dim in [64, 128, 200]:
        pca = PCA(n_components=dim, whiten=args.whiten)
        reduced_kmers = pca.fit_transform(kmers)

        # # Format to match learned reps from trained models
        #reduced_kmers = kmers
        reduced_kmers = pd.DataFrame(reduced_kmers)
        reduced_kmers[ 'name' ] = names

        reduced_kmers = reduced_kmers.set_index( 'name' )

        #print(reduced_kmers.head())
        
        output = args.save_dir+'4mer'+str(dim)+'_kmers.csv'
        #+str(args.whiten)
        reduced_kmers.to_csv(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--kmers', type=str, default='Proteindomains_complete_4-1mers_4mer_features.txt',
                       help='path to txt file with kmers')    
    parser.add_argument('--dim', type=int, default=200,
                       help='number of principal components')    
    parser.add_argument('--whiten', action='store_true', default=False,
                       help='whether to whiten (see sklearn.decomposition.PCA for more)')    
    parser.add_argument('--save_dir', type=str, default='../AffReg/',
                       help='directory to save reduced kmer representations after PCA')
    args = parser.parse_args()
    main(args)
    