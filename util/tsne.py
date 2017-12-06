from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

def tsne(matrix, perplexity = 40, n_iter = 2000, n_components = 2): 
      tsne = TSNE(
            n_components= n_components, verbose=1, perplexity=perplexity, n_iter=n_iter)
      return tsne.fit_transform(matrix)

def plot3d(plot_name, labels, latent_representation, take_first_n = 10000, 
      perplexity = 40, n_iter = 2000, n_components = 2): 
      ''' list int, str labels: color point according to this color 
          np.ndarray tsne_projection: [ Num of sequences X Dim of Latent Rep. ]
          take_first_n: only first n row will show in tsne projection 
      '''
      from mpl_toolkits.mplot3d import Axes3D
      assert len(labels) == latent_representation.shape[0], \
      '%s != %s ' %(len(labels),  latent_representation.shape[0])

      if take_first_n > len(labels):
            take_first_n = len(labels)

      labels = labels[:take_first_n]
      latent_representation = latent_representation[:take_first_n, :]

      tsne_projection = tsne(latent_representation, n_components = 3)

      # color by label, 
      unique_labels = sorted(list(set(labels)))
      colors = [ unique_labels.index(l) for l in labels]
      
      large_gene_symbol = []
      unique_labels_filtered = []
      for l in unique_labels: 
            all_l_indices = [i for i,x in enumerate(labels) if x == l]
            if len(all_l_indices) < 20:
                  continue  
            large_gene_symbol.append(all_l_indices)
            unique_labels_filtered.append(l)

      fig = plt.figure(figsize=(10, 8))
      ax = fig.add_subplot(111, projection='3d')

      colors = iter(cm.jet(np.linspace(0, 1, len(large_gene_symbol))))
      for l, all_l_indices in zip(unique_labels_filtered, large_gene_symbol):
            cur_color = next(colors)
            ax.scatter(
                  tsne_projection[all_l_indices,0], 
                  tsne_projection[all_l_indices,1],
                  tsne_projection[all_l_indices,2], 
                  c= cur_color, label=l, s = 30)

      ax.legend(fontsize=15)
      #ax = plt.gca()
      #legend = ax.get_legend()
      # ax.set_zlim(-10, 10)
      # ax.set_ylim(-10, 10)
      plt.title(plot_name, fontsize=18)
      # ax.set_xlim(-10, 10)
      plt.savefig(plot_name, bbox_inches='tight')
      plt.show()
      return fig


def plot(plot_name, labels, latent_representation, take_first_n = 10000, 
      n_iters=2000, perplexity=40): 
      ''' list int, str labels: color point according to this color 
          np.ndarray tsne_projection: [ Num of sequences X Dim of Latent Rep. ]
          take_first_n: only first n row will show in tsne projection 
      '''
      assert len(labels) == latent_representation.shape[0], \
      '%s != %s ' %(len(labels),  latent_representation.shape[0])

      if take_first_n > len(labels):
            take_first_n = len(labels)

      labels = labels[:take_first_n]
      latent_representation = latent_representation[:take_first_n, :]

      tsne_projection = tsne(latent_representation)

      # color by label, 
      unique_labels = sorted(list(set(labels)))
      colors = [ unique_labels.index(l) for l in labels]
      
      large_gene_symbol = []
      unique_labels_filtered = []
      for l in unique_labels: 
            all_l_indices = [i for i,x in enumerate(labels) if x == l]
            if len(all_l_indices) < 40:
                  continue
            large_gene_symbol.append(all_l_indices)
            unique_labels_filtered.append(l)

      fig = plt.figure(figsize=(20, 20))
      ax = fig.add_subplot(111)
      colors = iter(cm.jet(np.linspace(0, 1, len(large_gene_symbol))))
      for l, all_l_indices in zip(unique_labels_filtered, large_gene_symbol):
            
            cur_color = next(colors)            
            plt.scatter(
                  tsne_projection[all_l_indices,0], 
                  tsne_projection[all_l_indices,1], 
                  c= cur_color, label=l, s = 30)

      ax.legend(fontsize=16)
      plt.title(plot_name, fontsize=25)
      #legend = ax.get_legend()
      
      plt.savefig(plot_name, bbox_inches='tight')
      plt.show()

def hist(hist_name, array, show = True):
      plt.hist(np.array(array))
      plt.title(hist_name)
      plt.show()

# def tsne_plot(tsne_plot_name, labels, latent_representation, take_first_n = 500, 
#       perplexity = 40, n_iter = 1500, n_components = 2, ): 
#       ''' list int, str labels: color point according to this color 
#           np.ndarray latent_representation: [ Num of sequences X Dim of Latent Rep. ]
#           take_first_n: only first n row will show in tsne projection 
#       '''
#       assert len(labels) == latent_representation.shape[0], \
#       '%s != %s ' %(len(labels),  latent_representation.shape[0])

#       if take_first_n > len(labels):
#             take_first_n = len(labels)

#       labels = labels[:take_first_n]
#       latent_representation = latent_representation[:take_first_n, :]

#       tsne = TSNE(
#             n_components= n_components, verbose=1, perplexity=perplexity, n_iter=n_iter)
#       tsne_results = tsne.fit_transform(latent_representation)

#       # color by label, 
#       unique_labels = list(set(labels))
#       colors = [ unique_labels.index(l) for l in labels]
#       #print(colors)
#       #print(unique_labels)
#       colmaps = ['Blues', 'Greys', 'Reds', 'Purples', 'Greens', 'Oranges']
      
#       print(tsne_results.shape, len(unique_labels))
#       #plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, s = 1)
#       colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_labels))))
#       i = 0
#       colors_used = []
#       for l in unique_labels: 
#             all_l_indices = [i for i,x in enumerate(labels) if x == l]  
#             #c_subset = [colors[i] for i in all_l_indices]
#             #print(c_subset)
#             #print(l, tsne_results[all_l_indices,0])
#             cur_color = next(colors)
#             colors_used.append(cur_color)
#             plt.scatter(
#                   tsne_results[all_l_indices,0], 
#                   tsne_results[all_l_indices,1], 
#                   c= cur_color, label=l, s = 1)
#             i += 1
#       plt.legend()
#       ax = plt.gca()
#       legend = ax.get_legend()
#       i= 0
#       plt.savefig(tsne_plot_name, bbox_inches='tight')
#       plt.show()
