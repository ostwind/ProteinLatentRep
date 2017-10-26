from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 

def tsne_plot(tsne_plot_name, labels, latent_representation, take_first_n = 500, 
      perplexity = 40, n_iter = 1500, n_components = 2, ): 
      ''' list int, str labels: color point according to this color 
          np.ndarray latent_representation: [ Num of sequences X Dim of Latent Rep. ]
          take_first_n: only first n row will show in tsne projection 
      '''
      assert len(labels) == latent_representation.shape[0], \
      '%s != %s ' %(len(labels),  latent_representation.shape[0])

      if take_first_n > len(labels):
            take_first_n = len(labels)

      labels = labels[:take_first_n]
      latent_representation = latent_representation[:take_first_n, :]

      tsne = TSNE(
            n_components= n_components, verbose=1, perplexity=perplexity, n_iter=n_iter)
      tsne_results = tsne.fit_transform(latent_representation)

      # color by label, 
      unique_labels = list(set(labels))
      colors = [ unique_labels.index(l) for l in labels]
      #print(colors)
      #print(unique_labels)
      colmaps = ['Blues', 'Greys', 'Reds', 'Purples', 'Greens', 'Oranges']
      
      print(tsne_results.shape, len(unique_labels))
      #plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors, s = 1)
      colors = iter(cm.rainbow(np.linspace(0, 1, len(unique_labels))))
      i = 0
      colors_used = []
      for l in unique_labels: 
            all_l_indices = [i for i,x in enumerate(labels) if x == l]  
            #c_subset = [colors[i] for i in all_l_indices]
            #print(c_subset)
            #print(l, tsne_results[all_l_indices,0])
            cur_color = next(colors)
            colors_used.append(cur_color)
            plt.scatter(
                  tsne_results[all_l_indices,0], 
                  tsne_results[all_l_indices,1], 
                  c= cur_color, cmap=cur_color, label=l, s = 1)
            i += 1
      plt.legend()
      ax = plt.gca()
      legend = ax.get_legend()
      i= 0
      #for color_str in colors_used:#colmaps[:len(unique_labels)]:
            # legend.legendHandles[ 
            #       i ].set_color( 
            #             eval('plt.cm.%s(.8)' %(color_str)) )
            # i += 1      
      plt.savefig(tsne_plot_name, bbox_inches='tight')
      plt.show()