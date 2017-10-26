from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tsne_plot(tsne_plot_name, labels, latent_representation, take_first_n = 200, 
      perplexity = 40, n_iter = 500, n_components = 2, ): 
      ''' list int, str labels: color point according to this color 
          np.ndarray latent_representation: [ Num of sequences X Dim of Latent Rep. ]
          take_first_n: only first n row will show in tsne projection 
      '''

      labels = labels[:take_first_n]
      latent_representation = latent_representation[:take_first_n, :]

      tsne = TSNE(
            n_components= n_components, verbose=1, perplexity=perplexity, n_iter=n_iter)
      tsne_results = tsne.fit_transform(latent_representation)

      # color by label, 
      unique_labels = list(set(labels))
      colors = [ unique_labels.index(l) for l in labels]

      print(tsne_results.shape, len(unique_labels))
      plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=colors)
      plt.legend()
      plt.savefig(tsne_plot_name, bbox_inches='tight')
      plt.show()