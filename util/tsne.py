from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np 


def tsne(matrix,
      perplexity = 50, n_iter = 2000, n_components = 3, ): 
      tsne = TSNE(
            n_components= n_components, verbose=1, perplexity=perplexity, n_iter=n_iter)
      return tsne.fit_transform(matrix)


import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import random

def plotly_scatter(plot_name, labels, latent_representation, take_first_n = 10000):
      plotly.tools.set_credentials_file(username='mrnood', api_key='2o0NGhHfo4RXCJudQ3Mi')
      
      if take_first_n > len(labels):
            take_first_n = len(labels)

      labels = labels[:take_first_n]
      latent_representation = latent_representation[:take_first_n, :]
      latent_representation = tsne(latent_representation)

      unique_labels = sorted(list(set(labels)))
      #unique_labels = [l for l in unique_labels if l[0] != 'S'] # autencoder didnt work here, removing 'noise' atm
      # plot only gene symbols with >40 RRMs
      large_gene_symbol, unique_labels_filtered = [], []
      for l in unique_labels: 
            all_l_indices = [i for i,x in enumerate(labels) if x == l]
            if len(all_l_indices) < 40 or len(all_l_indices) > 60:
                  continue  
            print(l, len(all_l_indices))
            large_gene_symbol.append(all_l_indices)
            unique_labels_filtered.append(l)

      large_gene_symbol = np.array(large_gene_symbol)

      l = []
      N= len(unique_labels_filtered)
      c= ['hsl('+str(h)+',50%'+',40%)' for h in np.linspace(0, 360, N)]
      c2 = ['hsl('+str(h)+',50%'+',30%)' for h in np.linspace(0, 360, N)]
      c3 = ['hsl('+str(h)+',40%'+',60%)' for h in np.linspace(0, 360, N)]
      
      palette = [c2, c, c3]
      c = 0
      for unique_label, all_l_indices in zip(unique_labels_filtered, large_gene_symbol):
            cur_palette = c % 3
            #y.append((2000+i))
            #for point in all_l_indices:
            print(palette[cur_palette][c])
            trace0 = go.Scatter3d(
                  x=latent_representation[all_l_indices, 0],
                  y=latent_representation[all_l_indices, 1],
                  z=latent_representation[all_l_indices, 2],
                  mode='markers',
                  marker=dict(size=6,
                              #colorscale='jet',  # 'Viridis',
                              #line= dict(width=1),
                              color=palette[cur_palette][c],
                              #[unique_labels_filtered.index(unique_label)],
                              #opacity=0.75,
                              ), name=unique_label,)
            #text= unique_label) # The hover text goes here...
            l.append(trace0)
            c += 1
      layout = go.Layout(
                    scene = dict(
                    xaxis = dict(
                        #nticks=4,
                        #gridwidth=4,
                        title = '',
                        showgrid=False,
                        showline=False,
                        #zeroline=False,
                        showticklabels=False,
                         ),
                    yaxis = dict(
                        #nticks=4,
                        title = '',
                        showgrid=False,
                        #gridwidth=2,
                        showline=False,
                        #zeroline=False,
                        showticklabels=False,
                         ),
                    zaxis = dict(
                        #nticks=4,
                        title = '', 
                        gridwidth=2,
                        showgrid = False,
                        showline=False,
                        #zeroline=False,
                        showticklabels=False,
                        ),
                        ),
                  height = 1000,
                  width=1200,
                  margin=dict(
                  r=0, l=0,
                  b=0, t=0),
                  showlegend= True,
                  )

      fig= go.Figure(data=l, layout=layout)
      py.iplot(fig)


def plot3d(plot_name, labels, latent_representation, take_first_n = 10000, 
      perplexity = 40, n_iter = 2000, n_components = 2): 

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
      

      # plot only gene symbols with >40 RRMs
      large_gene_symbol, unique_labels_filtered = [], []
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
            plt.scatter(
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

def hist(hist_name, array, bins = 100, show = True, ):
      plt.hist(np.array(array), bins)
      plt.title(hist_name)
      
      if show:
            plt.show()
            return 

      plt.savefig('./CharAE_Cho/activations/%s' %(hist_name))
      plt.clf()
      

def plot(plot_name, labels, latent_representation, take_first_n = 10000, 
      n_iters=2000, perplexity=40): 

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
