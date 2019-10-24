# Latent Representation for RNA Recognition Motifs
Data Science Capstone Project for Millie Dwyer, Julie Helmers, Shasha Lin, and Lihan Yao. 
Supervised by Dr. Quaid Morris.  

## How can this representation be used?
After processing RNA Recognition Motif data from Uniprot, our models complete unsupervised learning on this input to generate an embedding.

Our embeddings can be utilized as input for: 
- Affinity Regression
- Sequence Fitness Prediction (under construction)

## Awards
NYU Center for Data Science Academy Award: Best Interdisciplinary Project, 2018 

## Viz
[Visualization of partial RRM latent representation using t-SNE projection (representation generated by NMT model) ](https://plot.ly/~mrnood/108)

<div>
    <a href="https://plot.ly/~mrnood/108/?share_key=atYJmd3B8OqYly2cw8RduT" target="_blank" title="sent_to_AR_params_documented" style="display: block; text-align: center;"><img src="https://plot.ly/~mrnood/108.png?share_key=atYJmd3B8OqYly2cw8RduT" alt="sent_to_AR_params_documented" style="max-width: 100%;width: 1200px;"  width="1200" onerror="this.onerror=null;this.src='https://plot.ly/404.png';" /></a>
</div>

A Recurrent Neural Network's neurons can capture meaningful features of biological sequences. Below are activation patterns of two neurons taken from a RNN layer of our model over the same batch of sequence data. Brighter amino acid cells indicate stronger activations from that neuron.      
![neuron63](./figs/neuron63.png)
![neuron11](./figs/neuron11.png)
