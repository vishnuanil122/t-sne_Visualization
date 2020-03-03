# t-SNE_Visualization
Visualising high-dimensional datasets using PCA and t-SNE in Python


t-Distributed Stochastic Neighbor Embedding (t-SNE) is an unsupervised, non-linear technique primarily used for data exploration and visualizing high-dimensional data. In simpler terms, t-SNE gives you a feel or intuition of how the data is arranged in a high-dimensional space. It was developed by Laurens van der Maatens and Geoffrey Hinton in 2008.

The key characteristic of t-SNE is that it solves a problem known as the crowding problem. The extent to which this problem occurs depends on the ratio between the intrinsic data dimensionality and the embedding dimensionality. So, if you embed in, say, thirty dimensions, the crowding problem is less severe than when you embed in two dimensions. As a result, it often works better if you increase the degrees of freedom of the t-distribution when embedding into thirty dimensions 
