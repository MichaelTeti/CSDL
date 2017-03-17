# CSDL
Compressed Deep Learning - combining compressed sensing and deep learning

Often, the goal of compressed sensing is to simultaneously sample and compress a sparse signal, then reconstruct it. Here, we take the compressed measurement vector, b, and learn directly on that, which has never been done before. We propose two methods of learning, one supervised and one unsupervised, from these measurements. 

## Supervised Method
In the [supervised method](https://github.com/MichaelTeti/CSDL/blob/master/CSDL.py), we take compressed measurements of images and send those directly to a deep neural network. When applied to hyperspectral satellite data, the network is able to learn over 10x faster on compressed data and get the same accuracy.

## Unsupervised Method
[Here](https://github.com/MichaelTeti/CSDL/blob/master/CS_DictionaryLearning.py), we take the compressed measurements and learn a dictionary for each class of images. To classify, we send new compressed measurements in and sum the alpha coefficients returned. 
