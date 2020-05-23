# Centroid-Model
A dynamic neural network with strictly local connectivity to compute image centroids

## File descriptions:
- run_examples.m    : Script that demonstrates how to call the 1-D and 2-D model functions
- model_1d.m        : Apply the 1-D model to an input image, beta. This version makes use of literal convolutions with the kernel w. It may be helpful for understanding the model, but will likely be slower than model_1d_matrix().
- model_1d_matrix.m : Apply the 1-D model to an input image, beta. This version casts everything as matrix operations and can (therefore) make use of Matlab's optimizations.
- model_2d.m        : Apply the 2-D model to an input image, beta. This version does not recast layers as vectors and makes use of literal convolutions with the kernel w. It may be helpful for understanding the model, but will likely be slower than model_2d_matrix().
- model_2d_matrix.m : Apply the 2-D model to an input image, beta. This version casts everything as matrix operations and can (therefore) make use of Matlab's optimizations.
- w_1d.m            : Generate the 1-D convolution matrix. W*A here is equivalent to W \ast A in the text
- w_2d.m            : Generate the 2-D convolution matrix. W*A here is equivalent to W \ast A in the text
- conv_1d.m         : Perform the 1-D convolution, given A is periodic
- conv_2d.m         : Perform the 2-D convolution, given A is periodic
- images.mat        : Matrices containing the example images used in the text
