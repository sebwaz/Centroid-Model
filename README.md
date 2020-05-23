# Centroid-Model
A dynamic neural network with strictly local connectivity to compute image centroids

## File descriptions:
- __run_examples.m__    : Script that demonstrates how to call the 1-D and 2-D model functions
- __model_1d.m__        : Apply the 1-D model to an input image, beta. This version makes use of literal convolutions with the kernel w. It may be helpful for understanding the model, but will likely be slower than model_1d_matrix().
- __model_1d_matrix.m__ : Apply the 1-D model to an input image, beta. This version casts everything as matrix operations and can (therefore) make use of Matlab's optimizations.
- __model_2d.m__        : Apply the 2-D model to an input image, beta. This version does not recast layers as vectors and makes use of literal convolutions with the kernel w. It may be helpful for understanding the model, but will likely be slower than model_2d_matrix().
- __model_2d_matrix.m__ : Apply the 2-D model to an input image, beta. This version casts everything as matrix operations and can (therefore) make use of Matlab's optimizations.
- __w_1d.m__            : Generate the 1-D convolution matrix. W*A here is equivalent to W \ast A in the text
- __w_2d.m__            : Generate the 2-D convolution matrix. W*A here is equivalent to W \ast A in the text
- __conv_1d.m__         : Perform the 1-D convolution, given A is periodic
- __conv_2d.m__         : Perform the 2-D convolution, given A is periodic
- __images.mat__        : Matrices containing the example images used in the text
