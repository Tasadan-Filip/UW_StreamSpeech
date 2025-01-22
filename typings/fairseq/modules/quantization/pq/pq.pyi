"""
This type stub file was generated by pyright.
"""

from .em import EM

class PQ(EM):
    """
    Quantizes the layer weights W with the standard Product Quantization
    technique. This learns a codebook of codewords or centroids of size
    block_size from W. For further reference on using PQ to quantize
    neural networks, see "And the Bit Goes Down: Revisiting the Quantization
    of Neural Networks", Stock et al., ICLR 2020.

    PQ is performed in two steps:
    (1) The matrix W (weights or fully-connected or convolutional layer)
        is reshaped to (block_size, -1).
            - If W is fully-connected (2D), its columns are split into
              blocks of size block_size.
            - If W is convolutional (4D), its filters are split along the
              spatial dimension.
    (2) We apply the standard EM/k-means algorithm to the resulting reshaped matrix.

    Args:
        - W: weight matrix to quantize of size (in_features x out_features)
        - block_size: size of the blocks (subvectors)
        - n_centroids: number of centroids
        - n_iter: number of k-means iterations
        - eps: for cluster reassignment when an empty cluster is found
        - max_tentatives for cluster reassignment when an empty cluster is found
        - verbose: print information after each iteration

    Remarks:
        - block_size be compatible with the shape of W
    """
    def __init__(self, W, block_size, n_centroids=..., n_iter=..., eps=..., max_tentatives=..., verbose=...) -> None:
        ...
    
    def encode(self): # -> None:
        """
        Performs self.n_iter EM steps.
        """
        ...
    
    def decode(self): # -> Tensor | Any:
        """
        Returns the encoded full weight matrix. Must be called after
        the encode function.
        """
        ...
    


