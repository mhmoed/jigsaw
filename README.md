# Jigsaw puzzle solver
This package contains a greedy and linear programming-based solver for jigsaw 
puzzle problems.

## Linear programming-based solver
The solver based on linear programming is implemented from the paper by 
Yu et al. (2015). This package implements the 'free' strategy and can only 
solve Type I problems (i.e. problems where the orientation of the pieces is 
known). For details about the method, please see [LP](LP.md).

## Greedy solver
The greedy solver implements a straightforward algorithm where, starting from a 
random initial piece, best matching pieces are assigned to open boundaries. The 
solver uses Mahalanobis Gradient Compatbility to determine the suitability of 
matches, as described by Gallagher et al. (2012).

## References
1. Yu, R., Russell, C., & Agapito, L. (2015). Solving Jigsaw Puzzles with
    Linear Programming. arXiv preprint arXiv:1511.04472.

1. Gallagher, A. C. (2012). Jigsaw puzzles with pieces of unknown
    orientation. In Computer Vision and Pattern Recognition (CVPR), 2012 IEEE
    Conference on (pp. 382-389). IEEE.