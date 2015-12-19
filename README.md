# Jigsaw puzzle solver
This package contains a linear programming-based solver for jigsaw puzzle 
problems. The solver is based on linear programming and is implemented from the 
paper by Yu et al. (2015). The current implementation only provides the 'free' 
strategy and can only solve Type I problems (i.e. problems where the 
orientation of the pieces is known).

## Running the code
To subdivide any square image into a number of blocks (sixteen by default) 
and shuffle them, use:

<pre>
    shuffle-image input.png shuffled.png [-n <i>number of pieces</i>] [-r <i>random seed</i>]
</pre>

To reconstruct the shuffled image with the LP-based algorithm, use:

<pre>
    solve-jigsaw-lp shuffled.png reconstructed.png [-n <i>number of pieces</i>] [-r <i>random seed</i>] [-m <i>max. number of simplex iterations</i>]
</pre>


## Known issues
``scikit-image`` may complain about unknown locales. 
To resolve this issue, please make sure to have a valid locale set, e.g. in 
a Bash shell enter:

    $ export LC_ALL=en_US.UTF-8; export LANG=en_US

## References
1. Yu, R., Russell, C., & Agapito, L. (2015). Solving Jigsaw Puzzles with
    Linear Programming. arXiv preprint arXiv:1511.04472.
