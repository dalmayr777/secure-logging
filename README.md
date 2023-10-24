# Implementation for http://eprint.iacr.org/2019/506

The software creates a random $m\times{}n$ binary matrix $M$ where each column has weight $k$. 
Then, it removes $\delta$ random rows from $M$.  Finally, it checks whether $M$ has rank $n$.

To compile, you need OpenMP, libsodium-dev, and a CPU with support for AVX instructions.

Run the binary with: `./onlyCheck n c k rowsToRemove logOfRuns`, where $n$ is the number of columns, $c$ is the expansion parameter such that $m=c\cdot{}n$, $k$ is the weight of each column, $rowsToRemove$ the number of rows to remove, and $logOfRuns$ the logarithm of the number of runs you want to run per sample.

Example:

```./onlyCheck 1024 1.1243 5 20 5```

Will do $n=1024, c=1.124300, m=1152, k=5, runs=32$, and remove $20$ rows in each run.
