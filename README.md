### graphblas_proj

Graph projections implemented with [graphblas](https://github.com/owensgroup/graphblas)

This is an alternative to the [gunrock implementation](https://github.com/gunrock/gunrock/tree/dev-refactor/tests/proj).  That implementation naively allocates a `|V|x|V|` array to store the projected graph.  Unfortunately, this leads to OOM errors very quickly.