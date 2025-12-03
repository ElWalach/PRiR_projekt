struct MatrixData {
    int n;
    int mode;  /* 0 = sekwencyjny, 1 = równoległy */
    double matrix<>;  /* spłaszczona macierz A (n*n elementów) + wektor b (n elementów) */
};

struct SolverResult {
    int n;
    double time_ms;
    int iterations;
    double solution<>;  /* wektor rozwiązania */
};

program CHEBYSHEV_SOLVER {
    version CHEBYSHEV_VERS {
        SolverResult SOLVE_LINEAR_SYSTEM(MatrixData) = 1;
    } = 1;
} = 0x20000001;