# Mathematical Specification: Covariance Estimation & Matrix Conditioning

## 1. The Small-Sample Breakdown of Empirical Covariance
For a lookback window of $T$ trading sessions across $N$ active portfolio constituents, the empirical sample covariance matrix is:
$$S = \frac{1}{T - 1} \sum_{t=1}^T (r_t - \bar{r})(r_t - \bar{r})^T$$

When the ratio $N / T \to c > 0$, the Marchenko-Pastur law dictates that sample eigenvalues spread severely away from population eigenvalues:
* The largest sample eigenvalues are systematically biased upward (overestimating risk).
* The smallest sample eigenvalues are systematically biased downward toward zero (underestimating risk).
* When $N \ge T$, $S$ is rank-deficient and singular, causing downstream mean-variance or quadratic solvers (`cvxpy`) to fail or allocate infinite leverage to noise vectors.

## 2. Ledoit-Wolf Optimal Linear Shrinkage
To regularize the risk estimate without subjective tuning, `src/optimizer.py` implements the **Ledoit-Wolf (2004)** shrinkage framework:
$$\Sigma_{\text{LW}} = (1 - \delta) S + \delta F$$
where:
* $S$ is the empirical sample covariance matrix.
* $F = \mu_F \mathbf{I}$ is the well-conditioned shrinkage target (diagonal equal-variance identity target where $\mu_F = \frac{1}{N}\text{Tr}(S)$).
* $\delta \in [0, 1]$ is the analytically optimal shrinkage intensity minimizing the expected quadratic loss:
$$\delta^* = \frac{\sum_{i=1}^N \text{Var}(s_{ii})}{\|S - F\|_F^2}$$

## 3. Low-Sample Historical Conditioning & Strict PSD Enforcement
During early tenures or sudden CEO transitions where $T < 5$, standard estimators degenerate. `src/optimizer.py` applies a two-tiered conditioning defense:
1. **Fallback Diagonalization**: When $T < 5$, the matrix degenerates safely to pure diagonal constituent variances with $\delta = 1.0$.
2. **Spectral Eigenvalue Flooring**: The matrix is symmetrized $\Sigma = \frac{1}{2}(\Sigma + \Sigma^T)$ and decomposed via spectral eigenvalue decomposition:
$$\Sigma = V \Lambda V^T \quad \text{where} \quad \Lambda = \text{diag}(\lambda_1, \dots, \lambda_N)$$
Any eigenvalue falling below $\epsilon = 10^{-6}$ is floored to $\epsilon$:
$$\tilde{\Lambda} = \text{diag}(\max(\lambda_i, 10^{-6})) \implies \Sigma_{\text{PSD}} = V \tilde{\Lambda} V^T$$
This guarantees that for any portfolio weight vector $w \neq 0$:
$$w^T \Sigma_{\text{PSD}} w > 0$$
ensuring quadratic solvers always satisfy convexity.
