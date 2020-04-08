functions {
    matrix L_cov_exp_quad_ARD(vector[] x,
                              real alpha,
                              vector rho,
                              real delta) {
      int N = size(x);
      matrix[N, N] K;
      real neg_half = -0.5;
      real sq_alpha = square(alpha);
      for (i in 1:(N-1)) {
        K[i, i] = sq_alpha + delta;
        for (j in (i + 1):N) {
          K[i, j] = sq_alpha * exp(neg_half *
                                   dot_self((x[i] - x[j]) ./ rho));
          K[j, i] = K[i, j];
        }
      }
      K[N, N] = sq_alpha + delta;
      return cholesky_decompose(K);
    }
}
data {
    int<lower=0> N;
    int<lower=1> d;
    int<lower=0> N_comp;
    int<lower=0> Nf;
    int yi[N];
    vector[N] y;
    int y_comp1[N_comp];
    int y_comp2[N_comp];
    vector[d] x[Nf];
    real<lower=0> alpha;
    vector<lower=0>[d] rho;
    real<lower=0> sigma;
    real<lower=0> delta;
}
transformed data {
    int ones[N_comp];
    for (i in 1:N_comp) //We cannot use rep_vector since it doesn't support integers
      ones[i] = 1;
}
parameters {
    vector[Nf] eta;
}
transformed parameters {
    vector[Nf] f;
    matrix[Nf, Nf] L_K = L_cov_exp_quad_ARD(x, alpha, rho, delta);
    f = L_K * eta;
}
model {
    eta ~ normal(0,1);
    y ~ normal(f[yi], sigma);
    for (i in 1:N_comp) {
      ones[i] ~ bernoulli(Phi((f[y_comp2[i]]-f[y_comp1[i]])/(sqrt(2)*sigma)));
    }
}
