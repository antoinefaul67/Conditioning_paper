import numpy as np
from scipy.stats import norm
from scipy.optimize import bisect
import utils as utl
import tensorflow as tf
import pandas as pd

# Distribution function of Gaussian mixture
def mixture_cdf(q, means, sds, weights):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    means = np.array(means)
    sds = np.array(sds)
    cdf_values = np.zeros_like(q, dtype = float)
    for i in range(len(means)):
        component_cdf = norm.cdf(q, loc = means[i], scale = sds[i])
        cdf_values += weights[i] * component_cdf
    return cdf_values

# Distribution function of truncated Gaussian mixture
def truncated_mixture_cdf(x, means, sds, weights, a=-np.inf, b=np.inf):
    if a > b:
        raise ValueError("Lower limit a must be less than or equal to upper limit b.")
    x = np.asarray(x, dtype=float)
    Fa = 0.0 if np.isneginf(a) else mixture_cdf(a, means, sds, weights)
    Fb = 1.0 if np.isposinf(b) else mixture_cdf(b, means, sds, weights)
    Fx = mixture_cdf(x, means, sds, weights)
    truncated_Fx = (Fx - Fa) / (Fb - Fa)
    truncated_Fx[x < a] = 0.0
    truncated_Fx[x > b] = 1.0
    return truncated_Fx

# Quantiles of Gaussian mixture
# The goal is to use this as little as possible,
# but I think we need it to compute F^{-1}(\alpha_i).
def mixture_ppf(p, means, sds, weights, tol=1e-6, max_iter=100):
    p = np.asarray(p, dtype=float)
    weights = np.array(weights, dtype=float)
    weights /= np.sum(weights)
    
    def mixture_cdf_scalar(x):
        return np.sum(weights * norm.cdf(x, loc=means, scale=sds))
    
    quantiles = np.zeros_like(p, dtype=float)
    for i, prob in enumerate(p):
        if prob <= 0:
            quantiles[i] = -np.inf
        elif prob >= 1:
            quantiles[i] = np.inf
        else:
            root = bisect(lambda x: mixture_cdf_scalar(x) - prob,
                          a=min(means) - 10 * max(sds),
                          b=max(means) + 10 * max(sds),
                          xtol=tol, maxiter=max_iter)
            quantiles[i] = root
    return quantiles

def extract_marginal_gmms(gmcm):
    """
    Return marginal GMM parameters for each marginal in gmcm.marg_dists.
    Each returned entry is a dict:
      { 'dim': j, 'n_components': K, 'weights': array(K,),
        'means': array(K,), 'stds': array(K,) }
    """
    marg_info_list = []
    # gmcm.marg_dists is a list of info_dict created in learn_marginals()
    for j, info in enumerate(gmcm.marg_dists):
        # The GaussianMixture object (from sklearn) was stored under 'marginal'
        gmm = info.get('marginal', None)
        if gmm is None:
            # fallback: try other keys or skip
            raise RuntimeError(f"No 'marginal' GMM object found for marginal {j}")
        # Extract weights, means, covariances
        weights = np.asarray(gmm.weights_, dtype=float).reshape(-1)
        means = np.asarray(gmm.means_, dtype=float).reshape(-1)  # shape (K,)
        covs = np.asarray(gmm.covariances_, dtype=float)
        # Sort weights, means, covs by mean
        idx = np.argsort(means)
        means = means[idx]
        covs = covs[idx]
        weights = weights[idx]
        # covs might be (K,1,1) or (K,) depending on fit; normalize to 1D variance array:
        if covs.ndim == 3 and covs.shape[1] == 1 and covs.shape[2] == 1:
            variances = covs.reshape(-1)
        elif covs.ndim == 2 and covs.shape[1] == 1:
            variances = covs.reshape(-1)
        elif covs.ndim == 1:
            variances = covs
        else:
            # unexpected shape: try to extract diagonal
            variances = np.array([np.atleast_2d(c).diagonal().sum() for c in covs])  # fallback
        stds = np.sqrt(variances)
        # normalize weights just in case
        if weights.sum() != 0:
            weights = weights / weights.sum()
        else:
            # fallback to uniform
            weights = np.ones_like(weights) / weights.size
        marg_info_list.append({
            'dim': j,
            'n_components': means.size,
            'weights': weights,
            'means': means,
            'stds': stds
        })
    return marg_info_list

# Pretty table view
def marg_info_to_df(marg_info_list):
    rows = []
    for info in marg_info_list:
        j = info['dim']
        K = info['n_components']
        for k in range(K):
            rows.append({
                'dim': j,
                'comp': k,
                'weight': float(info['weights'][k]),
                'mean': float(info['means'][k]),
                'std': float(info['stds'][k])
            })
    return pd.DataFrame(rows, columns=['dim','comp','weight','mean','std'])

def transport(latent_data, marg_info_list, latent_weights, mus, covs, quantiles_weights_list = None):
    observed_data = np.empty_like(latent_data)
    n_dim = latent_data.shape[1]
    for d in range(n_dim):
        marginal_distribution = marg_info_list[d]
        observed_weights = marginal_distribution['weights']
        observed_means = marginal_distribution['means']
        observed_sds = marginal_distribution['stds']
        latent_means = mus[:, d]
        latent_sds = np.sqrt(covs[:, d, d])
        quantiles_weights = quantiles_weights_list[d]
        observed_data[:, d] = transport_single_vectorized(latent_data[:, d], latent_means, latent_sds, latent_weights, observed_means, observed_sds, observed_weights, quantiles_weights)
    return observed_data

def obtain_quantiles_weights(gmcm):
    n_dim = gmcm.ndims
    quantiles_weights_list = []
    logits, mus, covs, _ = utl.vec2gmm_params(gmcm.ndims, gmcm.ncomps, gmcm.gmc.params)
    latent_weights = tf.math.softmax(logits).numpy()
    marg_info_list = extract_marginal_gmms(gmcm)
    for d in range(n_dim):
        marginal_distribution = marg_info_list[d]
        observed_weights = np.array(marginal_distribution['weights'])
        observed_weights /= np.sum(observed_weights)
        cum_observed_weights = np.cumsum(observed_weights)
        latent_means = mus[:, d].numpy() if isinstance(mus, tf.Tensor) else mus[:, d]
        latent_sds = np.sqrt(covs[:, d, d].numpy() if isinstance(covs, tf.Tensor) else covs[:, d, d])
        quantiles_weights = mixture_ppf(cum_observed_weights, latent_means, latent_sds, latent_weights)
        quantiles_weights = np.insert(quantiles_weights, 0, -np.inf)
        quantiles_weights_list.append(quantiles_weights)
    return quantiles_weights_list

def transport_single_vectorized(
    latent_data, latent_means, latent_sds, latent_weights,
    observed_means, observed_sds, observed_weights,
    quantiles_weights=None
):
    # Convert observed parameters to arrays
    observed_means = np.asarray(observed_means)
    observed_sds = np.asarray(observed_sds)
    observed_weights = np.asarray(observed_weights)
    observed_weights = observed_weights / np.sum(observed_weights)
    cum_observed_weights = np.cumsum(observed_weights)
    # Compute mixture quantiles once
    if quantiles_weights is None:
        quantiles_weights = mixture_ppf(cum_observed_weights, latent_means, latent_sds, latent_weights)
        quantiles_weights = np.insert(quantiles_weights, 0, -np.inf)
    # Compute latent mixture CDF for all data points
    latent_cdf = mixture_cdf(latent_data, latent_means, latent_sds, latent_weights)
    # Find which interval each data point belongs to
    ell = np.searchsorted(cum_observed_weights, latent_cdf)
    # Precompute Fa and Fb for each interval
    Fa = mixture_cdf(quantiles_weights[:-1], latent_means, latent_sds, latent_weights)
    Fb = mixture_cdf(quantiles_weights[1:], latent_means, latent_sds, latent_weights)
    # Compute truncated mixture CDF vectorized
    Ftr = (latent_cdf - Fa[ell]) / (Fb[ell] - Fa[ell])
    Ftr[latent_cdf < Fa[ell]] = 0
    Ftr[latent_cdf > Fb[ell]] = 1
    # Map to observed mixture using vectorized norm.ppf
    observed_data = observed_means[ell] + observed_sds[ell] * norm.ppf(Ftr)
    return observed_data