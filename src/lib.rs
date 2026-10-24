//! Generic implementation of expectation-maximization (EM), well-suited to
//! multiple different latent-variable models such as Gaussian mixture models
//! (GMMs), hidden Markov models (HMMs), probabilistic context-free grammars
//! (PCFGs), and more.

pub mod models;

/// A model which uses the expectation-maximization (EM) algorithm to fit
/// parameters.
pub trait Model {
    /// The type of the data that this model generates.
    type Datum;
    /// The type of the parameters that the model will fit.
    type Params;
    /// The type of the sufficient statistics for the model
    type Stats;

    /// Identity of element of the statistics monoid, shaped to the parameters.
    fn zero_stats(&self, params: &Self::Params) -> Self::Stats;

    /// E-step for a single datum, which adds its sufficient statistics into
    /// `acc` and return `log p(x | params)`.
    fn accumulate(&self, params: &Self::Params, datum: &Self::Datum, acc: &mut Self::Stats) -> f64;

    fn merge(&self, into: &mut Self::Stats, from: Self::Stats);

    /// M-step: argmax of Q given the aggregate statistics.
    fn maximize(&self, stats: Self::Stats) -> Self::Params;
}

pub struct Fit<P> {
    /// The fit parameters of the model
    pub params: P,
    /// Log-likelihoods at each point in the EM algorithm, including the initial log-likelihood
    pub log_lik: Vec<f64>,
    /// Whether the EM algorithm converged within the specified tolerance and maximum iterations
    pub converged: bool,
}

/// Fit the parameters with the expectation-maximization (EM) algorithm,
/// returning the final parameters, the log-likelihoods at each iteration, and
/// whether the algorithm converged.
pub fn em<M>(
    model: &M,
    data: &[M::Datum],
    mut params: M::Params,
    tol: f64,
    max_iter: usize,
) -> Fit<M::Params>
where
    M: Model,
{
    let mut trace = Vec::with_capacity(max_iter);
    for _ in 0..max_iter {
        let mut stats = model.zero_stats(&params);
        let log_lik: f64 = data
            .iter()
            .map(|datum| model.accumulate(&params, datum, &mut stats))
            .sum();

        // sanity check
        let delta = trace.last().map(|prev| log_lik - prev);
        debug_assert!(
            delta.map_or(true, |d| d > -1e-8),
            "EM decreased log-likelihood. This is probably due to a bug in the model's implementation of `accumulate`."
        );
        trace.push(log_lik);
        params = model.maximize(stats);
        if delta.is_some_and(|d| d.abs() < tol) {
            return Fit {
                params,
                log_lik: trace,
                converged: true,
            };
        }
    }
    Fit {
        params,
        log_lik: trace,
        converged: false,
    }
}

/// Fit the parameters with the expectation-maximization (EM) algorithm,
/// returning the final parameters, the log-likelihoods at each iteration, and
/// whether the algorithm converged.
///
/// This function runs the E-step in parallel using [`rayon`] and is only
/// available when the `rayon` feature is active.
#[cfg(feature = "rayon")]
pub fn par_em<M>(
    model: &M,
    data: &[M::Datum],
    mut params: M::Params,
    tol: f64,
    max_iter: usize,
) -> Fit<M::Params>
where
    M: Model + Sync,
    M::Datum: Sync,
    M::Params: Sync,
    M::Stats: Send,
{
    use rayon::prelude::*;

    let mut trace = Vec::with_capacity(max_iter);
    for _ in 0..max_iter {
        let (stats, log_lik) = data
            .par_iter()
            .map(|datum| {
                let mut local = model.zero_stats(&params);
                let ll = model.accumulate(&params, datum, &mut local);
                (local, ll)
            })
            .reduce(
                || (model.zero_stats(&params), 0.0),
                |(mut acc, ll), (local, local_ll)| {
                    model.merge(&mut acc, local);
                    (acc, ll + local_ll)
                },
            );

        // sanity check
        let delta = trace.last().map(|prev| log_lik - prev);
        debug_assert!(
            delta.map_or(true, |d| d > -1e-8),
            "EM decreased log-likelihood. This is probably due to a bug in the model's implementation of `accumulate`."
        );
        trace.push(log_lik);
        params = model.maximize(stats);
        if delta.is_some_and(|d| d.abs() < tol) {
            return Fit {
                params,
                log_lik: trace,
                converged: true,
            };
        }
    }
    Fit {
        params,
        log_lik: trace,
        converged: false,
    }
}
