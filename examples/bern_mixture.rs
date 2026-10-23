//! Bernoulli mixture model: coin flips come from two separate coins with
//! different biases. The EM algorithm can be used to estimate the biases of the
//! two coins and the mixing proportion.

use em_rs::{em, Model};

struct BernMix {}

struct BernMixParams {
    pi: Vec<f64>,
    biases: Vec<f64>,
}

struct BernMixStats {
    n: Vec<f64>,
    sx: Vec<f64>,
    sf: Vec<f64>,
}

fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

impl Model for BernMix {
    type Datum = Vec<bool>;
    type Params = BernMixParams;
    type Stats = BernMixStats;

    fn zero_stats(&self, params: &Self::Params) -> Self::Stats {
        let k = params.biases.len();
        BernMixStats {
            n: vec![0.0; k],
            sx: vec![0.0; k],
            sf: vec![0.0; k],
        }
    }

    fn accumulate(&self, params: &Self::Params, datum: &Self::Datum, acc: &mut Self::Stats) -> f64 {
        let logs: Vec<f64> = (0..params.biases.len())
            .map(|k| {
                let bias = params.biases[k];
                let log_prior = params.pi[k].ln();
                log_prior
                    + datum
                        .iter()
                        .map(|&x| if x { bias.ln() } else { (1.0 - bias).ln() })
                        .sum::<f64>()
            })
            .collect();
        let z = logsumexp(&logs);
        for (k, l) in logs.into_iter().enumerate() {
            let r = (l - z).exp();
            acc.n[k] += r;
            acc.sx[k] += r * datum.iter().filter(|&&x| x).count() as f64;
            acc.sf[k] += r * datum.len() as f64;
        }
        z
    }

    fn merge(&self, _into: &mut Self::Stats, _from: Self::Stats) {}

    fn maximize(&self, stats: Self::Stats) -> Self::Params {
        let total: f64 = stats.n.iter().sum();
        let pi = stats.n.iter().map(|&n| n / total).collect();
        let biases: Vec<f64> = stats
            .sx
            .iter()
            .zip(stats.sf.iter())
            .map(|(&sx, &sf)| if sf > 0.0 { sx / sf } else { 0.5 })
            .collect();
        BernMixParams { pi, biases }
    }
}

fn main() {
    let data = vec![
        vec![true, true, false, true, false],    // Coin 1
        vec![false, false, true, false, false],  // Coin 2
        vec![true, true, true, true, false],     // Coin 1
        vec![false, false, false, false, false], // Coin 2
    ];
    let initial_params = BernMixParams {
        pi: vec![0.5, 0.5],
        biases: vec![0.3, 0.7],
    };
    let fit = em(&BernMix {}, &data, initial_params, 1e-6, 1000);
    println!("Converged: {}", fit.converged);
    println!("Final parameters:");
    for (k, (pi_k, b)) in fit.params.pi.iter().zip(&fit.params.biases).enumerate() {
        println!("Coin {}: pi = {:.4}, bias = {:.4}", k + 1, pi_k, b);
    }
}
