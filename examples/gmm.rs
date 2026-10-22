//! Example of a 1-D Gaussian Mixture Model (GMM).

use em_rs::{Model, em};

struct Gmm1 {
    reg: f64,
}

#[derive(Clone)]
struct Gmm1Params {
    pi: Vec<f64>,
    mu: Vec<f64>,
    var: Vec<f64>,
}

struct Gmm1Stats {
    n: Vec<f64>,
    sx: Vec<f64>,
    sxx: Vec<f64>,
}

/// Log-normal PDF
fn log_normal(x: f64, mu: f64, var: f64) -> f64 {
    -0.5 * ((x - mu).powi(2) / var + (2.0 * std::f64::consts::PI * var).ln())
}

fn logsumexp(xs: &[f64]) -> f64 {
    let m = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    m + xs.iter().map(|x| (x - m).exp()).sum::<f64>().ln()
}

impl Model for Gmm1 {
    type Datum = f64;
    type Params = Gmm1Params;
    type Stats = Gmm1Stats;

    fn zero_stats(&self, p: &Gmm1Params) -> Gmm1Stats {
        let k = p.mu.len();
        Gmm1Stats {
            n: vec![0.0; k],
            sx: vec![0.0; k],
            sxx: vec![0.0; k],
        }
    }

    fn accumulate(&self, p: &Gmm1Params, &x: &f64, acc: &mut Gmm1Stats) -> f64 {
        let logs: Vec<f64> = (0..p.mu.len())
            .map(|k| p.pi[k].ln() + log_normal(x, p.mu[k], p.var[k]))
            .collect();
        let z = logsumexp(&logs);
        for (k, l) in logs.iter().enumerate() {
            let r = (l - z).exp();
            acc.n[k] += r;
            acc.sx[k] += r * x;
            acc.sxx[k] += r * x * x;
        }
        z
    }

    fn merge(&self, into: &mut Gmm1Stats, other: Gmm1Stats) {
        for k in 0..into.n.len() {
            into.n[k] += other.n[k];
            into.sx[k] += other.sx[k];
            into.sxx[k] += other.sxx[k];
        }
    }

    fn maximize(&self, s: Gmm1Stats) -> Gmm1Params {
        let total: f64 = s.n.iter().sum();
        let mu: Vec<f64> = s.sx.iter().zip(&s.n).map(|(sx, n)| sx / n).collect();
        let var = (0..mu.len())
            .map(|k| s.sxx[k] / s.n[k] - mu[k] * mu[k] + self.reg)
            .collect();
        let pi = s.n.iter().map(|n| n / total).collect();
        Gmm1Params { pi, mu, var }
    }
}

fn main() {
    let model = Gmm1 { reg: 1e-6 };
    let data = vec![
        1.0, 1.2, 0.8, 1.1, 0.9, // Cluster 1
        5.0, 5.2, 4.8, 5.1, 4.9, // Cluster 2
    ];
    let initial_params = Gmm1Params {
        pi: vec![0.5, 0.5],
        mu: vec![0.0, 6.0],
        var: vec![1.0, 1.0],
    };
    let fit = em(&model, &data, initial_params, 1e-6, 100);
    println!("Converged: {}", fit.converged);
    println!("Final parameters:");
    for (k, ((pi_k, mu_k), var_k)) in fit
        .params
        .pi
        .iter()
        .zip(&fit.params.mu)
        .zip(&fit.params.var)
        .enumerate()
    {
        println!(
            "Component {}: pi = {}, mu = {}, var = {}",
            k, pi_k, mu_k, var_k
        );
    }
}
