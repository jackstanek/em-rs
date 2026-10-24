//! Example of using the off-the-shelf Gaussian mixture model.

use em_rs::models::gmm::*;

fn main() {
    let model = Gmm1::default();
    let data = vec![
        1.0, 1.2, 0.8, 1.1, 0.9, // Cluster 1
        5.0, 5.2, 4.8, 5.1, 4.9, // Cluster 2
    ];
    let initial_params = Gmm1Params::from_iters(vec![0.5, 0.5], vec![0.0, 6.0], vec![1.0, 1.0]);
    let fit = em_rs::em(&model, &data, initial_params, 1e-6, 100);
    println!("Converged: {}", fit.converged);
    println!("Final parameters:");
    for (k, (pi_k, mu_k, var_k)) in fit.params.iter().enumerate() {
        println!(
            "Component {}: pi = {}, mu = {}, var = {}",
            k, pi_k, mu_k, var_k
        );
    }
}
