# `em-rs`

This library contains a simple, generic implementation of the
expectation-maximization (EM) algorithm in Rust. The EM algorithm is a powerful
statistical technique used for parameter estimation in probabilistic models,
particularly when dealing with incomplete or missing data, or models with
unknown latents such as mixture models or probabilistic automata.

Currently, you need to bring your own model implementation, though the `em()`
function handles parameter fitting. Some simple examples, such as a Bernoulli
mixture and a Gaussian mixture model, are provided in the `examples` folder.

## Installation

Currently you may install the library through git:

```
$ cargo add --git 'https://github.com/jackstanek/em-rs.git'
```

## Features

Not implemented yet:
- [ ] `rayon` (multi-threaded E-step)
- [ ] Built-in model classes (GMMs, HMMs, PCFGs...)
