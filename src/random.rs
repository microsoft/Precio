// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use num_traits::Zero;
use rand::Rng;
use rand_distr::{Distribution, Uniform};

pub(crate) trait RandomMod: Zero {
    /// Create a random value modulo the modulus.
    fn random_mod<R: Rng + ?Sized>(&self, rng: &mut R) -> Self;
}

macro_rules! random_mod_impl {
    ( $( $t:ty ),* ) => {
        $(
            impl RandomMod for $t {
                fn random_mod<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
                    let uniform = Uniform::new(Self::zero(), self);
                    uniform.sample(rng)
                }
            }
        )*
    }
}

random_mod_impl!(u8, u16, u32, u64, u128, usize);

pub(crate) mod prf {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    /// A 256-bit PRF key object.
    pub type PRFKey = [u8; 32];

    /// A 256-bit PRF value.
    pub type PRFValue = [u8; 32];

    /// Hmac type instantiated from Sha256.
    type HmacType = Hmac<Sha256>;

    /// A pseudo-random function object.
    #[derive(Clone)]
    pub struct PRF(HmacType);

    impl PRF {
        /// Create a new PRF object from the given key.
        pub fn new(key: &PRFKey) -> Self {
            PRF(HmacType::new_from_slice(key).unwrap())
        }

        /// A function to evaluate the PRF on a given input.
        pub fn eval(&self, input: &[u8]) -> [u8; 32] {
            // Clone the PRF and compute the value.
            let mut prf_clone = self.clone();
            prf_clone.0.update(input);
            let result = prf_clone.0.finalize();
            result.into_bytes().into()
        }
    }
}

pub mod hist_noise {
    use super::*;
    use rand_distr::{Distribution, Exp, Normal};

    pub trait NoiseDistribution: Distribution<usize> + Copy {
        /// Returns the shift parameter m.
        fn m(&self) -> i64;

        /// Sample many values.
        fn sample_n<R: Rng + ?Sized>(&self, rng: &mut R, n: usize) -> Vec<usize> {
            let mut samples = Vec::with_capacity(n);
            for _ in 0..n {
                samples.push(self.sample(rng));
            }
            samples
        }
    }

    /// A truncated shifted Laplace distribution.
    #[derive(Clone, Copy)]
    pub struct Laplace {
        /// The shift parameter m (non-positive).
        m: i64,

        /// A float version of the shift parameter m (non-positive).
        m_f64: f64,

        /// The scale parameter b (positive).
        b: f64,

        /// Exponential distribution.
        exp: Exp<f64>,
    }

    impl Laplace {
        /// Create a new Laplace noise distribution with scale parameter b.
        pub fn new(m: i64, b: f64) -> Result<Self, String> {
            if m <= 0 {
                if b > 0.0 {
                    let exp = Exp::new(1.0 / b).map_err(|e| e.to_string())?;
                    Ok(Laplace {
                        m,
                        m_f64: m as f64,
                        b,
                        exp,
                    })
                } else {
                    Err("Laplace scale parameter b must be positive.".to_string())
                }
            } else {
                Err("Laplace shift parameter m must be non-positive.".to_string())
            }
        }
    }

    impl Distribution<usize> for Laplace {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
            let mut laplace_sample;
            loop {
                laplace_sample = self.exp.sample(rng) - self.exp.sample(rng);
                if laplace_sample >= self.m_f64 {
                    break;
                }
            }

            // Shift the sample.
            (laplace_sample - self.m_f64).round() as usize
        }
    }

    impl NoiseDistribution for Laplace {
        fn m(&self) -> i64 {
            self.m
        }
    }

    /// Gaussian noise distribution.
    #[derive(Clone, Copy)]
    pub struct Gaussian {
        /// The shift parameter m (non-positive).
        m: i64,

        /// A float version of the shift parameter m (non-positive).
        m_f64: f64,

        /// The standard deviation parameter s (positive).
        s: f64,

        /// Normal distribution.
        normal: Normal<f64>,
    }

    impl Gaussian {
        /// Create a new Gaussian noise distribution with standard deviation parameter s.
        pub fn new(m: i64, s: f64) -> Result<Self, String> {
            if m <= 0 {
                if s > 0.0 {
                    let normal = Normal::new(0.0, s).map_err(|e| e.to_string())?;
                    Ok(Gaussian {
                        m,
                        m_f64: m as f64,
                        s,
                        normal,
                    })
                } else {
                    Err("Gaussian standard deviation parameter s must be positive.".to_string())
                }
            } else {
                Err("Gaussian shift parameter m must be non-positive.".to_string())
            }
        }
    }

    impl Distribution<usize> for Gaussian {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
            let mut gaussian_sample = self.normal.sample(rng);
            while gaussian_sample < self.m_f64 {
                gaussian_sample = self.normal.sample(rng);
            }

            // Shift the sample.
            (gaussian_sample - self.m_f64).round() as usize
        }
    }

    impl NoiseDistribution for Gaussian {
        fn m(&self) -> i64 {
            self.m
        }
    }
}

pub(crate) mod zipf {
    use super::*;
    use ::zipf::ZipfDistribution as RegularZipfDistribution;
    use rand::seq::SliceRandom;
    use rand_distr::Distribution;

    pub(crate) struct ZipfDistribution {
        /// Permutation in the case of a randomized range.
        range_permutation: Vec<usize>,

        /// The Zipf distribution.
        zipf: RegularZipfDistribution,
    }

    impl ZipfDistribution {
        /// Create a new Zipf distribution with parameter s for the interval [0, max_value].
        pub fn new(max_value: u32, s: f64, randomize_range: bool) -> Result<Self, ()> {
            let zipf = RegularZipfDistribution::new(max_value as usize + 1, s)?;
            let mut range_permutation = (0..max_value as usize + 1).collect::<Vec<_>>();

            if randomize_range {
                range_permutation.shuffle(&mut rand::thread_rng());
            }

            Ok(ZipfDistribution {
                range_permutation,
                zipf,
            })
        }
    }

    impl Distribution<usize> for ZipfDistribution {
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
            self.range_permutation[self.zipf.sample(rng) - 1]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::hist_noise::*;
    use super::prf::*;
    use super::*;
    use rand::rngs::OsRng;
    use rand::RngCore;
    use rand_distr::Distribution;

    #[test]
    fn test_prf_eval() {
        let mut rng = OsRng;
        let mut key: PRFKey = [0; 32];
        rng.fill_bytes(&mut key);
        let prf = PRF::new(&key);

        let input = b"test input";
        let output = prf.eval(input);

        assert_eq!(output.len(), 32);
    }

    #[test]
    fn test_prf_eval_different_inputs() {
        let mut rng = OsRng;
        let mut key: PRFKey = [0; 32];
        rng.fill_bytes(&mut key);
        let prf = PRF::new(&key);

        let input1 = b"test input 1";
        let input2 = b"test input 2";
        let input3 = b"test input 2";
        let output1 = prf.eval(input1);
        let output2 = prf.eval(input2);
        let output3 = prf.eval(input3);

        assert_ne!(output1, output2);
        assert_eq!(output2, output3);
    }

    #[test]
    fn test_prf_different_keys() {
        let mut rng = OsRng;
        let mut key1: PRFKey = [0; 32];
        rng.fill_bytes(&mut key1);
        let mut key2: PRFKey = [0; 32];
        rng.fill_bytes(&mut key2);
        let prf1 = PRF::new(&key1);
        let prf2 = PRF::new(&key2);

        let input = b"test input";
        let output1 = prf1.eval(input);
        let output2 = prf2.eval(input);

        assert_ne!(output1, output2);
    }

    #[test]
    fn test_sample_truncated_shifted_gaussian() {
        let mut rng = OsRng;
        let m = -50;
        let s = 8.0;
        let gaussian = Gaussian::new(m, s).unwrap();
        let mut attempts = 0;
        for _ in 0..10 {
            let sample1 = gaussian.sample(&mut rng);
            let sample2 = gaussian.sample(&mut rng);
            if sample1 == sample2 {
                attempts += 1;
                if attempts > 100 {
                    panic!("Failed to sample different values.");
                }
            } else {
                return;
            }
        }
    }

    #[test]
    fn test_sample_truncated_shifted_gaussian_array() {
        let mut rng = OsRng;

        let m = -100;
        let s = 8.0;
        let gaussian = Gaussian::new(m, s).unwrap();
        let mut attempts = 0;
        let samples1 = gaussian.sample_iter(&mut rng).take(10).collect::<Vec<_>>();
        let samples2 = gaussian.sample_iter(&mut rng).take(10).collect::<Vec<_>>();
        samples1
            .iter()
            .zip(samples2.iter())
            .for_each(|(sample1, sample2)| {
                if sample1 == sample2 {
                    attempts += 1;
                    if attempts > 100 {
                        panic!("Failed to sample different vectors.");
                    }
                } else {
                    return;
                }
            });
    }

    #[test]
    fn test_sample_truncated_shifted_laplace() {
        let mut rng = OsRng;
        let m = -50;
        let b = 8.0;
        let laplace = Laplace::new(m, b).unwrap();
        let mut attempts = 0;
        for _ in 0..10 {
            let sample1 = laplace.sample(&mut rng);
            let sample2 = laplace.sample(&mut rng);
            if sample1 == sample2 {
                attempts += 1;
                if attempts > 100 {
                    panic!("Failed to sample different values.");
                }
            } else {
                return;
            }
        }
    }

    #[test]
    fn test_sample_truncated_shifted_laplace_array() {
        let mut rng = OsRng;

        let m = -100;
        let b = 8.0;
        let laplace = Laplace::new(m, b).unwrap();
        let mut attempts = 0;
        let samples1 = laplace.sample_iter(&mut rng).take(10).collect::<Vec<_>>();
        let samples2 = laplace.sample_iter(&mut rng).take(10).collect::<Vec<_>>();
        samples1
            .iter()
            .zip(samples2.iter())
            .for_each(|(sample1, sample2)| {
                if sample1 == sample2 {
                    attempts += 1;
                    if attempts > 100 {
                        panic!("Failed to sample different vectors.");
                    }
                } else {
                    return;
                }
            });
    }

    #[test]
    fn test_zipf_array_sampling() {
        let mut rng = OsRng;
        let zipf = zipf::ZipfDistribution::new(100, 1.0, false).unwrap();
        let samples = zipf.sample_iter(&mut rng).take(10000).collect::<Vec<_>>();
        assert_eq!(samples.len(), 10000);
        assert_eq!(samples.iter().max(), Some(&100));
        assert_eq!(samples.iter().min(), Some(&0));
    }
}
