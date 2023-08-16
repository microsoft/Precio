// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use super::attr::AttrValueType;
use super::report::Report;
use super::report_handler::ReportHandler;
use crate::schema::AttributeType;
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::cmp::max;

/// A `ReportVector` represents a collection of `Report`s.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportVector<const U32_SIZE: usize> {
    report_handler: ReportHandler<U32_SIZE>,
    reports: Vec<Report<U32_SIZE>>,
}

pub struct ReportVectorIterator<'a, const U32_SIZE: usize> {
    inner: std::slice::Iter<'a, Report<U32_SIZE>>,
}

impl<'a, const U32_SIZE: usize> Iterator for ReportVectorIterator<'a, U32_SIZE> {
    type Item = &'a Report<U32_SIZE>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<'a, const U32_SIZE: usize> ExactSizeIterator for ReportVectorIterator<'a, U32_SIZE> {
    fn len(&self) -> usize {
        self.inner.len()
    }
}

impl<const U32_SIZE: usize> ReportVector<U32_SIZE> {
    /// Iterate over the `Report`s in the `ReportVector`.
    pub fn iter(&self) -> ReportVectorIterator<U32_SIZE> {
        ReportVectorIterator {
            inner: self.reports.iter(),
        }
    }

    /// Create a new `ReportVector`.
    pub fn new(attr_types: &[AttributeType]) -> Self {
        Self {
            report_handler: ReportHandler::new(attr_types),
            reports: Vec::new(),
        }
    }

    /// Create the `ReportVector` data from a random seed.
    pub(crate) fn random<R: Rng + ?Sized>(
        attr_types: &[AttributeType],
        rng: &mut R,
        count: usize,
    ) -> Self {
        let mut report_vector = Self::new(attr_types);
        for _ in 0..count {
            let report = report_vector.report_handler().create_random_report(rng);
            report_vector.push(report);
        }
        report_vector
    }

    /// Add a `Report` to the `ReportVector` multiple times. Panics if the `Report` is invalid.
    pub(crate) fn push_many(&mut self, report: Report<U32_SIZE>, count: usize) {
        assert!(
            self.report_handler().is_valid_report(&report),
            "push_many: The report is invalid."
        );

        for _ in 0..count {
            self.reports.push(report);
        }
    }

    /// Add a `Report` to the `ReportVector`. Panics if the `Report` is invalid.
    pub fn push(&mut self, report: Report<U32_SIZE>) {
        self.push_many(report, 1)
    }

    /// Remove the last `Report` from the `ReportVector`. Return `None` if the `ReportVector` is empty.
    pub fn pop(&mut self) -> Option<Report<U32_SIZE>> {
        self.reports.pop()
    }

    /// Remove the `Report` at the given index. Return `None` if the index is out of bounds.
    pub(crate) fn remove(&mut self, index: usize) -> Option<Report<U32_SIZE>> {
        if index < self.reports.len() {
            Some(self.reports.remove(index))
        } else {
            None
        }
    }

    /// Get the `Report` at the given index. Return `None` if the index is out of bounds.
    pub(crate) fn get(&self, index: usize) -> Option<&Report<U32_SIZE>> {
        self.reports.get(index)
    }

    /// Get the number of `Report`s in the `ReportVector`.
    pub fn len(&self) -> usize {
        self.reports.len()
    }

    /// Get the `ReportHandler` for the `ReportVector`.
    pub fn report_handler(&self) -> &ReportHandler<U32_SIZE> {
        &self.report_handler
    }

    /// Return an iterator to the attributes for every `Report` in the `ReportVector`.
    pub fn get_attr_iter(
        &self,
        attr_index: usize,
    ) -> impl ExactSizeIterator<Item = AttrValueType> + '_ {
        self.reports
            .iter()
            .copied()
            .map(move |report| self.report_handler.get_attr(&report, attr_index))
    }

    /// Set a particular attribute to a given value for every `Report` in the `ReportVector`.
    pub fn set_attr(&mut self, attr_index: usize, attr_value: AttrValueType) {
        self.reports
            .iter_mut()
            .for_each(|report| self.report_handler.set_attr(report, attr_index, attr_value));
    }

    /// Reveal an attribute for every `Report` in the `ReportVector`.
    pub(crate) fn reveal_attr(&mut self, attr_index: usize, attr_shares: Vec<AttrValueType>) {
        assert_eq!(
            self.reports.len(),
            attr_shares.len(),
            "reveal_attr: The number of reports and the number of attribute shares must be equal."
        );

        self.reports
            .iter_mut()
            .zip(attr_shares.into_iter())
            .for_each(|(report, attr_share)| {
                self.report_handler
                    .reveal_attr(report, attr_index, attr_share)
            });
    }

    /// Combine shares of two `ReportVector`s together. Categorical attributes are XORed
    /// and numerical attributes are added modulo their respective moduli.
    pub fn reveal(&mut self, other: Self) {
        assert_eq!(
            self.reports.len(),
            other.reports.len(),
            "reveal: The number of reports in the two report vectors must be equal."
        );
        assert_eq!(
            self.report_handler, other.report_handler,
            "reveal: The report handlers of the two report vectors must be equal."
        );

        self.reports
            .iter_mut()
            .zip(other.reports.into_iter())
            .for_each(|(report, other_report)| {
                self.report_handler.reveal(report, &other_report);
            });
    }

    /// Combine shares of two `ReportVector`s together. Categorical attributes are XORed
    /// and numerical attributes are added modulo their respective moduli. The other
    /// vector is generated from a given `Rng`. Returns the random `Report`s.
    pub(crate) fn reveal_from_random<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
    ) -> ReportVector<U32_SIZE> {
        Self {
            report_handler: self.report_handler.clone(),
            reports: self
                .reports
                .iter_mut()
                .map(|report| self.report_handler.reveal_from_random(rng, report))
                .collect(),
        }
    }

    /// Secret-share a `ReportVector` with random `Report`s generated from a given `Rng`.
    /// Return the random `Report`s. The created reports are subtracted from this vector.
    /// Categorical attributes are XORed and numerical attributes are subtracted modulo
    /// their respective moduli.
    pub fn share<R: Rng + ?Sized>(&mut self, rng: &mut R) -> ReportVector<U32_SIZE> {
        Self {
            report_handler: self.report_handler.clone(),
            reports: self
                .reports
                .iter_mut()
                .map(|report| self.report_handler.share(rng, report))
                .collect(),
        }
    }

    /// Remove every `Report` where the value of a given attribute is either
    /// the dummy value (all ones) or the value appears fewer than a threshold
    /// many times. Returns the number of `Report`s removed.
    pub fn prune(&mut self, attr_index: usize, threshold: usize) -> usize {
        let attr_type = self.report_handler.get_attr_types()[attr_index];
        assert!(
            attr_type.is_categorical(),
            "prune: The attribute must be categorical."
        );

        // Compute the full histogram for this attribute
        let dummy_attr_value = self.report_handler.get_dummy_attr_values()[attr_index];
        let mut histogram = vec![0usize; (dummy_attr_value as usize) + 1];
        self.reports.iter().for_each(|report| {
            histogram[self.report_handler.get_attr(report, attr_index) as usize] += 1;
        });

        let old_len = self.reports.len();
        self.reports.retain(|report| {
            let attr_value = self.report_handler.get_attr(report, attr_index);
            let attr_count = histogram[attr_value as usize];
            attr_value != dummy_attr_value && attr_count >= max::<usize>(threshold, 1)
        });
        let new_len = self.reports.len();

        old_len - new_len
    }

    /// Remove an attribute from every `Report` in the `ReportVector`.
    pub fn remove_attr(&mut self, attr_index: usize) {
        // First set all of the attribute values to zero.
        self.reports
            .iter_mut()
            .for_each(|report| self.report_handler.clear_attr(report, attr_index));

        // Next, update the `ReportHandler` to reflect the removal of the attribute.
        self.report_handler.remove_attr(attr_index);
    }

    /// Split off all reports with the given categorical attribute value.
    pub fn split_at(&mut self, attr_index: usize, attr_value: AttrValueType) -> Self {
        let attr_type = self.report_handler.get_attr_types()[attr_index];
        assert!(
            attr_type.is_categorical(),
            "prune: The attribute must be categorical."
        );

        // Separate all reports with the given attribute value.
        let mut split_reports = Vec::<Report<U32_SIZE>>::new();
        self.reports.retain(|report| {
            let report_attr_value = self.report_handler.get_attr(report, attr_index);
            if report_attr_value == attr_value {
                split_reports.push(*report);
                false
            } else {
                true
            }
        });

        // Create a new `ReportVector` with the split-off reports.
        let mut split_report_vector = Self {
            report_handler: self.report_handler.clone(),
            reports: split_reports,
        };

        // Delete the designated attribute.
        split_report_vector.remove_attr(attr_index);

        split_report_vector
    }

    /// Permute the `Report`s in this `ReportVector`.
    pub(crate) fn permute<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.reports.as_mut_slice().shuffle(rng);
    }
}

pub mod test_distr {
    use super::*;
    use crate::random::zipf::ZipfDistribution;
    use rand_distr::{Distribution, Normal};

    impl<const U32_SIZE: usize> ReportVector<U32_SIZE> {
        /// Adds many `Report`s to the `ReportVector` with attributes sampled from a Zipf distribution.
        pub fn push_many_zipf<R: Rng + Default>(
            &mut self,
            count: usize,
            s: f64,
            randomize_zipf: bool,
        ) {
            // Sample `count` many values for each attribute.
            let dummy_attr_values = self.report_handler.get_dummy_attr_values();
            let attr_types = self.report_handler.get_attr_types();

            let mut zipf_samples: Vec<Vec<AttrValueType>> = dummy_attr_values
                .iter()
                .zip(attr_types)
                .map(|(dummy_attr_value, attr_type)| {
                    let zipf_max = if attr_type.is_categorical() {
                        // For categorical attributes, we sample up to the dummy value minus one.
                        *dummy_attr_value - 1
                    } else {
                        // For numerical attributes, we sample from the entire range for that attribute
                        // divided by two, so that we don't overflow the modulus when multiplying by two.
                        (attr_type.get_modulus() - 1) / 2
                    };
                    let zipf = ZipfDistribution::new(zipf_max, s, randomize_zipf).unwrap();
                    zipf.sample_iter(R::default())
                        .map(|z| z as AttrValueType)
                        .take(count)
                        .collect::<Vec<AttrValueType>>()
                })
                .collect::<Vec<_>>();

            // Now just create `Report`s from the samples, removing them from zipf_samples at the same time.
            (0..count).for_each(|_| {
                let attr_values: Vec<AttrValueType> = zipf_samples
                    .iter_mut()
                    .map(|attr_samples| attr_samples.pop().unwrap())
                    .collect();

                let report = self.report_handler.create_report(&attr_values);
                self.reports.push(report);
            });
        }

        /// Adds many `Report`s to the `ReportVector` with attributes sampled from a Gaussian distribution
        /// centered at the middle of the range for each attribute.
        pub fn push_many_gaussian<R: Rng + Default>(&mut self, count: usize, s: f64) {
            // Sample `count` many values for each attribute.
            let dummy_attr_values = self.report_handler.get_dummy_attr_values();
            let attr_types = self.report_handler.get_attr_types();

            let mut gaussian_samples: Vec<Vec<AttrValueType>> = dummy_attr_values
                .iter()
                .zip(attr_types)
                .map(|(dummy_attr_value, attr_type)| {
                    let value_max = if attr_type.is_categorical() {
                        // For categorical attributes, we sample up to the dummy value minus one.
                        *dummy_attr_value - 1
                    } else {
                        // For numerical attributes, we sample from the entire range for that attribute
                        // divided by two, so that we don't overflow the modulus when multiplying by two.
                        (attr_type.get_modulus() - 1) / 2
                    } as f64;
                    let mean = value_max / 2.0;
                    let gaussian = Normal::new(mean as f64, s).unwrap();
                    gaussian
                        .sample_iter(R::default())
                        .map(|sample| sample.clamp(0.0, value_max).round() as AttrValueType)
                        .take(count)
                        .collect::<Vec<AttrValueType>>()
                })
                .collect::<Vec<_>>();

            // Now just create `Report`s from the samples, removing them from gaussian_samples at the same time.
            (0..count).for_each(|_| {
                let attr_values: Vec<AttrValueType> = gaussian_samples
                    .iter_mut()
                    .map(|attr_samples| attr_samples.pop().unwrap())
                    .collect();

                let report = self.report_handler.create_report(&attr_values);
                self.reports.push(report);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode;
    use rand::rngs::OsRng;
    use AttributeType::*;

    #[test]
    fn test_report_vector_new() {
        let report_vector = ReportVector::<1>::new(&[C2, C3, C4, C5]);
        assert_eq!(report_vector.len(), 0);
        assert_eq!(
            report_vector.report_handler().get_attr_sizes(),
            &[2, 3, 4, 5]
        );

        let report_vector = ReportVector::<1>::new(&[C2, C3, N4(10), N5(30)]);
        assert_eq!(report_vector.len(), 0);
        assert_eq!(
            report_vector.report_handler().get_attr_sizes(),
            &[2, 3, 4, 5]
        );
    }

    #[test]
    fn test_report_vector_push_pop_remove() {
        let mut rng = OsRng;

        let mut report_vector = ReportVector::<1>::new(&[C2, C3, C4, C5]);
        let report0 = report_vector
            .report_handler()
            .create_random_report(&mut rng);
        report_vector.push(report0);
        assert_eq!(report_vector.len(), 1);
        assert_eq!(*report_vector.get(0).unwrap(), report0);

        let report1 = report_vector
            .report_handler()
            .create_random_report(&mut rng);
        report_vector.push(report1);
        assert_eq!(report_vector.len(), 2);
        assert_eq!(*report_vector.get(1).unwrap(), report1);

        let report = report_vector.pop().unwrap();
        assert_eq!(report_vector.len(), 1);
        assert_eq!(report, report1);

        let report = report_vector.remove(0).unwrap();
        assert_eq!(report_vector.len(), 0);
        assert_eq!(report, report0);

        assert_eq!(report_vector.remove(0), None);
        assert_eq!(report_vector.pop(), None);

        let mut report_vector = ReportVector::<1>::new(&[N3(5), N4(13), N5(31), C5]);
        let report0 = report_vector
            .report_handler()
            .create_random_report(&mut rng);
        report_vector.push(report0);
        assert_eq!(report_vector.len(), 1);
        assert_eq!(*report_vector.get(0).unwrap(), report0);

        let report1 = report_vector
            .report_handler()
            .create_random_report(&mut rng);
        report_vector.push(report1);
        assert_eq!(report_vector.len(), 2);
        assert_eq!(*report_vector.get(1).unwrap(), report1);

        let report = report_vector.pop().unwrap();
        assert_eq!(report_vector.len(), 1);
        assert_eq!(report, report1);

        let report = report_vector.remove(0).unwrap();
        assert_eq!(report_vector.len(), 0);
        assert_eq!(report, report0);

        assert_eq!(report_vector.remove(0), None);
        assert_eq!(report_vector.pop(), None);
    }

    #[test]
    fn test_serialize_deserialize() {
        use AttributeType::*;

        let mut rng = OsRng;

        let mut report_vector =
            ReportVector::<3>::new(&[C10, C20, C10, N20(65537), N10(1023), C20]);
        report_vector.push(
            report_vector
                .report_handler()
                .create_random_report(&mut rng),
        );
        report_vector.push(
            report_vector
                .report_handler()
                .create_random_report(&mut rng),
        );
        report_vector.push(
            report_vector
                .report_handler()
                .create_random_report(&mut rng),
        );
        report_vector.push(
            report_vector
                .report_handler()
                .create_random_report(&mut rng),
        );
        report_vector.push(
            report_vector
                .report_handler()
                .create_random_report(&mut rng),
        );

        let encoded = bincode::serialize(&report_vector).unwrap();
        let report_vector_copy: ReportVector<3> = bincode::deserialize(&encoded).unwrap();

        assert_eq!(report_vector_copy, report_vector);
    }

    #[test]
    fn test_share_reveal() {
        use AttributeType::*;

        // Populate a `ReportVector` with 100 `Report`s.
        let mut rng = OsRng;
        let mut report_vector = ReportVector::<1>::new(&[C2, N3(6), C4, C5]);
        for _ in 0..100 {
            let random_report = report_vector
                .report_handler()
                .create_random_report(&mut rng);
            report_vector.push(random_report);
        }
        let report_vector_copy = report_vector.clone();

        // Share the `ReportVector`.
        let report_vector_shares = report_vector.share(&mut rng);
        assert_ne!(report_vector_copy, report_vector);

        // Reveal the `ReportVector`.
        report_vector.reveal(report_vector_shares);

        // Check that the `ReportVector` is empty.
        assert_eq!(report_vector, report_vector_copy);
    }

    #[test]
    fn test_prune() {
        use AttributeType::*;

        // Populate a `ReportVector` with 1000 `Report`s.
        let mut rng = OsRng;
        let mut report_vector = ReportVector::<1>::new(&[C2, N3(6), C4, C5]);
        for _ in 0..1000 {
            let random_report = report_vector
                .report_handler()
                .create_random_report(&mut rng);
            report_vector.push(random_report);
        }

        // Prune the first attribute
        let removed = report_vector.prune(0, 10);

        // This should remove only the dummy bucket, so approximately a third of the values.
        assert_ne!(report_vector.len(), 1000);
        assert!(200 < removed && removed < 300);

        // Prune the last attribute. There are so many values that this will prune everything.
        report_vector.prune(3, 100);
        assert_eq!(report_vector.len(), 0);
    }
}
