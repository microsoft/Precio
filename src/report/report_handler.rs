// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use crate::arith::Modulus;
use crate::report::attr::{random_attr_value, AttrValueType, MAX_ATTR_BIT_SIZE};
use crate::report::report::Report;
use crate::schema::{AttributeType, Schema};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::mem::size_of;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportHandler<const U32_SIZE: usize> {
    attr_types: Vec<AttributeType>,
    attr_sizes: Vec<usize>,
    attr_offsets: Vec<usize>,
    attr_filters: Vec<Report<U32_SIZE>>,
    report_filter: Report<U32_SIZE>,
    report_filter_cat_only: Report<U32_SIZE>,
}

impl<const U32_SIZE: usize> ReportHandler<U32_SIZE> {
    fn compute_attr_start_positions(attr_sizes: &[usize]) -> Vec<usize> {
        // This function assumes the inputs are already validated.
        let mut attr_offsets = Vec::new();
        let mut attr_offset = 0;
        for attr_size in attr_sizes {
            attr_offsets.push(attr_offset);
            attr_offset += attr_size;
        }

        attr_offsets
    }

    fn make_attr_filter(bit_size: usize, start_pos: usize) -> Report<U32_SIZE> {
        // This function assumes the inputs are already validated.
        let filter_attr = if bit_size != 0 {
            !0u32 >> (MAX_ATTR_BIT_SIZE - bit_size)
        } else {
            0
        };

        let filter = Report::<U32_SIZE>::from_u32(filter_attr);
        filter << start_pos
    }

    fn make_report_filter(attr_filters: &[Report<U32_SIZE>]) -> Report<U32_SIZE> {
        // OR together all the attribute filters to get the report filter.
        attr_filters
            .iter()
            .fold(Report::<U32_SIZE>::new(), |acc, filter| acc | *filter)
    }

    fn make_report_filter_cat_only(
        attr_types: &[AttributeType],
        attr_filters: &[Report<U32_SIZE>],
    ) -> Report<U32_SIZE> {
        // OR together all the categorical attribute filters to get the report filter.
        attr_types
            .iter()
            .zip(attr_filters)
            .filter(|(&at, _)| at.is_categorical())
            .fold(Report::<U32_SIZE>::new(), |acc, (_, &filter)| acc | filter)
    }

    /// Return whether a given `Schema` is compatible with the `ReportHandler`.
    pub(crate) fn is_compatible_schema(&self, schema: &Schema) -> bool {
        schema.is_compatible_attr_type_array(&self.attr_types)
    }

    /// Return whether a given vector of attribute sizes is valid.
    fn is_valid_attr_sizes(attr_sizes: &[usize]) -> bool {
        // None of the attributes can be less than 2 bits or more than MAX_ATTR_BIT_SIZE bits.
        if attr_sizes
            .iter()
            .any(|&attr_size| attr_size < 2 || attr_size > MAX_ATTR_BIT_SIZE)
        {
            return false;
        }

        // Check that the attribute sizes add up to at most the size of the report
        if attr_sizes.iter().sum::<usize>() > U32_SIZE * size_of::<u32>() * 8 {
            return false;
        }

        true
    }

    /// Return whether a given vector of `AttributeType`s is valid.
    pub(crate) fn is_valid_attr_types(attr_types: &[AttributeType]) -> bool {
        attr_types.iter().all(|&attr_type| attr_type.is_valid())
    }

    pub(crate) fn new(attr_types: &[AttributeType]) -> Self {
        let attr_sizes = attr_types
            .iter()
            .map(|attr_type| attr_type.get_size())
            .collect::<Vec<_>>();

        // Panic if the attribute types are invalid.
        assert!(
            Self::is_valid_attr_types(&attr_types),
            "ReportHandler::new: Invalid attribute types."
        );

        // Panic if the attribute sizes are invalid.
        assert!(
            Self::is_valid_attr_sizes(&attr_sizes),
            "ReportHandler::new: Invalid attribute sizes."
        );

        // Compute attribute start positions and create all filters
        let attr_offsets = Self::compute_attr_start_positions(&attr_sizes);
        let attr_filters = attr_sizes
            .iter()
            .zip(&attr_offsets)
            .map(|(attr_size, start_pos)| Self::make_attr_filter(*attr_size, *start_pos))
            .collect::<Vec<_>>();

        // The report_filter is the OR of all attribute filters.
        let report_filter = Self::make_report_filter(&attr_filters);
        let report_filter_cat_only = Self::make_report_filter_cat_only(&attr_types, &attr_filters);

        let report_handler = ReportHandler::<U32_SIZE> {
            attr_types: attr_types.to_vec(),
            attr_sizes: attr_sizes.to_vec(),
            attr_offsets,
            attr_filters,
            report_filter,
            report_filter_cat_only,
        };

        report_handler
    }

    /// Return the sizes of the attributes.
    pub(crate) fn get_attr_sizes(&self) -> &[usize] {
        &self.attr_sizes
    }

    /// Return the attribute types.
    pub(crate) fn get_attr_types(&self) -> &[AttributeType] {
        &self.attr_types
    }

    /// Return the attr_filter.
    pub(crate) fn get_report_filter(&self) -> Report<U32_SIZE> {
        self.report_filter.clone()
    }

    /// Return the vector of dummy values for all attributes (largest values with all bits set
    /// for categorical attributes, and zero for numerical).
    pub(crate) fn get_dummy_attr_values(&self) -> Vec<AttrValueType> {
        self.attr_types
            .iter()
            .map(|attr_type| {
                if attr_type.is_numerical() {
                    0
                } else {
                    !0u32 >> (MAX_ATTR_BIT_SIZE - attr_type.get_size())
                }
            })
            .collect::<Vec<_>>()
    }

    /// Return whether an attribute index is valid.
    pub(crate) fn is_valid_attr_index(&self, attr_index: usize) -> bool {
        attr_index < self.attr_sizes.len()
    }

    fn panic_if_invalid_attr_index(&self, attr_index: usize) {
        assert!(
            self.is_valid_attr_index(attr_index),
            "ReportHandler::panic_if_invalid_attr_index: Invalid attribute index."
        );
    }

    fn panic_if_invalid_attr_value(&self, attr_index: usize, attr_value: AttrValueType) {
        assert!(
            self.attr_types[attr_index].is_valid_value(attr_value),
            "ReportHandler::panic_if_invalid_attr_value: Invalid attribute value."
        );
    }

    fn panic_if_invalid_report(&self, report: &Report<U32_SIZE>) {
        assert!(
            self.is_valid_report(report),
            "ReportHandler::panic_if_invalid_report: Invalid report."
        );
    }

    pub(crate) fn get_numerical_attr_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.attr_types
            .iter()
            .enumerate()
            .filter(|(_, attr_type)| attr_type.is_numerical())
            .map(|(attr_index, _)| attr_index)
    }

    /// Set an attribute value in a given `Report`.
    pub fn set_attr(
        &self,
        report: &mut Report<U32_SIZE>,
        attr_index: usize,
        attr_value: AttrValueType,
    ) {
        self.panic_if_invalid_attr_index(attr_index);
        self.panic_if_invalid_attr_value(attr_index, attr_value);

        let attr_offset = self.attr_offsets[attr_index];

        // Move the value to the correct position.
        let attr_report = Report::<U32_SIZE>::from_u32(attr_value) << attr_offset;

        // Clean up the current value for this attribute.
        let attr_filter = self.attr_filters[attr_index];
        *report &= !attr_filter;

        // Set the new attribute value.
        *report |= attr_report;
    }

    /// Creates a new `Report` with given attribute values.
    pub fn create_report(&self, attr_values: &[AttrValueType]) -> Report<U32_SIZE> {
        assert!(attr_values.len() <= self.attr_sizes.len(),
            "ReportHandler::create_report: Number of attribute values must match number of attributes.");

        let mut report = Report::<U32_SIZE>::new();
        for (attr_index, attr_value) in attr_values.iter().enumerate() {
            self.set_attr(&mut report, attr_index, *attr_value);
        }
        report
    }

    /// Create a random `Report`.
    pub(crate) fn create_random_report<R: Rng + ?Sized>(&self, rng: &mut R) -> Report<U32_SIZE> {
        // Sample data for the random report and remove garbage bits.
        let mut random_report = Report::<U32_SIZE>::random(rng) & self.report_filter;

        // For each numerical attribute, sample a random value for the report.
        for attr_index in self.get_numerical_attr_indices() {
            // Sample a random numerical attribute value.
            let modulus = self.attr_types[attr_index].get_modulus();
            let attr_size = self.attr_sizes[attr_index];
            let random_value = random_attr_value(rng, attr_size, modulus);

            self.set_attr(&mut random_report, attr_index, random_value);
        }

        random_report
    }

    /// Create a noise `Report`.
    pub(crate) fn create_noise_report(
        &self,
        attr_index: usize,
        attr_value: AttrValueType,
    ) -> Report<U32_SIZE> {
        let attr_type = self.get_attr_types()[attr_index];
        assert!(
            attr_type.is_categorical(),
            "create_noise_report: The attribute must be categorical."
        );

        // First create a `Report` with all bits for categorical attributes set.
        let mut noise_report = self.report_filter_cat_only;

        // Next set the appropriate attribute to the desired value.
        self.set_attr(&mut noise_report, attr_index, attr_value);

        noise_report
    }

    /// Create a noise `Report` for the dummy bucket. Here every categorical attribute
    /// is set to the dummy value, and the numerical attributes are set to zero.
    pub(crate) fn create_dummy_noise_report(&self) -> Report<U32_SIZE> {
        // First create a `Report` with all bits for categorical attributes set.
        self.report_filter_cat_only
    }

    /// Add two `Report`s.
    fn add_reports(
        &self,
        report1: &Report<U32_SIZE>,
        report2: &Report<U32_SIZE>,
    ) -> Report<U32_SIZE> {
        self.panic_if_invalid_report(report1);
        self.panic_if_invalid_report(report2);

        let mut sum_report = *report1 ^ *report2;

        // For each numerical attribute, do modular addition.
        for attr_index in self.get_numerical_attr_indices() {
            // Get the values from the original reports.
            let attr_value1 = self.get_attr(&report1, attr_index);
            let attr_value2 = self.get_attr(&report2, attr_index);

            // Now compute the sum modulo the attribute modulus. Since numerical
            // attributes are at most 31 bits, the value for rhs is at most 2^31 - 1
            // and we can compute the sum below without u32 overflow, even if the
            // attribute value in rhs originally was invalid (e.g., a 32-bit number).
            let modulus = self.attr_types[attr_index].get_modulus();
            let attr_value = modulus.add_mod(attr_value1, attr_value2);

            self.set_attr(&mut sum_report, attr_index, attr_value);
        }

        sum_report
    }

    /// Subtract two `Report`s.
    fn sub_reports(
        &self,
        report1: &Report<U32_SIZE>,
        report2: &Report<U32_SIZE>,
    ) -> Report<U32_SIZE> {
        self.panic_if_invalid_report(report1);
        self.panic_if_invalid_report(report2);

        let mut diff_report = *report1 ^ *report2;

        // For each numerical attribute, do modular subtraction.
        for attr_index in self.get_numerical_attr_indices() {
            // Get the values from the original reports.
            let attr_value1 = self.get_attr(report1, attr_index);
            let attr_value2 = self.get_attr(report2, attr_index);

            // Now compute the sum modulo the attribute modulus. Since numerical
            // attributes are at most 31 bits, the value for rhs is at most 2^31 - 1
            // and we can compute the sum below without u32 overflow, even if the
            // attribute value in rhs originally was invalid (e.g., a 32-bit number).
            let modulus = self.attr_types[attr_index].get_modulus();
            let attr_value = modulus.sub_mod(attr_value1, attr_value2);

            self.set_attr(&mut diff_report, attr_index, attr_value);
        }

        diff_report
    }

    /// Get an attribute from a given `Report`.
    pub(crate) fn get_attr(&self, report: &Report<U32_SIZE>, attr_index: usize) -> AttrValueType {
        self.panic_if_invalid_attr_index(attr_index);

        let attr_offset = self.attr_offsets[attr_index];
        let attr_filter = self.attr_filters[attr_index];

        // Get the attribute value and make the attribute.
        let attr_report = *report & attr_filter;
        let attr_value = (attr_report >> attr_offset).as_u32_slice()[0];

        attr_value
    }

    /// Set an attribute value to zero in a given `Report`.
    pub(crate) fn clear_attr(&self, report: &mut Report<U32_SIZE>, attr_index: usize) {
        self.panic_if_invalid_attr_index(attr_index);

        let attr_filter = self.attr_filters[attr_index];
        *report &= !attr_filter;
    }

    /// Clean the attribute values in a given `Report`. This clears all garbage bits and
    /// subsequently reduces the numerical attributes modulo the attribute modulus.
    pub(crate) fn is_valid_report(&self, report: &Report<U32_SIZE>) -> bool {
        let garbage_bits = *report & !self.get_report_filter();
        if garbage_bits != Report::<U32_SIZE>::new() {
            return false;
        }

        // For each numerical attribute, check whether we exceed the modulus.
        for attr_index in self.get_numerical_attr_indices() {
            let attr_type = self.attr_types[attr_index];
            let attr_value = self.get_attr(report, attr_index);
            if !attr_type.is_valid_value(attr_value) {
                return false;
            }
        }

        true
    }

    /// Reveal an attribute value in a given `Report`.
    pub(crate) fn reveal_attr(
        &self,
        report: &mut Report<U32_SIZE>,
        attr_index: usize,
        attr_value_share: AttrValueType,
    ) {
        self.panic_if_invalid_attr_index(attr_index);
        self.panic_if_invalid_attr_value(attr_index, attr_value_share);

        let attr_offset = self.attr_offsets[attr_index];
        let mut attr_value_share = attr_value_share;

        if self.attr_types[attr_index].is_numerical() {
            let attr_value = self.get_attr(report, attr_index);
            let modulus = self.attr_types[attr_index].get_modulus();
            attr_value_share = modulus.add_mod(attr_value, attr_value_share);
            self.clear_attr(report, attr_index);
        }

        // Move the value to the correct position.
        let attr_report = Report::<U32_SIZE>::from_u32(attr_value_share) << attr_offset;

        // XOR in the attribute value.
        *report ^= attr_report;
    }

    /// Reveal a `Report` by combining with another `Report`.
    pub(crate) fn reveal(&self, lhs: &mut Report<U32_SIZE>, rhs: &Report<U32_SIZE>) {
        *lhs = self.add_reports(lhs, rhs);
    }

    /// Secret-share a `Report` with a random report generated from a given `Rng`.
    /// Return the random `Report`. The categorical attributes are XORed and the
    /// numerical attributes are subtracted modulo the attribute modulus.
    pub(crate) fn share<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        report: &mut Report<U32_SIZE>,
    ) -> Report<U32_SIZE> {
        let random_report = self.create_random_report(rng);
        *report = self.sub_reports(report, &random_report);

        random_report
    }

    /// Secret-share a `Report` with a random report generated from a given `Rng`.
    /// Return the random `Report`. The categorical attributes are XORed and the
    /// numerical attributes are added modulo the attribute modulus.
    pub(crate) fn reveal_from_random<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        report: &mut Report<U32_SIZE>,
    ) -> Report<U32_SIZE> {
        let random_report = self.create_random_report(rng);
        *report = self.add_reports(report, &random_report);

        random_report
    }

    /// Remove an attribute from the `ReportHandler`.
    pub(crate) fn remove_attr(&mut self, attr_index: usize) {
        self.panic_if_invalid_attr_index(attr_index);

        // Remove the attribute type.
        self.attr_types.remove(attr_index);

        // Remove the attribute size.
        self.attr_sizes.remove(attr_index);

        // Remove the attribute offset.
        self.attr_offsets.remove(attr_index);

        // Remove the attribute filter.
        self.attr_filters.remove(attr_index);

        // Update the report filter.
        self.report_filter = Self::make_report_filter(&self.attr_filters);

        // Update the report filter for categorical attributes.
        self.report_filter_cat_only =
            Self::make_report_filter_cat_only(&self.attr_types, &self.attr_filters);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode;
    use rand::rngs::OsRng;

    #[test]
    fn test_make_attr_filter() {
        let filter = ReportHandler::<1>::make_attr_filter(0, 0);
        assert_eq!(filter, Report::<1>::from_u32(0));

        let filter = ReportHandler::<1>::make_attr_filter(0, 32);
        assert_eq!(filter, Report::<1>::from_u32(0));

        let filter = ReportHandler::<1>::make_attr_filter(4, 0);
        assert_eq!(filter, Report::<1>::from_u32(0b1111));

        let filter = ReportHandler::<1>::make_attr_filter(32, 0);
        assert_eq!(filter, Report::<1>::from_u32(!0));

        let filter = ReportHandler::<1>::make_attr_filter(5, 1);
        assert_eq!(filter, Report::<1>::from_u32(0b111110));

        let filter = ReportHandler::<2>::make_attr_filter(20, 20);
        assert_eq!(
            filter,
            Report::<2>::from_u32_slice(&[0b11111111111100000000000000000000, 0b11111111])
        );
    }

    #[test]
    fn test_make_report_filter() {
        let fi_10_0 = ReportHandler::<1>::make_attr_filter(10, 0);
        let fi_7_10 = ReportHandler::<1>::make_attr_filter(7, 10);
        let fi_3_17 = ReportHandler::<1>::make_attr_filter(3, 17);
        let filters = [fi_10_0, fi_7_10, fi_3_17];

        // Create an array of attributes.
        let attrs = [AttributeType::C10, AttributeType::N7(20), AttributeType::C3];

        // Create a report filter.
        let report_filter = ReportHandler::<1>::make_report_filter(&filters);
        assert_eq!(
            report_filter,
            Report::<1>::from_u32(0b00000000000_111_1111111_1111111111)
        );

        let report_filter_cat_only =
            ReportHandler::<1>::make_report_filter_cat_only(&attrs, &filters);
        assert_eq!(
            report_filter_cat_only,
            Report::<1>::from_u32(0b00000000000_111_0000000_1111111111)
        );
    }

    #[test]
    fn test_new() {
        let attr_types = vec![AttributeType::C2, AttributeType::N3(5), AttributeType::C4];
        let report_handler = ReportHandler::<1>::new(&attr_types);
        assert_eq!(report_handler.attr_offsets, vec![0, 2, 5]);

        let attr_types = vec![
            AttributeType::C32,
            AttributeType::C2,
            AttributeType::C3,
            AttributeType::C4,
            AttributeType::N5(30),
        ];
        let report_handler = ReportHandler::<2>::new(&attr_types);
        assert_eq!(report_handler.attr_offsets, vec![0, 32, 34, 37, 41]);
    }

    #[test]
    #[should_panic(expected = "ReportHandler::new: Invalid attribute types.")]
    fn test_new_panic1() {
        let attr_types = vec![AttributeType::C2, AttributeType::N3(9), AttributeType::C4];
        let _report_handler = ReportHandler::<1>::new(&attr_types);
    }

    #[test]
    #[should_panic(expected = "ReportHandler::new: Invalid attribute sizes.")]
    fn test_new_panic3() {
        let attr_types = (0..7).map(|_| AttributeType::C10).collect::<Vec<_>>();
        let _report_handler = ReportHandler::<2>::new(&attr_types);
    }

    #[test]
    fn test_set_attr() {
        let attr_types = vec![
            AttributeType::C2,
            AttributeType::C3,
            AttributeType::N4(12),
            AttributeType::C5,
        ];
        let report_handler = ReportHandler::<1>::new(&attr_types);

        let mut report = Report::<1>::new();
        report_handler.set_attr(&mut report, 0, 1);
        assert_eq!(report, Report::<1>::from_u32(0b01));

        report_handler.set_attr(&mut report, 1, 2);
        assert_eq!(report, Report::<1>::from_u32(0b010_01));

        report_handler.set_attr(&mut report, 2, 3);
        assert_eq!(report, Report::<1>::from_u32(0b0011_010_01));

        report_handler.set_attr(&mut report, 3, 4);
        assert_eq!(report, Report::<1>::from_u32(0b00100_0011_010_01));

        report_handler.set_attr(&mut report, 0, 0);
        assert_eq!(report, Report::<1>::from_u32(0b00100_0011_010_00));

        report_handler.set_attr(&mut report, 1, 0);
        assert_eq!(report, Report::<1>::from_u32(0b00100_0011_000_00));

        report_handler.set_attr(&mut report, 2, 0);
        assert_eq!(report, Report::<1>::from_u32(0b00100_0000_000_00));

        report_handler.set_attr(&mut report, 3, 0);
        assert_eq!(report, Report::<1>::from_u32(0b00000_0000_000_00));

        let attr_types = vec![
            AttributeType::C10,
            AttributeType::N20(0b1111_1111_1111_1111_1111),
            AttributeType::C30,
        ];
        let report_handler = ReportHandler::<2>::new(&attr_types);

        let mut report = Report::<2>::new();
        report_handler.set_attr(&mut report, 0, 0b0101010101);
        assert_eq!(report, Report::<2>::from_u32_slice(&[0b0101010101, 0b0]));

        report_handler.set_attr(&mut report, 1, 0b01001001001001001001);
        assert_eq!(
            report,
            Report::<2>::from_u32_slice(&[0b010010010010010010010101010101, 0])
        );

        report_handler.set_attr(&mut report, 2, 0b110110110110110110110110110110);
        assert_eq!(
            report,
            Report::<2>::from_u32_slice(&[
                0b10010010010010010010010101010101,
                0b1101101101101101101101101101
            ])
        );
    }

    #[test]
    fn test_create_report() {
        let attr_types = vec![
            AttributeType::C2,
            AttributeType::C3,
            AttributeType::C4,
            AttributeType::C5,
        ];
        let report_handler = ReportHandler::<1>::new(&attr_types);

        let attr_values = vec![0, 0, 0, 0];
        let report = report_handler.create_report(&attr_values);
        assert_eq!(report, Report::<1>::from_u32(0b0000000000000));

        let attr_values = vec![0b10, 0b011, 0b1011, 0b01100];
        let report = report_handler.create_report(&attr_values);
        assert_eq!(report, Report::<1>::from_u32(0b01100101101110));

        let attr_types = vec![AttributeType::C10, AttributeType::C20, AttributeType::C30];
        let report_handler = ReportHandler::<2>::new(&attr_types);

        let attr_values = vec![
            0b0101010101,
            0b01001001001001001001,
            0b110110110110110110110110110110,
        ];
        let report = report_handler.create_report(&attr_values);
        assert_eq!(
            report,
            Report::<2>::from_u32_slice(&[
                0b10010010010010010010010101010101,
                0b1101101101101101101101101101
            ])
        );
    }

    #[test]
    fn test_get_attr() {
        let attr_types = vec![
            AttributeType::C2,
            AttributeType::C3,
            AttributeType::C4,
            AttributeType::C5,
        ];
        let report_handler = ReportHandler::<1>::new(&attr_types);

        let report = Report::<1>::from_u32(0b01100101101110);
        assert_eq!(report_handler.get_attr(&report, 0), 0b10);
        assert_eq!(report_handler.get_attr(&report, 1), 0b011);
        assert_eq!(report_handler.get_attr(&report, 2), 0b1011);
        assert_eq!(report_handler.get_attr(&report, 3), 0b01100);

        let attr_types = vec![AttributeType::C10, AttributeType::C20, AttributeType::C30];
        let report_handler = ReportHandler::<2>::new(&attr_types);

        let report = Report::<2>::from_u32_slice(&[
            0b10010010010010010010010101010101,
            0b1101101101101101101101101101,
        ]);
        assert_eq!(report_handler.get_attr(&report, 0), 0b0101010101);
        assert_eq!(report_handler.get_attr(&report, 1), 0b01001001001001001001);
        assert_eq!(
            report_handler.get_attr(&report, 2),
            0b110110110110110110110110110110
        );
    }

    #[test]
    fn test_clear_attr() {
        let attr_types = vec![
            AttributeType::C2,
            AttributeType::C3,
            AttributeType::C4,
            AttributeType::C5,
        ];
        let report_handler = ReportHandler::<1>::new(&attr_types);

        let mut report = Report::<1>::from_u32(0b01100101101110);
        report_handler.clear_attr(&mut report, 0);
        assert_eq!(report, Report::<1>::from_u32(0b01100101101100));

        report_handler.clear_attr(&mut report, 1);
        assert_eq!(report, Report::<1>::from_u32(0b01100101100000));

        report_handler.clear_attr(&mut report, 2);
        assert_eq!(report, Report::<1>::from_u32(0b01100000000000));

        report_handler.clear_attr(&mut report, 3);
        assert_eq!(report, Report::<1>::from_u32(0b00000000000000));

        let attr_types = vec![AttributeType::C10, AttributeType::C20, AttributeType::C30];
        let report_handler = ReportHandler::<2>::new(&attr_types);

        let mut report = Report::<2>::from_u32_slice(&[
            0b10010010010010010010010101010101,
            0b1101101101101101101101101101,
        ]);
        report_handler.clear_attr(&mut report, 0);
        assert_eq!(
            report,
            Report::<2>::from_u32_slice(&[
                0b10010010010010010010010000000000,
                0b1101101101101101101101101101
            ])
        );

        report_handler.clear_attr(&mut report, 1);
        assert_eq!(
            report,
            Report::<2>::from_u32_slice(&[
                0b10000000000000000000000000000000,
                0b1101101101101101101101101101
            ])
        );

        report_handler.clear_attr(&mut report, 2);
        assert_eq!(report, Report::<2>::from_u32_slice(&[0b0, 0b0]));
    }

    #[test]
    fn test_reveal_attr() {
        let attr_types = vec![
            AttributeType::C2,
            AttributeType::C3,
            AttributeType::C4,
            AttributeType::C5,
        ];
        let report_handler = ReportHandler::<1>::new(&attr_types);

        let mut report = Report::<1>::from_u32(0b01100_1011_011_10);

        report_handler.reveal_attr(&mut report, 0, 0b01);
        assert_eq!(report, Report::<1>::from_u32(0b01100_1011_011_11));

        report_handler.reveal_attr(&mut report, 1, 0b101);
        assert_eq!(report, Report::<1>::from_u32(0b01100_1011_110_11));

        report_handler.reveal_attr(&mut report, 2, 0b0110);
        assert_eq!(report, Report::<1>::from_u32(0b01100_1101_110_11));

        report_handler.reveal_attr(&mut report, 3, 0b10110);
        assert_eq!(report, Report::<1>::from_u32(0b11010_1101_110_11));
    }

    #[test]
    fn test_serialize_deserialize() {
        let attr_types = vec![
            AttributeType::C4,
            AttributeType::C8,
            AttributeType::C12,
            AttributeType::C16,
            AttributeType::C24,
        ];
        let report_handler = ReportHandler::<2>::new(&attr_types);
        let encoded = bincode::serialize(&report_handler).unwrap();
        let report_handler_copy: ReportHandler<2> = bincode::deserialize(&encoded).unwrap();
        assert_eq!(
            report_handler.attr_filters,
            report_handler_copy.attr_filters
        );
        assert_eq!(report_handler.attr_sizes, report_handler_copy.attr_sizes);
        assert_eq!(
            report_handler.attr_offsets,
            report_handler_copy.attr_offsets
        );
    }

    #[test]
    fn test_dummy_attr_value() {
        let attr_types = vec![
            AttributeType::C4,
            AttributeType::C8,
            AttributeType::C12,
            AttributeType::C16,
            AttributeType::C24,
            AttributeType::C32,
        ];
        let report_handler = ReportHandler::<3>::new(&attr_types);
        let dummy_attr_values = report_handler.get_dummy_attr_values();
        assert_eq!(dummy_attr_values[0], 0b1111);
        assert_eq!(dummy_attr_values[1], 0b1111_1111);
        assert_eq!(dummy_attr_values[2], 0b1111_1111_1111);
        assert_eq!(dummy_attr_values[3], 0b1111_1111_1111_1111);
        assert_eq!(dummy_attr_values[4], 0b1111_1111_1111_1111_1111_1111);
        assert_eq!(
            dummy_attr_values[5],
            0b1111_1111_1111_1111_1111_1111_1111_1111
        );
    }

    #[test]
    fn test_add_sub_report() {
        let attr_types = vec![
            AttributeType::C4,
            AttributeType::N8(200),
            AttributeType::C12,
            AttributeType::N16(50000),
        ];
        let report_handler = ReportHandler::<2>::new(&attr_types);

        let mut rng = OsRng;
        let report1 = report_handler.create_random_report(&mut rng);
        let report2 = report_handler.create_random_report(&mut rng);

        let sum_report = report_handler.add_reports(&report1, &report2);
        assert_ne!(report1, sum_report);

        let diff_report = report_handler.sub_reports(&sum_report, &report2);
        assert_eq!(report1, diff_report);

        let n16_value_report1 = report_handler.get_attr(&report1, 3);
        let n16_value_report2 = report_handler.get_attr(&report2, 3);
        let n16_value_sum = report_handler.get_attr(&sum_report, 3);
        let n16_value_true_sum = 50000u32.add_mod(n16_value_report1, n16_value_report2);
        assert_eq!(n16_value_sum, n16_value_true_sum);
    }
}
