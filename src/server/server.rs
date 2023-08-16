// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use super::histogram::Histogram;
use super::ot::{OTMessage, OTProtocolHandler};
use super::role::Role;
use crate::arith::Modulus;
use crate::random::hist_noise::NoiseDistribution;
use crate::report::attr::AttrValueType;
use crate::report::report::Report;
use crate::report::report_vector::ReportVector;
use crate::schema::*;
use itertools::izip;
use num_traits::Zero;
use rand::{Rng, SeedableRng};
use std::cell::RefCell;
use std::rc::Rc;

cfg_if::cfg_if! {
    if #[cfg(feature = "wide_summation_modulus")] {
        /// Using a 64-bit summation modulus. Disable feature `wide_summation_modulus`
        /// to use a `u32` instead if you do not need the extra range.
        pub type SummationModulus = u64;
    } else {
        /// Using a 32-bit summation modulus. Enable feature `wide_summation_modulus`
        /// to use a `u64` instead.
        pub type SummationModulus = u32;
    }
}

#[derive(Clone)]
pub struct Server<const REPORT_U32_SIZE: usize> {
    role: Role,
    schema: Schema,
    reports: Option<ReportVector<REPORT_U32_SIZE>>,
    sum_protocol_data: Option<(OTProtocolHandler<SummationModulus>, String)>,
}

impl<const REPORT_U32_SIZE: usize> Server<REPORT_U32_SIZE> {
    /// Check that a given `ReportVector` is compatible with the `Schema` of the `Server`.
    fn is_compatible_report_vector(
        &self,
        reports: &ReportVector<REPORT_U32_SIZE>,
    ) -> Result<(), String> {
        let report_handler = reports.report_handler();
        if !report_handler.is_compatible_schema(&self.schema) {
            return Err(format!(
                "ReportVector with attributes {:?} is incompatible with schema {:?}.",
                report_handler.get_attr_types(),
                self.schema
            ));
        }

        Ok(())
    }

    /// Create a new server with the given role.
    pub fn new(role: Role, schema: Schema) -> Self {
        Self {
            role,
            schema,
            reports: None,
            sum_protocol_data: None,
        }
    }

    /// Return the current role.
    pub fn get_role(&self) -> Role {
        self.role
    }

    /// Add reports to the current set of reports the `Server` holds.
    pub fn add_reports(&mut self, reports: ReportVector<REPORT_U32_SIZE>) -> Result<(), String> {
        // Check that the `ReportVector` is compatible with the `Schema`.
        self.is_compatible_report_vector(&reports)?;

        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        // If there are no reports yet, simply set the current reports to the given reports.
        if Option::is_none(&self.reports) {
            self.reports = Some(reports);
            return Ok(());
        }

        // At this point we know the `ReportHandler`s are the same. Merge the reports.
        let curr_reports = self.reports.as_mut().unwrap();
        for report in reports.iter() {
            curr_reports.push(*report);
        }
        Ok(())
    }

    /// Add noise reports to the reports the `Server` holds. If successful, the function
    /// returns the number of noise reports added.
    pub fn add_noise_reports<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        attr_name: &str,
        noise_distr: impl NoiseDistribution,
        dummy_noise_distr: impl NoiseDistribution,
    ) -> Result<usize, String> {
        // Do we have any reports to add noise to?
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        let reports = self.reports.as_mut().unwrap();
        let report_handler = reports.report_handler().clone();

        // Check that the attribute name is valid.
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name)?;

        // Ensure the attribute is categorical.
        let attr_type = attr_type_from_attr_name(&self.schema, attr_name)?;
        if !attr_type.is_categorical() {
            return Err(format!(
                "Attribute {} (type {:?}) is not categorical.",
                attr_name, attr_type
            ));
        }

        // Get the dummy attribute value.
        let dummy_attr_value = report_handler.get_dummy_attr_values()[attr_index];

        // We add noise reports for each possible attribute value. The number of reports is
        // sampled from a truncated shifted Laplace distribution. We use the same distribution
        // for all attribute values except the dummy attribute value, which is sampled from
        // a different distribution.
        let mut rng: &mut R = rng;
        let dummy_noise_sample = dummy_noise_distr.sample(&mut rng);
        let noise_samples = noise_distr.sample_n(&mut rng, dummy_attr_value as usize);

        let old_reports_len = reports.len();
        for attr_value in 0..dummy_attr_value {
            // First create a `Report` with all bits set.
            let noise_report = report_handler.create_noise_report(attr_index, attr_value);

            // Next push these repeatedly to the reports vector. This function will clean
            // up all of the garbage 1-bits from the `Report`.
            reports.push_many(noise_report, noise_samples[attr_value as usize]);
        }

        // For the dummy attribute, add noise from a different distribution.
        let dummy_noise_report = report_handler.create_dummy_noise_report();
        reports.push_many(dummy_noise_report, dummy_noise_sample);

        let new_reports_len = reports.len();
        Ok(new_reports_len - old_reports_len)
    }

    /// Add empty (all bits zero) reports.
    pub fn add_empty_reports(&mut self, count: usize) -> Result<(), String> {
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        let reports = self.reports.as_mut().unwrap();
        let zero_report = Report::<REPORT_U32_SIZE>::new();
        reports.push_many(zero_report, count);

        Ok(())
    }

    /// Clear the current set of reports the `Server` holds.
    pub fn extract_reports(&mut self) -> Result<ReportVector<REPORT_U32_SIZE>, String> {
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        Ok(self.reports.take().unwrap())
    }

    /// Secret-share the entire `ReportVector`.
    pub(crate) fn share<T>(
        &mut self,
        rng_seed: T::Seed,
    ) -> Result<ReportVector<REPORT_U32_SIZE>, String>
    where
        T: Rng + SeedableRng,
    {
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        let mut rng = T::from_seed(rng_seed);
        Ok(self.reports.as_mut().unwrap().share(&mut rng))
    }

    /// Reveal the secret-shared `ReportVector`.
    pub fn reveal(&mut self, shares: ReportVector<REPORT_U32_SIZE>) -> Result<(), String> {
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        // Check that the `ReportVector` is compatible with the `Schema`.
        self.is_compatible_report_vector(&shares)?;

        let reports = self.reports.as_mut().unwrap();

        // The number of attribute shares must be equal to the number of reports.
        if reports.len() != shares.len() {
            return Err("Invalid number of shares.".to_string());
        }

        // Reveal the attribute values.
        reports.reveal(shares);

        Ok(())
    }

    /// Return the number of reports the `Server` holds.
    pub fn get_report_count(&self) -> Result<usize, String> {
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        Ok(self.reports.as_ref().unwrap().len())
    }

    /// Returns the values for any of the categorical attributes in the `ReportVector`
    /// the `Server` holds.
    pub fn get_attr_values(&self, attr_name: &str) -> Result<Vec<AttrValueType>, String> {
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        // Check that the attribute name is valid.
        let reports = self.reports.as_ref().unwrap();
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name)?;

        Ok(reports.get_attr_iter(attr_index).collect())
    }

    /// Remove an attribute from the `Schema` and `ReportVector` the `Server` holds.
    pub fn remove_attr(&mut self, attr_name: &str) -> Result<(), String> {
        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        // Check that the attribute name is valid.
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name)?;

        // Remove the attribute from the schema.
        self.schema.remove_attr(attr_name);

        if let Some(reports) = self.reports.as_mut() {
            // Remove the attribute from the reports.
            reports.remove_attr(attr_index);
        }

        Ok(())
    }

    /// Split the `Server`s `ReportVector` into two `ReportVector`s. The function returns another
    /// `Server` that holds the `ReportVector` reports where the attribute had the given value.
    /// In the returned `Server`, the attribute is removed from the `Schema` and `ReportVector`.
    /// The attribute to split at must be categorical.
    pub fn split_at(
        &mut self,
        attr_name: &str,
        attr_value: AttrValueType,
    ) -> Result<Server<REPORT_U32_SIZE>, String> {
        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        // Check that the attribute name is valid.
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name)?;

        // Get the attribute type and check that the attribute is categorical.
        let attr_type = attr_type_from_attr_name(&self.schema, attr_name)?;
        if !attr_type.is_categorical() {
            return Err(format!(
                "Attribute {} (type {:?}) is not categorical.",
                attr_name, attr_type
            ));
        }

        // Verify that attr_value is valid for this type.
        if !attr_type.is_valid_value(attr_value) {
            return Err(format!(
                "Attribute value {:?} is not valid for attribute {} (type {:?}).",
                attr_value, attr_name, attr_type
            ));
        }

        // This is the schema for the return value.
        let mut sub_schema = self.schema.clone();
        sub_schema.remove_attr(attr_name);

        // If the given value is invalid or there are no reports anyway, return an empty
        // `Server` with this attribute removed.
        let attr = attr_from_attr_value(attr_type, attr_value);
        if attr.is_err() || self.reports.is_none() {
            return Ok(Server::new(self.role, sub_schema.clone()));
        }

        // We have some reports. Split them and create a new `Server` with the split-off.
        let reports = self.reports.as_mut().unwrap();
        let split_reports = reports.split_at(attr_index, attr_value);
        let mut sub_server = Server::new(self.role, sub_schema.clone());
        sub_server.add_reports(split_reports)?;

        Ok(sub_server)
    }

    /// Reveal secret-shared attribute values for every report the `Server` holds.
    pub fn reveal_attr(
        &mut self,
        attr_name: &str,
        attr_shares: Vec<AttrValueType>,
    ) -> Result<(), String> {
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        let reports = self.reports.as_mut().unwrap();

        // Check that the attribute name is valid.
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name)?;

        // Get the attribute type and check that the attribute is categorical.
        let attr_type = attr_type_from_attr_name(&self.schema, attr_name)?;
        if !attr_type.is_categorical() {
            return Err(format!(
                "Attribute {} (type {:?}) is not categorical.",
                attr_name, attr_type
            ));
        }

        // The number of attribute shares must be equal to the number of reports.
        if reports.len() != attr_shares.len() {
            return Err("Invalid number of attribute shares.".to_string());
        }

        // Verify that each value in attr_shares is valid for this type.
        let attr_type = attr_type_from_attr_name(&self.schema, attr_name)?;
        for attr_share in attr_shares.iter() {
            if !attr_type.is_valid_value(*attr_share) {
                return Err(format!(
                    "Attribute value {:?} is not valid for attribute {} (type {:?}).",
                    attr_share, attr_name, attr_type
                ));
            }
        }

        // Reveal the attribute values.
        reports.reveal_attr(attr_index, attr_shares);

        Ok(())
    }

    /// Prune the `Server`s reports according to a given attribute. The attribute to prune
    /// at must be categorical.
    pub fn prune(&mut self, attr_name: &str, threshold: usize) -> Result<usize, String> {
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        let reports = self.reports.as_mut().unwrap();

        // Check that the attribute name is valid.
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name)?;

        // Get the attribute type and check that the attribute is categorical.
        let attr_type = attr_type_from_attr_name(&self.schema, attr_name)?;
        if !attr_type.is_categorical() {
            return Err(format!(
                "Attribute {} (type {:?}) is not categorical.",
                attr_name, attr_type
            ));
        }

        // Prune the reports.
        let pruned = reports.prune(attr_index, threshold);

        Ok(pruned)
    }

    /// Create a histogram for a given attribute from the reports the `Server` holds.
    pub fn make_histogram(
        &self,
        attr_name: &str,
        m: i64,
        prune_threshold: usize,
    ) -> Result<Rc<RefCell<Histogram>>, String> {
        if self.reports == None {
            return Err("No report vector found.".to_string());
        }

        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        if m > 0 {
            return Err("Gaussian shift parameter m must be non-positive.".to_string());
        }

        let reports = self.reports.as_ref().unwrap();

        // Check that the attribute name is valid.
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name)?;

        // Create the histogram. This function returns an error if the attribute is categorical.
        let histogram = Histogram::new(
            &self.schema,
            attr_name,
            reports.get_attr_iter(attr_index),
            m,
            prune_threshold,
        )?;

        Ok(histogram)
    }

    /// Performs the oblivious permutation protocol.
    pub fn oblivious_permute<T: Rng + SeedableRng>(
        &mut self,
        rng_seed1: T::Seed,
        rng_seed2: T::Seed,
        mut partial_result: Option<ReportVector<REPORT_U32_SIZE>>,
    ) -> Result<Option<ReportVector<REPORT_U32_SIZE>>, String> {
        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        match self.role {
            Role::First => {
                if self.reports == None {
                    return Err("No report vector found.".to_string());
                }
                if partial_result == None {
                    Err("Partial result should not be None.".to_string())
                } else {
                    let mut rng12 = T::from_seed(rng_seed1);
                    let mut rng13 = T::from_seed(rng_seed2);

                    let reports = self.reports.as_mut().unwrap();
                    let partial_result_reports = partial_result.as_mut().unwrap();

                    reports.permute(&mut rng12);
                    reports.share(&mut rng12);
                    let ret = self.extract_reports().unwrap();

                    partial_result_reports.permute(&mut rng13);
                    partial_result_reports.share(&mut rng13);
                    self.add_reports(partial_result.unwrap()).unwrap();

                    Ok(Some(ret))
                }
            }
            Role::Second => {
                if self.reports == None {
                    return Err("No report vector found.".to_string());
                }
                if partial_result.is_none() {
                    let mut rng23 = T::from_seed(rng_seed1);
                    let mut rng12 = T::from_seed(rng_seed2);

                    let reports = self.reports.as_mut().unwrap();

                    reports.permute(&mut rng12);
                    reports.reveal_from_random(&mut rng12);
                    reports.permute(&mut rng23);
                    reports.reveal_from_random(&mut rng23);

                    Ok(Some(self.extract_reports().unwrap()))
                } else {
                    Err("Partial result should be None.".to_string())
                }
            }
            Role::Third => {
                if self.reports != None {
                    return Err("Report vector should be empty.".to_string());
                }
                if partial_result == None {
                    Err("Partial result should not be None.".to_string())
                } else {
                    let mut rng13 = T::from_seed(rng_seed1);
                    let mut rng23 = T::from_seed(rng_seed2);

                    self.reports = None;
                    self.add_reports(partial_result.unwrap()).unwrap();
                    let reports = self.reports.as_mut().unwrap();

                    reports.permute(&mut rng23);
                    reports.share(&mut rng23);
                    reports.permute(&mut rng13);
                    reports.reveal_from_random(&mut rng13);

                    Ok(None)
                }
            }
        }
    }

    /// Rotate the role after an oblivious permutation.
    pub fn rotate_role(&mut self) -> Result<(), String> {
        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        self.role.rotate();
        Ok(())
    }

    /// Initiate the summation protocol.
    pub fn summation_initialize<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        attr_name: &str,
        summation_modulus: SummationModulus,
    ) -> Result<(), String> {
        // Check that a summation protocol is not initialized.
        if self.is_summation_initialized() {
            return Err(
                "Cannot perform this action while summation protocol is initialized.".to_string(),
            );
        }

        // Ensure the attribute is valid and numerical.
        let attr_type = attr_type_from_attr_name(&self.schema, attr_name)?;
        if !attr_type.is_numerical() {
            return Err(format!(
                "Attribute {} (type {:?}) is not numerical.",
                attr_name, attr_type
            ));
        }

        // Ensure that `summation_modulus` is larger than the attribute modulus.
        let attr_modulus = SummationModulus::from(attr_type.get_modulus());
        if summation_modulus <= attr_modulus {
            return Err(format!(
                "Summation modulus {} is not larger than the attribute modulus {}.",
                summation_modulus, attr_modulus
            ));
        }

        match OTProtocolHandler::<SummationModulus>::new(self.role, rng, summation_modulus) {
            Ok(handler) => {
                self.sum_protocol_data = Some((handler, attr_name.to_string()));
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Finalize the summation protocol.
    pub fn summation_finalize(&mut self) {
        self.sum_protocol_data = None;
    }

    /// Return whether the summation protocol has been initialized.
    pub fn is_summation_initialized(&self) -> bool {
        self.sum_protocol_data.is_some()
    }

    /// Create the OT key. This function is only called by `Role::Third`. The resulting message
    /// is sent to `Role::Second`.
    pub fn summation_create_key(&self) -> Result<OTMessage<SummationModulus>, String> {
        // Check that the OT handler is set.
        if !self.is_summation_initialized() {
            Err("Summation protocol has not been initialized.".to_string())
        } else {
            self.sum_protocol_data
                .as_ref()
                .unwrap()
                .0
                .create_key_message()
        }
    }

    /// Create seed messages. This function is only called by `Role::Third`. The output messages
    /// are sent to `Role::First`.
    pub fn summation_create_seeds<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        count: usize,
    ) -> Result<Vec<OTMessage<SummationModulus>>, String> {
        // Check that the OT handler is set.
        if !self.is_summation_initialized() {
            return Err("Summation protocol has not been initialized.".to_string());
        }

        let ot_handler = &self.sum_protocol_data.as_ref().unwrap().0;
        ot_handler.create_seed_messages(rng, count)
    }

    /// Create masked bit messages. This function is only called by `Role::First` and must be
    /// called with the seed messages received from `Role::Third`. The output messages are
    /// sent to `Role::Second`.
    pub fn summation_create_masked_bits<R: Rng + SeedableRng>(
        &mut self,
        seed_msgs: Vec<OTMessage<SummationModulus>>,
    ) -> Result<Vec<OTMessage<SummationModulus>>, String> {
        // Check that the OT handler is set.
        if !self.is_summation_initialized() {
            return Err("Summation protocol has not been initialized.".to_string());
        }

        // Check that the reports are set.
        if self.reports.is_none() {
            return Err("No report vector found.".to_string());
        }

        let reports = self.reports.as_ref().unwrap();
        let (ot_handler, attr_name) = &mut self.sum_protocol_data.as_mut().unwrap();
        let attr_type = attr_type_from_attr_name(&self.schema, attr_name)?;
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name)?;
        let attr_modulus = attr_type.get_modulus();

        // Check that the number of reports matches the size of `seed_msgs`.
        if reports.len() != seed_msgs.len() {
            return Err("Number of reports does not match number of seed messages.".to_string());
        }

        // Create the input bit vector from reports.
        let input_bits = reports
            .get_attr_iter(attr_index)
            .map(|attr_value| attr_modulus.add_mod(attr_value, attr_value) & 1 != 0)
            .collect::<Vec<_>>();

        ot_handler.create_masked_bit_messages::<R>(seed_msgs, &input_bits)
    }

    /// Receive the OT key. This function is only called by `Role::Second`.
    pub fn summation_receive_key(
        &mut self,
        key_msg: OTMessage<SummationModulus>,
    ) -> Result<(), String> {
        // Check that the OT handler is set.
        if !self.is_summation_initialized() {
            return Err("Summation protocol has not been initialized.".to_string());
        }

        self.sum_protocol_data
            .as_mut()
            .unwrap()
            .0
            .process_key_message(key_msg)
    }

    /// Create reveal messages. This function is only called by `Role::Second` and must be
    /// called with the masked bits received from `Role::First`. The function outputs the
    /// final output for `Role::Second` and messages to be sent back to `Role::First`.
    pub fn summation_create_reveal_msgs<R: Rng + SeedableRng>(
        &mut self,
        rng: &mut (impl Rng + ?Sized),
        masked_bits: Vec<OTMessage<SummationModulus>>,
    ) -> Result<(Vec<OTMessage<SummationModulus>>, SummationModulus), String> {
        // Check that the OT handler is set.
        if !self.is_summation_initialized() {
            return Err("Summation protocol has not been initialized.".to_string());
        }

        // Check that the reports are set.
        if self.reports.is_none() {
            return Err("No report vector found.".to_string());
        }

        let reports = self.reports.as_ref().unwrap();
        let (ot_handler, attr_name) = &self.sum_protocol_data.as_mut().unwrap();
        let attr_type = attr_type_from_attr_name(&self.schema, attr_name.as_str())?;
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name.as_str())?;
        let attr_modulus = attr_type.get_modulus();
        let summation_modulus = ot_handler.get_modulus();

        // Check that the number of reports matches the size of `masked_bits`.
        if reports.len() != masked_bits.len() {
            return Err(
                "Number of reports does not match number of masked bit messages.".to_string(),
            );
        }

        // Create the input bit vector from reports.
        let input_bits = reports
            .get_attr_iter(attr_index)
            .map(|attr_value| attr_modulus.add_mod(attr_value, attr_value) & 1 != 0)
            .collect::<Vec<_>>();

        let (reveal_msgs, lsb_shares) =
            ot_handler.create_reveal_messages::<R>(rng, masked_bits, &input_bits)?;

        let attr_modulus = SummationModulus::from(attr_modulus);
        let mut s2_output = izip!(
            lsb_shares.iter().copied(),
            input_bits.iter().copied(),
            reports
                .get_attr_iter(attr_index)
                .map(|attr_value| attr_modulus.add_mod(attr_value.into(), attr_value.into()))
        )
        .map(|(lsb_share, input_bit, attr_value)| -> SummationModulus {
            let mut q = summation_modulus.sub_mod(
                summation_modulus.sub_mod(input_bit.into(), lsb_share),
                lsb_share,
            );
            q = summation_modulus.mul_mod(q, attr_modulus);
            summation_modulus.sub_mod(attr_value, q)
        })
        .fold(SummationModulus::zero(), |acc, x| {
            summation_modulus.add_mod(acc, x)
        });

        // Divide the result by two.
        let two_inv = summation_modulus.inv_mod(SummationModulus::from(2u16));
        s2_output = summation_modulus.mul_mod(s2_output, two_inv);

        Ok((reveal_msgs, s2_output))
    }

    /// Receive reveal messages. This function is only called by `Role::First` and must be
    /// called with the reveal messages received from `Role::Second`. The function outputs the
    /// final outputs for `Role::First`. After this step, all parties must call `finalize_summation`.
    pub fn summation_receive_reveal_msgs(
        &mut self,
        reveal_msgs: Vec<OTMessage<SummationModulus>>,
    ) -> Result<SummationModulus, String> {
        // Check that the OT handler is set.
        if !self.is_summation_initialized() {
            return Err("Summation protocol has not been initialized.".to_string());
        }

        let reports = self.reports.as_ref().unwrap();
        let (ot_handler, attr_name) = &mut self.sum_protocol_data.as_mut().unwrap();
        let attr_type = attr_type_from_attr_name(&self.schema, attr_name.as_str())?;
        let attr_index = attr_index_from_attr_name(&self.schema, attr_name.as_str())?;
        let attr_modulus = attr_type.get_modulus();
        let summation_modulus = ot_handler.get_modulus();

        let lsb_shares = ot_handler.process_reveal_messages(reveal_msgs)?;

        let attr_modulus = SummationModulus::from(attr_modulus);
        let input_bit_iter = reports
            .get_attr_iter(attr_index)
            .map(|attr_value| attr_modulus.add_mod(attr_value.into(), attr_value.into()) & 1 != 0);
        let mut s1_output = izip!(
            lsb_shares.iter().copied(),
            input_bit_iter,
            reports
                .get_attr_iter(attr_index)
                .map(|attr_value| attr_modulus.add_mod(attr_value.into(), attr_value.into()))
        )
        .map(|(lsb_share, input_bit, attr_value)| -> SummationModulus {
            let mut q = summation_modulus.sub_mod(
                summation_modulus.sub_mod(input_bit.into(), lsb_share),
                lsb_share,
            );
            q = summation_modulus.mul_mod(q, attr_modulus);
            summation_modulus.sub_mod(attr_value, q)
        })
        .fold(SummationModulus::zero(), |acc, x| {
            summation_modulus.add_mod(acc, x)
        });

        // Divide the result by two.
        let two_inv = summation_modulus.inv_mod(SummationModulus::from(2u16));
        s1_output = summation_modulus.mul_mod(s1_output, two_inv);

        Ok(s1_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_reports_err() {
        let schema =
            Schema::try_from(r#"[["attr1","c2"],["attr2","c10"],["attr3","c4"]]"#).unwrap();

        // Create a `ReportVector` with one 10-bit attribute.
        let report_vector =
            ReportVector::<1>::new(&schema.clone().remove_attr("attr2").get_attr_types());
        let mut server = Server::<1>::new(Role::First, schema.clone());
        assert!(server.add_reports(report_vector).is_err());
    }
}
