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
        type SummationModulus = u64;
    } else {
        /// Using a 32-bit summation modulus. Enable feature `wide_summation_modulus`
        /// to use a `u64` instead.
        type SummationModulus = u32;
    }
}

#[derive(Clone)]
struct Server<const REPORT_U32_SIZE: usize> {
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
                if partial_result == None{
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
    use crate::random::hist_noise::Gaussian;
    use crate::random::hist_noise::Laplace;
    use crate::report::report_vector::ReportVector;
    use crate::schema::Schema;
    use bincode;
    use rand::rngs::{OsRng, StdRng};
    use std::cmp::{max, min};
    use std::collections::HashMap;
    use std::time::Instant;

    #[allow(unused_imports)]
    use crate::client::create_report_shares;

    #[allow(unused_imports)]
    use std::fs;

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

    #[test]
    fn test_layered_histogram() {
        // We demonstrate a schema consisting of three categorical attributes.
        let schema =
            Schema::try_from(r#"[["attr1","c2"],["attr2","c10"],["attr3","c4"]]"#).unwrap();

        // The total bit-size of the attributes is 2+10+4=16 bits, so a single u32 suffices.
        const U32_SIZE: usize = 1;

        let mut rng = OsRng;

        // This is the truncation point for the differential privacy noise distribution.
        let m: i64 = -10;

        // We use Laplace-DP in this example. Here set epsilon to 1.0 for a single layered
        // execution. For determining the total epsilon, see https://eprint.iacr.org/2021/1490.
        let epsilon = 1.0;
        let noise_distr = Laplace::new(m, 1.0 / epsilon).unwrap();

        // This is the threshold for pruning. We do not care about values that show up fewer than
        // this many times (including added noise reports).
        let threshold: usize = 50;

        // Create a `ReportVector` from this schema. This represents the clients' reports.
        let mut report_vector = ReportVector::<U32_SIZE>::new(&schema.get_attr_types());

        // Add random Zipf-distributed reports and create secret-shares of the reports.
        let report_count = 10000;
        let zipf_parameter = 1.0;
        report_vector.push_many_zipf::<OsRng>(report_count, zipf_parameter, false);
        let report_vector_share = report_vector.share(&mut rng);

        // We will first create a histogram for attr1.
        let attr_name = "attr1";

        // At this point `report_vector` and `report_vector_share` are secret-shares of
        // the reports. These are given to the first and second server, respectively.
        // The third server is not given any input.
        let mut server1 = Server::<U32_SIZE>::new(Role::First, schema.clone());
        server1.add_reports(report_vector).unwrap();
        let mut server2 = Server::<U32_SIZE>::new(Role::Second, schema.clone());
        server2.add_reports(report_vector_share).unwrap();
        let mut server3 = Server::<U32_SIZE>::new(Role::Third, schema.clone());

        // Both Server1 and Server2 add noise. They share with each other the numbers of noise
        // reports they added, so they can both pad their report vectors to the same size.
        let server1_noise_reports = server1
            .add_noise_reports(&mut rng, attr_name, noise_distr, noise_distr)
            .unwrap();
        server2.add_empty_reports(server1_noise_reports).unwrap();

        let server2_noise_reports = server2
            .add_noise_reports(&mut rng, attr_name, noise_distr, noise_distr)
            .unwrap();
        server1.add_empty_reports(server2_noise_reports).unwrap();

        // The three servers need to perform an oblivious shuffle. First, each pair of servers
        // agrees on a random seed. For example, here Server1 and Server2 agree on seed12.
        let seed12 = rng.gen::<[u8; 32]>();
        let seed23 = rng.gen::<[u8; 32]>();
        let seed13 = rng.gen::<[u8; 32]>();

        // Each server needs to call the `oblivious_permute` function. They communicate the
        // output to the next server in order S2->S1->S3. The last server receives no output.
        let to_server1 = server2
            .oblivious_permute::<StdRng>(seed23.clone(), seed12.clone(), None)
            .unwrap();
        let to_server3 = server1
            .oblivious_permute::<StdRng>(seed12.clone(), seed13.clone(), to_server1)
            .unwrap();
        server3
            .oblivious_permute::<StdRng>(seed13.clone(), seed23.clone(), to_server3)
            .unwrap();

        // Server2 and Server3 rotate their roles for the next round of the protocol.
        // Server1's role remains the same.
        server2.rotate_role().unwrap();
        server3.rotate_role().unwrap();

        // We rename the servers for simplicity here.
        let temp = server3;
        let mut server3 = server2;
        let mut server2 = temp;

        // Next, Server1 and Server2 reveal secret shares for the attribute for which
        // they want to compute the histogram.
        let server1_attr_shares = server1.get_attr_values(attr_name).unwrap();
        let server2_attr_shares = server2.get_attr_values(attr_name).unwrap();

        // After the shares as exchanged, the attribute values can be revealed.
        server1.reveal_attr(attr_name, server2_attr_shares).unwrap();
        server2.reveal_attr(attr_name, server1_attr_shares).unwrap();

        // Next, both parties prune their reports with the same threshold. This removes
        // all reports with a value for attr1 that occurs less than `threshold` times.
        let removed1 = server1.prune(attr_name, threshold).unwrap();
        let removed2 = server2.prune(attr_name, threshold).unwrap();

        // If everything went correctly, both servers removed the same number of reports.
        assert_eq!(removed1, removed2);

        // Server1 will create a new histogram object.
        let histogram = server1
            .make_histogram(attr_name, noise_distr.m(), threshold)
            .unwrap();

        // For the next round of layered histogram, suppose the analyst wants to zoom in
        // only on those reports where attr1 == 0. We use `split_at` to separate those
        // reports from the rest. Note that `split_at` removes the attribute from both
        // the input and the output `Server` objects.
        let mut server1 = server1.split_at(attr_name, 0).unwrap();
        let mut server2 = server2.split_at(attr_name, 0).unwrap();

        // Server3 removes the already handled attribute from its schema.
        server3.remove_attr(attr_name).unwrap();

        // Next, suppose we want to look at attr2 when attr1 == 0. We repeat the same steps.
        let attr_name = "attr2";

        // Add the noise reports and pad the report vectors.
        let server1_noise_reports = server1
            .add_noise_reports(&mut rng, attr_name, noise_distr, noise_distr)
            .unwrap();
        server2.add_empty_reports(server1_noise_reports).unwrap();
        let server2_noise_reports = server2
            .add_noise_reports(&mut rng, attr_name, noise_distr, noise_distr)
            .unwrap();
        server1.add_empty_reports(server2_noise_reports).unwrap();

        // Choose seeds and perform the oblivious shuffle.
        let seed12 = rng.gen::<[u8; 32]>();
        let seed23 = rng.gen::<[u8; 32]>();
        let seed13 = rng.gen::<[u8; 32]>();

        let to_server1 = server2
            .oblivious_permute::<StdRng>(seed23.clone(), seed12.clone(), None)
            .unwrap();
        let to_server3 = server1
            .oblivious_permute::<StdRng>(seed12.clone(), seed13.clone(), to_server1)
            .unwrap();
        server3
            .oblivious_permute::<StdRng>(seed13.clone(), seed23.clone(), to_server3)
            .unwrap();

        // Rotate the roles.
        server2.rotate_role().unwrap();
        server3.rotate_role().unwrap();

        // Rename the servers for simplicity.
        let temp = server3;
        let mut server3 = server2;
        let mut server2 = temp;

        // Exchange and reveal the secret shares for attr2.
        let server1_attr_shares = server1.get_attr_values(attr_name).unwrap();
        let server2_attr_shares = server2.get_attr_values(attr_name).unwrap();

        server1.reveal_attr(attr_name, server2_attr_shares).unwrap();
        server2.reveal_attr(attr_name, server1_attr_shares).unwrap();

        // Prune the report vectors.
        let removed1 = server1.prune(attr_name, threshold).unwrap();
        let removed2 = server2.prune(attr_name, threshold).unwrap();
        assert_eq!(removed1, removed2);

        // We already have a `Histogram` object that we want to add this data for attr2
        // as a sub-histogram.
        let sub_histogram = server1
            .make_histogram(attr_name, noise_distr.m(), threshold)
            .unwrap();

        // Joining the histograms is done with the `join_at` function. Here the join_at
        // value means that `sub_histogram` is for attr1 == 0. This way we can patch
        // together (sub-)histograms into a tree-like structure. The `join_at` function
        // returns (wrapped inn Rc<RefCell<_>>) a reference to the newly joined sub-histogram.
        // This way we can in the next round join to this `sub_histogram`.
        let sub_histogram = histogram.borrow_mut().join_at(0, sub_histogram).unwrap();

        // Next, suppose the analyst wants to zoom in on those reports where attr2 == 1.
        // We split off those reports to constitute our `Server` objects for the last round.
        let mut server1 = server1.split_at(attr_name, 1).unwrap();
        let mut server2 = server2.split_at(attr_name, 1).unwrap();
        server3.remove_attr(attr_name).unwrap();

        // The only attribute left is `attr3`.
        let attr_name = "attr3";

        let server1_noise_reports = server1
            .add_noise_reports(&mut rng, attr_name, noise_distr, noise_distr)
            .unwrap();
        server2.add_empty_reports(server1_noise_reports).unwrap();
        let server2_noise_reports = server2
            .add_noise_reports(&mut rng, attr_name, noise_distr, noise_distr)
            .unwrap();
        server1.add_empty_reports(server2_noise_reports).unwrap();

        let seed12 = rng.gen::<[u8; 32]>();
        let seed23 = rng.gen::<[u8; 32]>();
        let seed13 = rng.gen::<[u8; 32]>();

        let to_server1 = server2
            .oblivious_permute::<StdRng>(seed23.clone(), seed12.clone(), None)
            .unwrap();
        let to_server3 = server1
            .oblivious_permute::<StdRng>(seed12.clone(), seed13.clone(), to_server1)
            .unwrap();
        server3
            .oblivious_permute::<StdRng>(seed13.clone(), seed23.clone(), to_server3)
            .unwrap();

        server2.rotate_role().unwrap();
        server3.rotate_role().unwrap();

        // Rename the servers for simplicity here.
        let temp = server3;
        // let mut server3 = server2;
        let mut server2 = temp;

        let server1_attr_shares = server1.get_attr_values(attr_name).unwrap();
        let server2_attr_shares = server2.get_attr_values(attr_name).unwrap();
        server1.reveal_attr(attr_name, server2_attr_shares).unwrap();
        server2.reveal_attr(attr_name, server1_attr_shares).unwrap();

        let removed1 = server1.prune(attr_name, threshold).unwrap();
        let removed2 = server2.prune(attr_name, threshold).unwrap();
        assert_eq!(removed1, removed2);

        // Finally, patch together the last sub-histogram.
        let sub_sub_histogram = server1
            .make_histogram(attr_name, noise_distr.m(), threshold)
            .unwrap();
        sub_histogram
            .borrow_mut()
            .join_at(0, sub_sub_histogram)
            .unwrap();

        // Our `histogram` object represents now a layered histogram as follows:
        //
        //                           o     [root node]
        //                          /|\
        //                         / | \
        //                        0  1 ... [attr1 values with counts]
        //                       /|\
        //                      / | \
        //                     0  1 ...    [attr2 values with counts]
        //                       /|\
        //                      / | \
        //                     0  1 ...    [attr3 values with counts]
        //
        // The `histogram` object is a reference to the root node. We can now call
        // `Histogram::get_all_counts` to get the counts for each attribute value,
        // or `Histogram::get_count` to get the count for a specific value.
        //
        // We can explore the layered histogram structure by using `Histogram::filter`
        // to get a reference to a sub-histogram attached at a given attribute value.
    }

    #[test]
    fn histogram_with_summation() {
        // In this example we have a numerical attribute in addition to two
        // categorical attributes. The numerical attribute is 8 bits and has
        // a modulus of 201. This means the largest valid input value for the
        // attribute is floor(201/2) = 100.
        let schema =
            Schema::try_from(r#"[["attr1","c6"],["attr2",{"n8": 201}],["attr3","c8"]]"#).unwrap();

        // The total bit-size of the attributes is 6+8+8=22 bits, so we only need
        // a single u32 to store them all.
        const U32_SIZE: usize = 1;

        // Parameters for this protocol run. The `summation_modulus` must be some
        // sufficiently large odd positive integer that the sum is guaranteed to
        // not exceed. Here we choose the largest possible value for a u32.
        let summation_modulus = !SummationModulus::zero();

        // This is the truncation point for our noise distribution.
        let m: i64 = -20;

        // In this example we use Gaussian-DP. Here we set the standard deviation
        // to be 0.1. Again, for determining the total epsilon, see the paper at
        // https://eprint.iacr.org/2021/1490.
        let s = 0.1;
        let noise_distr = Gaussian::new(m, s).unwrap();
        let dummy_noise_distr = Gaussian::new(m, s).unwrap();

        // We prune everything with count less than 40.
        let threshold: usize = 40;

        // All test data is sampled from a Zipf distribution with this parameter.
        let zipf_parameter = 1.2;
        let report_count = 1000000;

        let mut rng: OsRng = OsRng;

        // Create a `ReportVector` with the three attributes and sample random data.
        let mut report_vector = ReportVector::<U32_SIZE>::new(&schema.get_attr_types());
        report_vector.push_many_zipf::<OsRng>(report_count, zipf_parameter, true);

        // Before secret-sharing this, for the purposes of this test we compute the sum
        // of the numerical attribute values in the clear. We can compare to this to
        // check for correctness.
        let true_sum = report_vector
            .get_attr_iter(1)
            .fold(SummationModulus::zero(), |acc, x| {
                summation_modulus.add_mod(acc, x.into())
            });

        // Now secret-share the reports.
        let report_vector_share = report_vector.share(&mut rng);

        // As before, we create three servers. Server1 and Server2 hold the two shares
        // of the reports. Server3 holds no input.
        let mut server1 = Server::<U32_SIZE>::new(Role::First, schema.clone());
        server1.add_reports(report_vector).unwrap();
        let mut server2 = Server::<U32_SIZE>::new(Role::Second, schema.clone());
        server2.add_reports(report_vector_share).unwrap();
        let mut server3 = Server::<U32_SIZE>::new(Role::Third, schema.clone());

        // This lambda-function performs the sum protocol. The function returns the
        // outputs for Server1 and Server2, as well as the total communication in bytes.
        // The two servers' outputs are secret-shares of the sum.
        let sum_protocol = |server1: &mut Server<U32_SIZE>,
                            server2: &mut Server<U32_SIZE>,
                            server3: &mut Server<U32_SIZE>,
                            attr_name: &str,
                            summation_modulus: SummationModulus|
         -> (SummationModulus, SummationModulus, usize) {
            // We keep track of the communication in this variable.
            let mut communication = 0usize;
            let mut rng = OsRng;

            // This is the number of reports. We need it later.
            let report_count = server1.get_report_count().unwrap();

            // Each server must independently call `summation_initialize`.
            server1
                .summation_initialize(&mut rng, attr_name, summation_modulus)
                .unwrap();
            server2
                .summation_initialize(&mut rng, attr_name, summation_modulus)
                .unwrap();
            server3
                .summation_initialize(&mut rng, attr_name, summation_modulus)
                .unwrap();

            // The servers need to call the following sequence of functions and communicate
            // the outputs of the functions to each other, as indicated by the function names
            // below. All objects are serializable, as is demonstrated here. We serialize here
            // only to measure the communication cost of the protocol.
            let from_s3_to_s2 = server3.summation_create_key().unwrap();
            communication += bincode::serialize(&from_s3_to_s2).unwrap().len();

            let from_s3_to_s1 = server3
                .summation_create_seeds(&mut rng, report_count)
                .unwrap();
            communication += bincode::serialize(&from_s3_to_s1).unwrap().len();

            server2.summation_receive_key(from_s3_to_s2).unwrap();

            let from_s1_to_s2 = server1
                .summation_create_masked_bits::<StdRng>(from_s3_to_s1)
                .unwrap();
            communication += bincode::serialize(&from_s1_to_s2).unwrap().len();

            let (from_s2_to_s1, s2_output) = server2
                .summation_create_reveal_msgs::<StdRng>(&mut rng, from_s1_to_s2)
                .unwrap();
            communication += bincode::serialize(&from_s2_to_s1).unwrap().len();

            let s1_output = server1
                .summation_receive_reveal_msgs(from_s2_to_s1)
                .unwrap();

            server1.summation_finalize();
            server2.summation_finalize();
            server3.summation_finalize();

            // At this point, `s1_output` and `s2_output` hold the result for Server1
            // and Server2, respectively. In practice, we need to add some noise to these
            // to make them differentially private. This is simple, so we omit it here, and
            // instead just return the two shares and the total communication.
            (s1_output, s2_output, communication)
        };

        // This lambda-function executes the histogram protocol, finds the most commonly
        // occuring value for the given attribute, and removes all other reports from
        // the servers given to it.
        let filter_to_attr_with_most_frequent_value =
            |server1: &mut Server<U32_SIZE>,
             server2: &mut Server<U32_SIZE>,
             server3: &mut Server<U32_SIZE>,
             attr_name: &str| {
                let mut rng = OsRng;

                // Add noise reports and a corresponding number of empty reports.
                let server1_noise_reports = server1
                    .add_noise_reports(&mut rng, attr_name, noise_distr, dummy_noise_distr)
                    .unwrap();
                server2.add_empty_reports(server1_noise_reports).unwrap();
                let server2_noise_reports = server2
                    .add_noise_reports(&mut rng, attr_name, noise_distr, dummy_noise_distr)
                    .unwrap();
                server1.add_empty_reports(server2_noise_reports).unwrap();

                // Perform the oblivious shuffle protocol.
                let seed12 = rng.gen::<[u8; 32]>();
                let seed23 = rng.gen::<[u8; 32]>();
                let seed13 = rng.gen::<[u8; 32]>();

                let to_server1 = server2
                    .oblivious_permute::<StdRng>(seed23.clone(), seed12.clone(), None)
                    .unwrap();
                let to_server3 = server1
                    .oblivious_permute::<StdRng>(seed12.clone(), seed13.clone(), to_server1)
                    .unwrap();
                server3
                    .oblivious_permute::<StdRng>(seed13.clone(), seed23.clone(), to_server3)
                    .unwrap();

                server2.rotate_role().unwrap();
                server3.rotate_role().unwrap();

                // Rename the servers for simplicity.
                let temp = server3.clone();
                *server3 = server2.clone();
                *server2 = temp;

                // Exchange shares and reveal the attribute values.
                let server1_attr_shares = server1.get_attr_values(attr_name).unwrap();
                let server2_attr_shares = server2.get_attr_values(attr_name).unwrap();
                server1.reveal_attr(attr_name, server2_attr_shares).unwrap();
                server2.reveal_attr(attr_name, server1_attr_shares).unwrap();

                // Prune reports with rarely occurring attribute value.
                let removed1 = server1.prune(attr_name, threshold).unwrap();
                let removed2 = server2.prune(attr_name, threshold).unwrap();
                assert_eq!(removed1, removed2);

                // Server1 computes the histogram.
                let histogram = server1
                    .make_histogram(attr_name, noise_distr.m(), threshold)
                    .unwrap();

                // Find the most commonly occurring value for the attribute.
                let most_common_attr_value = histogram
                    .borrow()
                    .get_all_counts()
                    .iter()
                    .max_by_key(|(_, &v)| v)
                    .map(|(k, _)| *k)
                    .unwrap();

                // Split off sub-servers for those records where the attribute has
                // the most common value.
                *server1 = server1.split_at(attr_name, most_common_attr_value).unwrap();
                *server2 = server2.split_at(attr_name, most_common_attr_value).unwrap();
                server3.remove_attr(attr_name).unwrap();
            };

        // The actual example really starts here. It proceeds by repeating two steps:
        // 1) Summing the numerical attribute value. After each execution, we compare to the
        //    true sum (in the first iteration) or to the previous sum (in subsequent iterations).
        //    We verify that the sum is decreasing when we filter down into sub-histograms.
        // 2) Filtering down to the most common value of one of the categorical attributes.

        // Round 1: Sum the numerical attribute values for attr2.
        let start = Instant::now();
        let (s1_output, s2_output, communication) = sum_protocol(
            &mut server1,
            &mut server2,
            &mut server3,
            "attr2",
            summation_modulus,
        );
        let duration = start.elapsed();
        println!(
            "Time to run sum protocol on {} reports: {:?}",
            server1.get_report_count().unwrap(),
            duration
        );
        println!("Communication for sum protocol: {}", communication);
        let private_sum = summation_modulus.add_mod(s1_output, s2_output);
        println!("Private sum: {}; true sum: {}", private_sum, true_sum);
        assert_eq!(private_sum, true_sum);

        // Filter down to most common attr1 value.
        filter_to_attr_with_most_frequent_value(&mut server1, &mut server2, &mut server3, "attr1");

        // Round 2: Sum the numerical attribute values for attr2.
        let start = Instant::now();
        let (s1_output, s2_output, communication) = sum_protocol(
            &mut server1,
            &mut server2,
            &mut server3,
            "attr2",
            summation_modulus,
        );
        let duration = start.elapsed();
        println!(
            "Time to run sum protocol on {} reports: {:?}",
            server1.get_report_count().unwrap(),
            duration
        );
        println!("Communication for sum protocol: {}", communication);
        let previous_private_sum = private_sum;
        let private_sum = summation_modulus.add_mod(s1_output, s2_output);
        println!(
            "Private sum: {}; previous private sum: {}",
            private_sum, previous_private_sum
        );
        assert!(private_sum < previous_private_sum);

        // Filter down to most common attr3 value.
        filter_to_attr_with_most_frequent_value(&mut server1, &mut server2, &mut server3, "attr3");

        // Round 3: Sum the numerical attribute values.
        let start = Instant::now();
        let (s1_output, s2_output, communication) = sum_protocol(
            &mut server1,
            &mut server2,
            &mut server3,
            "attr2",
            summation_modulus,
        );
        let duration = start.elapsed();
        println!(
            "Time to run sum protocol on {} reports: {:?}",
            server1.get_report_count().unwrap(),
            duration
        );
        println!("Communication for sum protocol: {}", communication);
        let previous_private_sum = private_sum;
        let private_sum = summation_modulus.add_mod(s1_output, s2_output);
        println!(
            "Private sum: {}; previous private sum: {}",
            private_sum, previous_private_sum
        );
        assert!(private_sum < previous_private_sum);

        // There are no more categorical attributes left.
    }

    // This is a helper function for the next examples. It builds a complete
    // non-private histogram from a `ReportVector`.
    fn build_histogram<const REPORT_U32_SIZE: usize>(
        schema: Schema,
        mut report_vector: ReportVector<REPORT_U32_SIZE>,
    ) -> Option<Rc<RefCell<Histogram>>> {
        // Nothing left to do.
        if schema.get_attr_types().len() == 0 {
            return None;
        }

        let attr_names = schema.get_attr_names();
        let attr_name = attr_names[0].clone();
        let attr_index = schema.get_attr_index(&attr_name).unwrap();

        let histogram = Histogram::new(
            &schema,
            &attr_name,
            report_vector.get_attr_iter(attr_index),
            0,
            0,
        )
        .unwrap();
        let all_values = histogram
            .borrow()
            .get_all_counts()
            .keys()
            .copied()
            .collect::<Vec<_>>();

        let mut sub_schema = schema.clone();
        sub_schema.remove_attr(&attr_name);

        // Next create the sub-histograms if the sub-schema is non-empty.
        if sub_schema.len() > 0 {
            for attr_value in all_values {
                let sub_report_vector = report_vector.split_at(attr_index, attr_value);
                let sub_histogram = build_histogram(sub_schema.clone(), sub_report_vector);
                if sub_histogram.is_some() {
                    histogram
                        .borrow_mut()
                        .join_at(attr_value, sub_histogram.unwrap())
                        .unwrap();
                }
            }
        }

        Some(histogram)
    }

    // This is a helper struct that we use to pass protocol parameters in the
    // next examples.
    #[derive(Clone, Copy)]
    struct ProtocolParams<NoiseDistr: NoiseDistribution, NoiseDistrDummy: NoiseDistribution> {
        noise_distr: NoiseDistr,
        noise_distr_dummy: NoiseDistrDummy,
        reports_prune_threshold: usize,
        histogram_prune_threshold: usize,
    }

    // This is a helper function for the next examples. It builds a complete
    // histogram using the private histogram protocol given two `Server` holding
    // shares of the data. It returns the histogram and the communication in bytes.
    fn build_private_histogram<
        const REPORT_U32_SIZE: usize,
        NoiseDistr: NoiseDistribution,
        NoiseDistrDummy: NoiseDistribution,
    >(
        protocol_params: ProtocolParams<NoiseDistr, NoiseDistrDummy>,
        schema: Schema,
        mut server1: Server<REPORT_U32_SIZE>,
        mut server2: Server<REPORT_U32_SIZE>,
    ) -> (Option<Rc<RefCell<Histogram>>>, usize) {
        let mut communication = 0usize;

        // Nothing left to do.
        if schema.get_attr_types().len() == 0 {
            return (None, communication);
        }

        let reports_prune_threshold = protocol_params.reports_prune_threshold;
        let histogram_prune_threshold = protocol_params.histogram_prune_threshold;
        let noise_distr = protocol_params.noise_distr;
        let noise_distr_dummy = protocol_params.noise_distr_dummy;

        assert_eq!(server1.get_role(), Role::First);
        assert_eq!(server2.get_role(), Role::Second);
        assert_eq!(server1.get_report_count(), server2.get_report_count());

        let mut rng = OsRng;
        let attr_names = schema.get_attr_names();
        // let attr_name = attr_names[rng.gen_range(0..attr_names.len())].clone();
        let attr_name = attr_names[0].clone();

        let server1_noise_reports = server1
            .add_noise_reports(&mut rng, &attr_name, noise_distr, noise_distr_dummy)
            .unwrap();
        server2.add_empty_reports(server1_noise_reports).unwrap();
        let server2_noise_reports = server2
            .add_noise_reports(&mut rng, &attr_name, noise_distr, noise_distr_dummy)
            .unwrap();
        server1.add_empty_reports(server2_noise_reports).unwrap();

        // The three servers need to perform an oblivious shuffle.
        let seed12 = rng.gen::<[u8; 32]>();
        let seed23 = rng.gen::<[u8; 32]>();
        let seed13 = rng.gen::<[u8; 32]>();

        let to_server1 = server2
            .oblivious_permute::<StdRng>(seed23.clone(), seed12.clone(), None)
            .unwrap();
        server2.rotate_role().unwrap();
        communication += bincode::serialize(&to_server1).unwrap().len();

        let to_server3 = server1
            .oblivious_permute::<StdRng>(seed12.clone(), seed13.clone(), to_server1)
            .unwrap();
        server1.rotate_role().unwrap();
        communication += bincode::serialize(&to_server3).unwrap().len();

        let mut server3 = Server::new(Role::Third, schema.clone());
        server3
            .oblivious_permute::<StdRng>(seed13.clone(), seed23.clone(), to_server3)
            .unwrap();
        server3.rotate_role().unwrap();

        let server1_attr_shares = server1.get_attr_values(&attr_name).unwrap();
        let server3_attr_shares = server3.get_attr_values(&attr_name).unwrap();
        communication += bincode::serialize(&server1_attr_shares).unwrap().len();
        communication += bincode::serialize(&server3_attr_shares).unwrap().len();

        server1
            .reveal_attr(&attr_name, server3_attr_shares)
            .unwrap();
        server3
            .reveal_attr(&attr_name, server1_attr_shares)
            .unwrap();

        let removed1 = server1.prune(&attr_name, reports_prune_threshold).unwrap();
        let removed2 = server3.prune(&attr_name, reports_prune_threshold).unwrap();
        assert_eq!(removed1, removed2);

        // Done with this histogram.
        let histogram = server1
            .make_histogram(&attr_name, noise_distr.m(), histogram_prune_threshold)
            .unwrap();

        let mut sub_schema = schema.clone();
        sub_schema.remove_attr(&attr_name);

        // Next create the sub-histograms if the sub-schema is non-empty.
        if sub_schema.len() > 0 {
            let all_values = histogram
                .borrow()
                .get_all_counts()
                .keys()
                .copied()
                .collect::<Vec<_>>();
            for attr_value in all_values {
                // println!(
                //     "server1 report count: {}",
                //     server1.get_report_count().unwrap()
                // );
                // println!(
                //     "Exploring {} at value {} (count = {})",
                //     attr_name,
                //     attr_value,
                //     histogram.borrow().get_count(attr_value)
                // );
                let sub_server1 = server1.split_at(&attr_name, attr_value).unwrap();
                let sub_server3 = server3.split_at(&attr_name, attr_value).unwrap();
                let (sub_histogram, sub_communication) = build_private_histogram(
                    protocol_params,
                    sub_schema.clone(),
                    sub_server1,
                    sub_server3,
                );
                communication += sub_communication;

                if sub_histogram.is_some() {
                    histogram
                        .borrow_mut()
                        .join_at(attr_value, sub_histogram.unwrap())
                        .unwrap();
                }
            }
        }

        (Some(histogram), communication)
    }

    // This is a helper function for the next examples. It uses the private histogram
    // protocol to find the heavy-hitter, i.e., the most commonly occurring report,
    // for the data shared by the two `Server`s. It returns the histogram it explored,
    // the count for the heavy-hitter, and the communication in bytes.
    fn private_heavy_hitter_internal<
        const REPORT_U32_SIZE: usize,
        NoiseDistr: NoiseDistribution,
        NoiseDistrDummy: NoiseDistribution,
    >(
        protocol_params: ProtocolParams<NoiseDistr, NoiseDistrDummy>,
        schema: Schema,
        mut server1: Server<REPORT_U32_SIZE>,
        mut server2: Server<REPORT_U32_SIZE>,
        hh_count: usize,
    ) -> (Option<Rc<RefCell<Histogram>>>, usize, usize) {
        let mut communication = 0usize;

        // Nothing left to do.
        if schema.get_attr_types().len() == 0 {
            return (None, 0, communication);
        }

        let mut hh_count = hh_count;

        let reports_prune_threshold = protocol_params.reports_prune_threshold;
        let histogram_prune_threshold = protocol_params.histogram_prune_threshold;
        let noise_distr = protocol_params.noise_distr;
        let noise_distr_dummy = protocol_params.noise_distr_dummy;

        assert_eq!(server1.get_role(), Role::First);
        assert_eq!(server2.get_role(), Role::Second);
        assert_eq!(server1.get_report_count(), server2.get_report_count());

        let mut rng = OsRng;
        let attr_names = schema.get_attr_names();
        let attr_name = attr_names[0].clone();

        let server1_noise_reports = server1
            .add_noise_reports(&mut rng, &attr_name, noise_distr, noise_distr_dummy)
            .unwrap();
        server2.add_empty_reports(server1_noise_reports).unwrap();
        let server2_noise_reports = server2
            .add_noise_reports(&mut rng, &attr_name, noise_distr, noise_distr_dummy)
            .unwrap();
        server1.add_empty_reports(server2_noise_reports).unwrap();

        // The three servers need to perform an oblivious shuffle.
        let seed12 = rng.gen::<[u8; 32]>();
        let seed23 = rng.gen::<[u8; 32]>();
        let seed13 = rng.gen::<[u8; 32]>();

        let to_server1 = server2
            .oblivious_permute::<StdRng>(seed23.clone(), seed12.clone(), None)
            .unwrap();
        server2.rotate_role().unwrap();
        communication += bincode::serialize(&to_server1).unwrap().len();

        let to_server3 = server1
            .oblivious_permute::<StdRng>(seed12.clone(), seed13.clone(), to_server1)
            .unwrap();
        server1.rotate_role().unwrap();
        communication += bincode::serialize(&to_server3).unwrap().len();

        let mut server3 = Server::new(Role::Third, schema.clone());
        server3
            .oblivious_permute::<StdRng>(seed13.clone(), seed23.clone(), to_server3)
            .unwrap();
        server3.rotate_role().unwrap();

        let server1_attr_shares = server1.get_attr_values(&attr_name).unwrap();
        let server3_attr_shares = server3.get_attr_values(&attr_name).unwrap();
        communication += bincode::serialize(&server1_attr_shares).unwrap().len();
        communication += bincode::serialize(&server3_attr_shares).unwrap().len();

        server1
            .reveal_attr(&attr_name, server3_attr_shares)
            .unwrap();
        server3
            .reveal_attr(&attr_name, server1_attr_shares)
            .unwrap();

        let adjusted_reports_prune_threshold = max(reports_prune_threshold, hh_count);
        let removed1 = server1
            .prune(&attr_name, adjusted_reports_prune_threshold)
            .unwrap();
        let removed2 = server3
            .prune(&attr_name, adjusted_reports_prune_threshold)
            .unwrap();
        assert_eq!(removed1, removed2);

        // Done with this histogram.
        let adjusted_histogram_prune_threshold = max(histogram_prune_threshold, hh_count);
        let histogram = server1
            .make_histogram(
                &attr_name,
                noise_distr.m(),
                adjusted_histogram_prune_threshold,
            )
            .unwrap();

        let mut sub_schema = schema.clone();
        sub_schema.remove_attr(&attr_name);

        // If no attributes are left, then update hh_count.
        if sub_schema.len() == 0 {
            let max_count = histogram
                .borrow()
                .get_all_counts()
                .values()
                .copied()
                .max()
                .unwrap_or(0);
            if max_count > hh_count {
                hh_count = max_count;
            }
        }

        // Next create the sub-histograms if the sub-schema is non-empty.
        if sub_schema.len() > 0 {
            // We want to go over the keys in the way of largest value (count) first.
            let mut all_values_and_counts = Vec::from_iter(histogram.borrow().get_all_counts());
            all_values_and_counts.sort_by(|&(_, a), &(_, b)| b.cmp(&a));
            let all_values = all_values_and_counts
                .iter()
                .map(|&kv| kv.0)
                .collect::<Vec<_>>();
            for attr_value in all_values {
                // Exit the loop if the count in this branch is less than the current hh_count.
                if histogram.borrow().get_count(attr_value) <= hh_count {
                    break;
                }

                // println!(
                //     "server1 report count: {}",
                //     server1.get_report_count().unwrap()
                // );
                // println!(
                //     "Exploring {} at value {} (count = {})",
                //     attr_name,
                //     attr_value,
                //     histogram.borrow().get_count(attr_value)
                // );
                let sub_server1 = server1.split_at(&attr_name, attr_value).unwrap();
                let sub_server3 = server3.split_at(&attr_name, attr_value).unwrap();
                let (sub_histogram, sub_hh_count, sub_communication) =
                    private_heavy_hitter_internal(
                        protocol_params,
                        sub_schema.clone(),
                        sub_server1,
                        sub_server3,
                        hh_count,
                    );
                communication += sub_communication;

                if sub_histogram.is_some() {
                    histogram
                        .borrow_mut()
                        .join_at(attr_value, sub_histogram.unwrap())
                        .unwrap();

                    if sub_hh_count > hh_count {
                        hh_count = sub_hh_count;
                        println!("Updated heavy hitter count to {}", hh_count);
                    }
                }
            }
        }

        (Some(histogram), hh_count, communication)
    }

    // This is a helper function for the next examples. It initiates the recursive
    // calls to `private_heavy_hitter_internal`. It returns the histogram it explored,
    // the count for the heavy-hitter, and the communication in bytes.
    fn private_heavy_hitter<
        const REPORT_U32_SIZE: usize,
        NoiseDistr: NoiseDistribution,
        NoiseDistrDummy: NoiseDistribution,
    >(
        protocol_params: ProtocolParams<NoiseDistr, NoiseDistrDummy>,
        schema: Schema,
        server1: Server<REPORT_U32_SIZE>,
        server2: Server<REPORT_U32_SIZE>,
    ) -> (Option<Rc<RefCell<Histogram>>>, usize, usize) {
        private_heavy_hitter_internal::<REPORT_U32_SIZE, NoiseDistr, NoiseDistrDummy>(
            protocol_params,
            schema,
            server1,
            server2,
            0,
        )
    }

    // This is a helper function for the next examples. It finds the top-`n`
    // heavy-hitters from a given `ReportVector`, i.e., the top-`n` most frequently
    // occurring reports. It returns a vector of `(count, report)` pairs.
    fn heavy_hitter_n<const REPORT_U32_SIZE: usize>(
        report_vector: ReportVector<REPORT_U32_SIZE>,
        n: usize,
    ) -> Vec<(usize, Report<REPORT_U32_SIZE>)> {
        let mut report_map = HashMap::new();
        for report in report_vector.iter() {
            let counter = report_map.entry(report.as_u32_slice()).or_insert(0);
            *counter += 1;
        }

        // Now sort by the count and for top-n return the count and the report.
        let n = min(n, report_map.len());
        let mut report_counts = Vec::new();
        for (report, count) in report_map {
            report_counts.push((count, Report::from_u32_slice(report)));
        }
        report_counts.sort_by(|a, b| b.0.cmp(&a.0));
        report_counts.resize(n, (0, Report::from_u32_slice(&[0; REPORT_U32_SIZE])));

        report_counts
    }

    // This is a helper function for the next examples. It compares two layered
    // histograms and returns the maximum difference between the counts at any node.
    fn compare_histograms(
        true_histogram: Rc<RefCell<Histogram>>,
        approximate_histogram: Rc<RefCell<Histogram>>,
    ) -> usize {
        // Check that the histograms are compatible.
        assert_eq!(
            true_histogram.borrow().get_attr_type(),
            approximate_histogram.borrow().get_attr_type()
        );
        assert_eq!(
            true_histogram.borrow().get_attr_name(),
            approximate_histogram.borrow().get_attr_name()
        );
        assert_eq!(
            true_histogram.borrow().get_schema(),
            approximate_histogram.borrow().get_schema()
        );

        // Iterate over all values.
        let mut error = 0usize;
        for (value, true_count) in true_histogram.borrow().get_all_counts() {
            let approximate_count = approximate_histogram.borrow().get_count(value);

            // What is the difference for this node?
            let mut local_error = true_count.abs_diff(approximate_count);

            // If possible, compute error also for sub-histograms.
            let true_sub_histogram = true_histogram.borrow().filter(value);
            let approximate_sub_histogram = approximate_histogram.borrow().filter(value);
            if true_sub_histogram.is_some() && approximate_sub_histogram.is_some() {
                local_error = max(
                    local_error,
                    compare_histograms(
                        true_sub_histogram.unwrap(),
                        approximate_sub_histogram.unwrap(),
                    ),
                );
            }

            // Is the error bigger than what we've seen before?
            error = max(error, local_error);
        }

        error
    }

    // This is a helper function for the next examples. It computes the
    // total number of nodes in a layered histogram tree.
    fn total_nodes_internal(histogram: Rc<RefCell<Histogram>>) -> usize {
        let mut total = 0;
        for (value, _) in histogram.borrow().get_all_counts() {
            let sub_histogram = histogram.borrow().filter(value);
            if sub_histogram.is_some() {
                total += 1 + total_nodes_internal(sub_histogram.unwrap());
            }
        }
        total
    }

    // This is a helper function for the next examples. It initiates the
    // recursive calls to `total_nodes_internal`.
    fn total_nodes(histogram: Rc<RefCell<Histogram>>) -> usize {
        let empty = histogram.borrow().get_total_count() == 0;
        if empty {
            0
        } else {
            1 + total_nodes_internal(histogram)
        }
    }

    #[test]
    fn full_histogram_exploration() {
        // This example explores the full histogram structure for a generated test dataset
        // of reports. In other words, in order for all attributes, it computes the histogram
        // and recursively sub-histograms at each occurring value.

        let schema =
            Schema::try_from(r#"[["attr1","c3"], ["attr2","c4"], ["attr3","c5"], ["attr4","c6"]]"#)
                .unwrap();

        const U32_COUNT: usize = 1;

        // We collect the protocol parameters here for convenience.
        let protocol_params = ProtocolParams {
            noise_distr: Gaussian::new(-30, 5.0).unwrap(),
            noise_distr_dummy: Gaussian::new(-200, 20.0).unwrap(),
            reports_prune_threshold: 500,
            histogram_prune_threshold: 500,
        };

        let mut rng = OsRng;

        let start = Instant::now();
        let mut report_vector = ReportVector::<U32_COUNT>::new(&schema.get_attr_types());

        let report_count = 1000000;
        let zipf_parameter = 1.0;
        let randomize_zipf = true;
        report_vector.push_many_zipf::<OsRng>(report_count, zipf_parameter, randomize_zipf);

        let duration = start.elapsed();
        println!("Time to create data: {:?}", duration);

        let start = Instant::now();
        let histogram = build_histogram(schema.clone(), report_vector.clone()).unwrap();
        let duration = start.elapsed();
        println!("Time to build plaintext histogram: {:?}", duration);
        println!(
            "Total nodes in full histogram: {}",
            total_nodes(histogram.clone())
        );

        let report_vector_share = report_vector.share(&mut rng);
        let start = Instant::now();
        let mut server1 = Server::new(Role::First, schema.clone());
        server1.add_reports(report_vector).unwrap();
        let duration = start.elapsed();
        println!("Time to set data (per server): {:?}", duration);

        let mut server2 = Server::new(Role::Second, schema.clone());
        server2.add_reports(report_vector_share).unwrap();

        let start = Instant::now();
        let (private_histogram, communication) =
            build_private_histogram(protocol_params, schema.clone(), server1, server2);
        let private_histogram = private_histogram.unwrap();
        let duration = start.elapsed();
        println!("Time to build private histogram: {:?}", duration);
        println!(
            "Total nodes in private histogram: {}",
            total_nodes(private_histogram.clone())
        );
        println!("Communication for private histogram: {}", communication);

        println!(
            "Absolute value of largest error in the private histogram: {}",
            compare_histograms(histogram.clone(), private_histogram.clone())
        );
    }

    #[test]
    #[ignore = "This test can be run manually. Input data should be in a file `full_histogram.json`."]
    fn full_histogram_exploration_from_file() {
        // This example explores the full histogram structure for a dataset of reports
        // loaded from a file `full_histogram.json`. In other words, in order for all
        // attributes, it computes the histogram and recursively sub-histograms at each
        // occurring value.

        // Make sure to set the `U32_COUNT` correctly here.
        const U32_COUNT: usize = 1;

        // We collect the protocol parameters here for convenience.
        let protocol_params = ProtocolParams {
            noise_distr: Gaussian::new(-30, 5.0).unwrap(),
            noise_distr_dummy: Gaussian::new(-200, 20.0).unwrap(),
            reports_prune_threshold: 500,
            histogram_prune_threshold: 500,
        };

        let mut rng = OsRng;

        let filename = "full_histogram.json";
        let json = fs::read_to_string(filename)
            .expect(format!("Failed to read file ({})", filename).as_str());
        let (schema, plain_report_vector, report_vector, report_vector_share) =
            create_report_shares::<U32_COUNT>(&mut rng, &json).unwrap();

        let start = Instant::now();
        let histogram = build_histogram(schema.clone(), plain_report_vector.clone()).unwrap();
        let duration = start.elapsed();
        println!("Time to build plaintext histogram: {:?}", duration);
        println!(
            "Total nodes in full histogram: {}",
            total_nodes(histogram.clone())
        );

        let start = Instant::now();
        let mut server1 = Server::new(Role::First, schema.clone());
        server1.add_reports(report_vector).unwrap();
        let duration = start.elapsed();
        println!("Time to set data (per server): {:?}", duration);

        let mut server2 = Server::new(Role::Second, schema.clone());
        server2.add_reports(report_vector_share).unwrap();

        let start = Instant::now();
        let (private_histogram, communication) =
            build_private_histogram(protocol_params, schema.clone(), server1, server2);
        let private_histogram = private_histogram.unwrap();
        let duration = start.elapsed();
        println!("Time to build private histogram: {:?}", duration);
        println!(
            "Total nodes in private histogram: {}",
            total_nodes(private_histogram.clone())
        );
        println!("Communication for private histogram: {}", communication);

        println!(
            "Absolute value of largest error in the private histogram: {}",
            compare_histograms(histogram.clone(), private_histogram.clone())
        );
    }

    #[test]
    fn heavy_hitter() {
        // This example finds the heavy-hitter, i.e., the most commonly occurring report
        // from a generated test dataset of reports using the private histogram protocol.

        let schema = Schema::try_from(
            r#"[["attr1","c16"], ["attr2","c16"], ["attr3","c16"], ["attr4","c16"]]"#,
        )
        .unwrap();

        const U32_COUNT: usize = 2;

        // We collect the protocol parameters here for convenience.
        let protocol_params = ProtocolParams {
            noise_distr: Gaussian::new(-128, 4.0).unwrap(),
            noise_distr_dummy: Gaussian::new(-30000, 8.0).unwrap(),
            reports_prune_threshold: 1200,
            histogram_prune_threshold: 1200,
        };

        let mut rng = OsRng;

        let start = Instant::now();
        let mut report_vector = ReportVector::<U32_COUNT>::new(&schema.get_attr_types());

        let report_count = 1000000;
        let zipf_parameter = 1.5;
        let randomize_zipf = true;
        report_vector.push_many_zipf::<OsRng>(report_count, zipf_parameter, randomize_zipf);

        let duration = start.elapsed();
        println!("Time to create data: {:?}", duration);

        let true_heavy_hitters = heavy_hitter_n(report_vector.clone(), 10);
        println!("True heavy hitters:");
        for (count, report) in true_heavy_hitters {
            println!("{}: {:?}", count, report);
        }

        let report_vector_share = report_vector.share(&mut rng);

        let start = Instant::now();
        let mut server1 = Server::new(Role::First, schema.clone());
        server1.add_reports(report_vector).unwrap();
        let duration = start.elapsed();
        println!("Time to set data (per server): {:?}", duration);

        let mut server2 = Server::new(Role::Second, schema.clone());
        server2.add_reports(report_vector_share).unwrap();

        let start = Instant::now();
        let (private_histogram, hh_count, communication) =
            private_heavy_hitter(protocol_params, schema.clone(), server1, server2);
        let duration = start.elapsed();
        println!("Time to find private heavy hitter: {:?}", duration);
        println!("Heavy hitter count: {}", hh_count);
        println!(
            "Total nodes explored in private histogram: {}",
            total_nodes(private_histogram.unwrap().clone())
        );
        println!("Communication for private histogram: {}", communication);
    }

    #[test]
    #[ignore = "This test can be run manually. Input data should be in a file `heavy_hitter.json`."]
    fn heavy_hitter_from_file() {
        // This example finds the heavy-hitter, i.e., the most commonly occurring report
        // from a dataset of reports loaded from a file `heavy_hitter.json` using the
        // private histogram protocol.

        // Make sure to set the `U32_COUNT` correctly here.
        const U32_COUNT: usize = 2;

        // We collect the protocol parameters here for convenience.
        let protocol_params = ProtocolParams {
            noise_distr: Gaussian::new(-128, 4.0).unwrap(),
            noise_distr_dummy: Gaussian::new(-30000, 8.0).unwrap(),
            reports_prune_threshold: 1200,
            histogram_prune_threshold: 1200,
        };

        let mut rng = OsRng;

        let filename = "heavy_hitter.json";
        let json = fs::read_to_string(filename)
            .expect(format!("Failed to read file ({})", filename).as_str());
        let (schema, plain_report_vector, report_vector, report_vector_share) =
            create_report_shares::<U32_COUNT>(&mut rng, &json).unwrap();

        let true_heavy_hitters = heavy_hitter_n(plain_report_vector.clone(), 10);
        println!("True heavy hitters:");
        for (count, report) in true_heavy_hitters {
            println!("{}: {:?}", count, report);
        }

        let start = Instant::now();
        let mut server1 = Server::new(Role::First, schema.clone());
        server1.add_reports(report_vector).unwrap();
        let duration = start.elapsed();
        println!("Time to set data (per server): {:?}", duration);

        let mut server2 = Server::new(Role::Second, schema.clone());
        server2.add_reports(report_vector_share).unwrap();

        let start = Instant::now();
        let (private_histogram, hh_count, communication) =
            private_heavy_hitter(protocol_params, schema.clone(), server1, server2);
        let duration = start.elapsed();
        println!("Time to find private heavy hitter: {:?}", duration);
        println!("Heavy hitter count: {}", hh_count);
        println!(
            "Total nodes explored in private histogram: {}",
            total_nodes(private_histogram.unwrap().clone())
        );
        println!("Communication for private histogram: {}", communication);
    }
}
