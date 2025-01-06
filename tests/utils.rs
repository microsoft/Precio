// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#![allow(dead_code)]

use precio::hist_noise::NoiseDistribution;
use precio::server::{Server, SummationModulus};
use rand::rngs::{OsRng, StdRng};

// This is a helper struct that we use to pass protocol parameters
// in some of the examples.
#[derive(Clone, Copy)]
pub(crate) struct ProtocolParams<NoiseDistr: NoiseDistribution, NoiseDistrDummy: NoiseDistribution>
{
    pub(crate) noise_distr: NoiseDistr,
    pub(crate) noise_distr_dummy: NoiseDistrDummy,
    pub(crate) reports_prune_threshold: usize,
    pub(crate) histogram_prune_threshold: usize,
}

// This function performs the sum protocol. The function returns the
// outputs for Server1 and Server2, as well as the total communication in bytes.
// The two servers' outputs are secret-shares of the sum.
pub(crate) fn sum_protocol<const U32_SIZE: usize>(
    server1: &mut Server<U32_SIZE>,
    server2: &mut Server<U32_SIZE>,
    server3: &mut Server<U32_SIZE>,
    attr_name: &str,
    summation_modulus: SummationModulus,
) -> (SummationModulus, SummationModulus, usize) {
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
}
