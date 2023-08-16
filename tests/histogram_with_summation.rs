// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use precio::arith::{Modulus, Zero};
use precio::hist_noise::{Gaussian, NoiseDistribution};
use precio::server::{Role, Server, SummationModulus};
use precio::ReportVector;
use precio::Schema;
use rand::rngs::{OsRng, StdRng};
use rand::Rng;
use std::time::Instant;

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
