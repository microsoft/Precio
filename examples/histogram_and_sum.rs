// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#![allow(unused_mut)]

use precio::arith::{Modulus, Zero};
use precio::hist_noise::{Gaussian, NoiseDistribution};
use precio::server::{Role, Server, SummationModulus};
use precio::ReportVector;
use rand::rngs::{OsRng, StdRng};
use rand::Rng;

fn main() {
    // The `create_data` example creates a file `plain_data.json` with contents (for example):
    //
    // {
    //     "schema": [["Country", "c8"], ["Browser", "c3"], ["Spend", {"n11": 2001}]],
    //     "reports": [
    //       { "attributes": [{ "c8": 110 }, { "c3": 6 }, { "n11": [480, 2001] }] },
    //       { "attributes": [{ "c8": 15  }, { "c3": 0 }, { "n11": [880, 2001] }] },
    //       ...
    //     ]
    // }
    //
    // It also creates two secret shares of this data in `plain_data_share1.json` and `plain_data_share2.json`.
    // This example assumes that `create_data` has been ran so that the data files exist.

    // Each report in this example is 22 bits, so can be encoded into a single `u32`.
    const U32_SIZE: usize = 1;

    // Read in `plain_data_share1.json` and `plain_data_share2.json`.
    let plain_data_share1_str = std::fs::read_to_string("plain_data_share1.json").unwrap();
    let (report_vector_share1, schema) =
        ReportVector::<U32_SIZE>::from_json(&plain_data_share1_str, true).unwrap();

    let plain_data_share2_str = std::fs::read_to_string("plain_data_share2.json").unwrap();
    let (report_vector_share2, _) =
        ReportVector::<U32_SIZE>::from_json(&plain_data_share2_str, true).unwrap();

    // Create three server instances, with roles `Role::First`, `Role::Second`, and `Role::Third`, respectively.
    let mut server1 = Server::<U32_SIZE>::new(Role::First, schema.clone());
    server1.add_reports(report_vector_share1).unwrap();

    let mut server2 = Server::<U32_SIZE>::new(Role::Second, schema.clone());
    server2.add_reports(report_vector_share2).unwrap();

    let mut server3 = Server::<U32_SIZE>::new(Role::Third, schema.clone());

    let mut rng = OsRng;

    // DP parameters resulting in (2, 2^{-40})-DP.
    let m_buckets: i64 = -50;
    let m_flush: i64 = -250;
    let sigma_buckets = 6.8;
    let sigma_flush = 20.0;
    let noise_distr_buckets = Gaussian::new(m_buckets, sigma_buckets).unwrap();
    let noise_distr_flush = Gaussian::new(m_flush, sigma_flush).unwrap();

    // This is the threshold for pruning. We do not care about values that show up fewer than
    // this many times (including added noise reports). The probability that a valid path
    // with more than 1000 *true* items is pruned is <= 0.01.
    let threshold: usize = 1053;

    // We will first create a histogram for "Country".
    let attr_name = "Country";

    // We measure the communication cost.
    let mut communication: usize = 0;

    // Both Server1 and Server2 add noise. They share with each other the numbers of noise
    // reports they added, so they can both pad their report vectors to the same size.
    let server1_noise_reports = server1
        .add_noise_reports(&mut rng, attr_name, noise_distr_buckets, noise_distr_flush)
        .unwrap();
    server2.add_empty_reports(server1_noise_reports).unwrap();

    let server2_noise_reports = server2
        .add_noise_reports(&mut rng, attr_name, noise_distr_buckets, noise_distr_flush)
        .unwrap();
    server1.add_empty_reports(server2_noise_reports).unwrap();

    // Obliviously permute the reports.
    let (mut server1, mut server2, mut server3, permute_communication) =
        oblivious_permute::<U32_SIZE>(server1, server2, server3);
    communication += permute_communication;

    // Next, Server1 and Server2 reveal secret shares for the attribute for which
    // they want to compute the histogram.
    let server1_attr_shares = server1.get_attr_values(attr_name).unwrap();
    let server2_attr_shares = server2.get_attr_values(attr_name).unwrap();

    // The shared must be communicated, so we add their sizes to the communication cost.
    communication += bincode::serialize(&server1_attr_shares).unwrap().len();
    communication += bincode::serialize(&server2_attr_shares).unwrap().len();

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
        .make_histogram(attr_name, noise_distr_buckets.m(), threshold)
        .unwrap();

    // Which country is the most frequent in the data?
    let (num_reports, most_frequent_country) =
        histogram.borrow().get_inverted_frequency_table(1)[0];
    println!(
        "Most frequent country index: {most_frequent_country} (number of reports {num_reports})"
    );

    // Report the communication cost.
    println!(
        "Communication for \"Country\" histogram: {} MB",
        communication >> 20
    );
    communication = 0;

    // For the next round of layered histogram, suppose the analyst wants to zoom in
    // only on the most frequest country. We use `split_at` to separate those
    // reports from the rest. Note that `split_at` removes the attribute from both
    // the input and the output `Server` objects.
    let mut server1 = server1.split_at(attr_name, most_frequent_country).unwrap();
    let mut server2 = server2.split_at(attr_name, most_frequent_country).unwrap();

    // Server3 removes the already handled attribute from its schema.
    server3.remove_attr(attr_name).unwrap();

    // Next, suppose we want to look at "Browser" when "Country" is the most frequent value. We repeat the same steps.
    let attr_name = "Browser";

    // Add the noise reports and pad the report vectors.
    let server1_noise_reports = server1
        .add_noise_reports(&mut rng, attr_name, noise_distr_buckets, noise_distr_flush)
        .unwrap();
    server2.add_empty_reports(server1_noise_reports).unwrap();
    let server2_noise_reports = server2
        .add_noise_reports(&mut rng, attr_name, noise_distr_buckets, noise_distr_flush)
        .unwrap();
    server1.add_empty_reports(server2_noise_reports).unwrap();

    // Obliviously permute the reports.
    let (mut server1, mut server2, mut server3, permute_communication) =
        oblivious_permute::<U32_SIZE>(server1, server2, server3);
    communication += permute_communication;

    // Exchange and reveal the secret shares for attr2.
    let server1_attr_shares = server1.get_attr_values(attr_name).unwrap();
    let server2_attr_shares = server2.get_attr_values(attr_name).unwrap();

    // The shared must be communicated, so we add their sizes to the communication cost.
    communication += bincode::serialize(&server1_attr_shares).unwrap().len();
    communication += bincode::serialize(&server2_attr_shares).unwrap().len();

    server1.reveal_attr(attr_name, server2_attr_shares).unwrap();
    server2.reveal_attr(attr_name, server1_attr_shares).unwrap();

    // Prune the report vectors.
    let removed1 = server1.prune(attr_name, threshold).unwrap();
    let removed2 = server2.prune(attr_name, threshold).unwrap();
    assert_eq!(removed1, removed2);

    // We already have a `Histogram` object that we want to add this data for "Browser"
    // as a sub-histogram.
    let sub_histogram = server1
        .make_histogram(attr_name, noise_distr_buckets.m(), threshold)
        .unwrap();

    // Joining the histograms is done with the `join_at` function. Here the join_at
    // value means that `sub_histogram` is for "Country" being the most frequent one.
    // This way we can patch together (sub-)histograms into a tree-like structure. The `join_at` function
    // returns (wrapped inn Rc<RefCell<_>>) a reference to the newly joined sub-histogram.
    // This way we can in the next round join to this `sub_histogram`.
    let sub_histogram = histogram
        .borrow_mut()
        .join_at(most_frequent_country, sub_histogram)
        .unwrap();

    // Find the most frequent browser for the most frequent country.
    let most_frequent_browsers = sub_histogram.borrow().get_inverted_frequency_table(2);
    let (num_reports_most_freq_browser, most_frequent_browser) = most_frequent_browsers[0];
    println!("Most frequent browser index: {most_frequent_browser} (number of reports: {num_reports_most_freq_browser})");

    // Find the second most frequent browser for the most frequent country.
    let (num_reports_second_most_freq_browser, second_most_frequent_browser) =
        most_frequent_browsers[1];
    println!("Second most frequent browser index: {second_most_frequent_browser} (number of reports: {num_reports_second_most_freq_browser})");

    // Report the communication cost.
    println!(
        "Communication for \"Browser\" histogram: {} MB",
        communication >> 20
    );

    // Next, suppose the analyst wants to compute the average spend for people from the most frequent country using the two most frequent browsers.
    let mut server1_for_sum = server1.split_at(attr_name, most_frequent_browser).unwrap();
    let mut server2_for_sum = server2.split_at(attr_name, most_frequent_browser).unwrap();
    server3.remove_attr(attr_name).unwrap();

    // Run the sum protocol.
    let summation_modulus: SummationModulus = !SummationModulus::zero();
    let (sum_share1, sum_share2, communication) = sum_protocol(
        &mut server1_for_sum,
        &mut server2_for_sum,
        &mut server3,
        "Spend",
        summation_modulus,
    );

    // Server1 and Server2 then reveal the sum modulo the summation modulus.
    let sum = summation_modulus.add_mod(sum_share1, sum_share2);
    let avg_spend = sum as f64 / num_reports_most_freq_browser as f64;
    println!("Average spend (most frequent country, most frequent browser): {avg_spend}");

    // Communication cost for the first sum.
    println!(
        "Communication for sum protocol (most frequent browser): {} MB",
        communication >> 20
    );

    // Now repeat the same for the second most frequent browser.
    let mut server1_for_sum = server1
        .split_at(attr_name, second_most_frequent_browser)
        .unwrap();
    let mut server2_for_sum = server2
        .split_at(attr_name, second_most_frequent_browser)
        .unwrap();
    let (sum_share1, sum_share2, communication) = sum_protocol(
        &mut server1_for_sum,
        &mut server2_for_sum,
        &mut server3,
        "Spend",
        summation_modulus,
    );
    let sum = summation_modulus.add_mod(sum_share1, sum_share2);
    let avg_spend = sum as f64 / num_reports_second_most_freq_browser as f64;
    println!("Average spend (most frequent country, second most frequent browser): {avg_spend}");

    // Communication cost for the second sum.
    println!(
        "Communication for sum protocol (second most frequent browser): {} MB",
        communication >> 20
    );
}

/// This function performs the oblivious shuffle protocol. The function returns the
/// total communication in bytes. The function takes three servers as input, with roles
/// in order `Role::First`, `Role::Second`, `Role::Third`. The function permutes the reports
/// shared by `server1` and `server2`, resulting in `server1` and `server3` having the
/// permuted set of reports. It then swaps the roles of `server2` and `server3`.
fn oblivious_permute<const U32_SIZE: usize>(
    mut server1: Server<U32_SIZE>,
    mut server2: Server<U32_SIZE>,
    mut server3: Server<U32_SIZE>,
) -> (Server<U32_SIZE>, Server<U32_SIZE>, Server<U32_SIZE>, usize) {
    let mut communication: usize = 0;

    // Ensure server1 has `Role::First`, server2 has `Role::Second`, and server3 has `Role::Third`.
    assert_eq!(server1.get_role(), Role::First);
    assert_eq!(server2.get_role(), Role::Second);
    assert_eq!(server3.get_role(), Role::Third);

    let mut rng = OsRng;

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
    communication += bincode::serialize(&to_server1).unwrap().len();

    let to_server3 = server1
        .oblivious_permute::<StdRng>(seed12.clone(), seed13.clone(), to_server1)
        .unwrap();
    communication += bincode::serialize(&to_server3).unwrap().len();

    server3
        .oblivious_permute::<StdRng>(seed13.clone(), seed23.clone(), to_server3)
        .unwrap();

    // Server2 and Server3 rotate their roles for the next round of the protocol.
    // Server1's role remains the same.
    server2.rotate_role().unwrap();
    server3.rotate_role().unwrap();

    // Swap `server2` and `server3`.
    (server1, server3, server2, communication)
}

/// This function performs the sum protocol. The function returns the
/// outputs for Server1 and Server2, as well as the total communication in bytes.
/// The two servers' outputs are secret-shares of the sum.
fn sum_protocol<const U32_SIZE: usize>(
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
