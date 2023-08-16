// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use precio::client;
use precio::hist_noise::{Gaussian, NoiseDistribution};
use precio::server::{Histogram, Role, Server};
use precio::ReportVector;
use precio::Schema;
use rand::rngs::{OsRng, StdRng};
use rand::Rng;
use std::cell::RefCell;
use std::cmp::max;
use std::fs;
use std::rc::Rc;
use std::time::Instant;

mod utils;
use utils::ProtocolParams;

// This is a helper function that builds a complete non-private histogram
// from a `ReportVector`.
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

// This is a helper function that builds a complete histogram using the private
// histogram protocol given two `Server` holding shares of the data. It returns
// the histogram and the communication in bytes.
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

// This is a helper function that compares two layered histograms and returns
// the maximum difference between the counts at any node.
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
        utils::total_nodes(histogram.clone())
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
        utils::total_nodes(private_histogram.clone())
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
    let json =
        fs::read_to_string(filename).expect(format!("Failed to read file ({})", filename).as_str());
    let (schema, plain_report_vector, report_vector, report_vector_share) =
        client::create_report_shares::<U32_COUNT>(&mut rng, &json).unwrap();

    let start = Instant::now();
    let histogram = build_histogram(schema.clone(), plain_report_vector.clone()).unwrap();
    let duration = start.elapsed();
    println!("Time to build plaintext histogram: {:?}", duration);
    println!(
        "Total nodes in full histogram: {}",
        utils::total_nodes(histogram.clone())
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
        utils::total_nodes(private_histogram.clone())
    );
    println!("Communication for private histogram: {}", communication);

    println!(
        "Absolute value of largest error in the private histogram: {}",
        compare_histograms(histogram.clone(), private_histogram.clone())
    );
}
