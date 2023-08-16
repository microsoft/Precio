// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use precio::client;
use precio::hist_noise::{Gaussian, NoiseDistribution};
use precio::server::{Histogram, Role, Server};
use precio::Report;
use precio::ReportVector;
use precio::Schema;
use rand::rngs::{OsRng, StdRng};
use rand::Rng;
use std::cell::RefCell;
use std::cmp::{max, min};
use std::collections::HashMap;
use std::fs;
use std::rc::Rc;
use std::time::Instant;

mod utils;
use utils::ProtocolParams;

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
            let (sub_histogram, sub_hh_count, sub_communication) = private_heavy_hitter_internal(
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

#[test]
fn heavy_hitter() {
    // This example finds the heavy-hitter, i.e., the most commonly occurring report
    // from a generated test dataset of reports using the private histogram protocol.

    let schema =
        Schema::try_from(r#"[["attr1","c16"], ["attr2","c16"], ["attr3","c16"], ["attr4","c16"]]"#)
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
        utils::total_nodes(private_histogram.unwrap().clone())
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
    let json =
        fs::read_to_string(filename).expect(format!("Failed to read file ({})", filename).as_str());
    let (schema, plain_report_vector, report_vector, report_vector_share) =
        client::create_report_shares::<U32_COUNT>(&mut rng, &json).unwrap();

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
        utils::total_nodes(private_histogram.unwrap().clone())
    );
    println!("Communication for private histogram: {}", communication);
}
