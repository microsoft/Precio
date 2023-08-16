// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use precio::hist_noise::{Laplace, NoiseDistribution};
use precio::server::{Role, Server};
use precio::ReportVector;
use precio::Schema;
use rand::rngs::{OsRng, StdRng};
use rand::Rng;

#[test]
fn layered_histogram() {
    // We demonstrate a schema consisting of three categorical attributes.
    let schema = Schema::try_from(r#"[["attr1","c2"],["attr2","c10"],["attr3","c4"]]"#).unwrap();

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
