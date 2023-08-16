// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#![allow(dead_code)]

use precio::hist_noise::NoiseDistribution;
use precio::server::Histogram;
use std::cell::RefCell;
use std::rc::Rc;

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

// This is a helper function for the next examples. It computes the
// total number of nodes in a layered histogram tree.
pub(crate) fn total_nodes_internal(histogram: Rc<RefCell<Histogram>>) -> usize {
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
pub(crate) fn total_nodes(histogram: Rc<RefCell<Histogram>>) -> usize {
    let empty = histogram.borrow().get_total_count() == 0;
    if empty {
        0
    } else {
        1 + total_nodes_internal(histogram)
    }
}
