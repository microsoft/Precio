// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use precio::ReportVector;
use precio::Schema;
use rand::rngs::OsRng;

fn main() {
    // We demonstrate a schema consisting of three categorical attributes.
    let schema =
        Schema::try_from(r#"[["Country","c8"],["Browser","c3"],["Spend",{"n11": 2001}]]"#).unwrap();

    // The total bit-size of the attributes is 2+10+4=16 bits, so a single u32 suffices.
    const U32_SIZE: usize = 1;

    let mut rng = OsRng;

    // Create a `ReportVector` from this schema. This represents the clients' reports.
    let mut report_vector = ReportVector::<U32_SIZE>::new(&schema.get_attr_types());

    // Add random Zipf-distributed reports and create secret-shares of the reports.
    let report_count = 10_000_000;
    let zipf_parameter = 1.1;
    report_vector.push_many_zipf::<OsRng>(report_count, zipf_parameter, true);

    // Serialize the report_vector to JSON. Write to plain_data.json.
    // Make this so that the JSON is nicely formatted for viewing.
    let json = report_vector.to_json_pretty(&schema).unwrap();
    std::fs::write("plain_data.json", json).unwrap();

    // Now split the report_vector and serialize the two shares.
    let report_vector_share = report_vector.share(&mut rng);
    let json1 = report_vector.to_json_pretty(&schema).unwrap();
    let json2 = report_vector_share.to_json_pretty(&schema).unwrap();
    std::fs::write("plain_data_share1.json", json1).unwrap();
    std::fs::write("plain_data_share2.json", json2).unwrap();
}
