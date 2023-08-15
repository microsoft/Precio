// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use crate::report::report_handler::ReportHandler;
pub use crate::report::report_vector::ReportVector;
use crate::schema::{Attribute, Schema};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::result::Result;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
struct RawReport {
    attributes: Vec<Attribute>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
struct RawReportVector {
    schema: Schema,
    reports: Vec<RawReport>,
}

pub fn create_report_shares<const REPORT_U32_SIZE: usize>(
    rng: &mut (impl Rng + ?Sized),
    input_json: &str,
) -> Result<
    (
        Schema,
        ReportVector<REPORT_U32_SIZE>, // Original report vector.
        ReportVector<REPORT_U32_SIZE>, // First half of shares.
        ReportVector<REPORT_U32_SIZE>, // Second half of shares.
    ),
    String,
> {
    // Load the input JSON into a RawReportVector.
    let reports_json_res: Result<RawReportVector, _> = serde_json::from_str(&input_json);
    let raw_reports: RawReportVector;
    match reports_json_res {
        Ok(r) => {
            raw_reports = r;
        }
        Err(e) => return Err(format!("Error parsing input JSON: {}.", e.to_string())),
    }

    // Extract the schema and reports from the RawReportVector.
    let schema = raw_reports.schema;
    let reports = raw_reports.reports;

    // Check that the schema is valid.
    if !schema.is_valid() {
        return Err(format!("Invalid schema: {:?}.", schema));
    }

    // From the schema, compute a vector of attribute sizes. Ensure these are valid
    // and create a `ReportVector`.
    let attr_types = schema.get_attr_types();
    if !ReportHandler::<REPORT_U32_SIZE>::is_valid_attr_types(&attr_types) {
        return Err(format!("Invalid attribute types: {:?}.", attr_types));
    }
    let mut report_vector = ReportVector::<REPORT_U32_SIZE>::new(&attr_types);
    let report_handler = report_vector.report_handler().clone();

    // Get a vector of dummy attribute values from the `ReportHandler`.
    let dummy_attr_values = report_handler.get_dummy_attr_values();

    for report in reports.iter() {
        // Check that each report is compatible with the schema.
        if !schema.is_compatible_attr_array(&report.attributes) {
            return Err(format!(
                "Report {:?} is invalid for the schema {:?}.",
                report, schema
            ));
        }

        // For categorical attributes, check that the clients' values are not dummy
        // values. For numerical attributes this is not necessary.
        if report
            .attributes
            .iter()
            .zip(dummy_attr_values.iter().copied())
            .filter(|(&attr, _)| attr.get_type().is_categorical())
            .any(|(attr, dummy_value)| attr.get_value() == dummy_value)
        {
            return Err(format!(
                "Report {:?} contains dummy values {:?} for categorical attributes.",
                report, dummy_attr_values
            ));
        }

        // Accumulate them to the `ReportVector`.
        let attr_values = report
            .attributes
            .iter()
            .map(|attr| attr.get_value())
            .collect::<Vec<_>>();
        let report = report_handler.create_report(&attr_values);
        report_vector.push(report);
    }

    // Share the `ReportVector` and return the shares.
    let mut report_vector_share1 = report_vector.clone();
    let report_vector_share2 = report_vector_share1.share(rng);
    Ok((
        schema,
        report_vector,
        report_vector_share1,
        report_vector_share2,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_categorical_serialize_deserialize_json() {
        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
            ("attr3".to_string(), AttributeType::C4),
            ("attr4".to_string(), AttributeType::C15),
        ]);
        let reports = vec![
            RawReport {
                attributes: vec![
                    Attribute::C2(0b10),
                    Attribute::C3(0b011),
                    Attribute::C4(0b0101),
                    Attribute::C15(0b110_1101_1011_0110),
                ],
            },
            RawReport {
                attributes: vec![
                    Attribute::C2(0b00),
                    Attribute::C3(0b010),
                    Attribute::C4(0b1101),
                    Attribute::C15(0b000_1111_0011_0101),
                ],
            },
            RawReport {
                attributes: vec![
                    Attribute::C2(0b1),
                    Attribute::C3(0b100),
                    Attribute::C4(0b0101),
                    Attribute::C15(0b010_0011_0000_0010),
                ],
            },
        ];
        let raw_report_vector = RawReportVector { schema, reports };
        let json = serde_json::to_string(&raw_report_vector).unwrap();
        let raw_report_vector2: RawReportVector = serde_json::from_str(&json).unwrap();
        assert_eq!(raw_report_vector, raw_report_vector2);

        let json = r#"
            {
                "schema":[["attr1","c2"],["attr2","c3"],["attr3","c4"],["attr4","c15"]],
                "reports":[
                    {"attributes":[{"c2":2},{"c3":3},{"c4":5},{"c15":28086}]},
                    {"attributes":[{"c2":0},{"c3":2},{"c4":13},{"c15":3893}]},
                    {"attributes":[{"c2":1},{"c3":4},{"c4":5},{"c15":8962}]}
                ]
            }"#;
        let raw_report_vector: RawReportVector = serde_json::from_str(&json).unwrap();
        assert_eq!(raw_report_vector, raw_report_vector2);
    }

    #[test]
    fn test_mixed_serialize_deserialize_json() {
        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::N3(6)),
            ("attr3".to_string(), AttributeType::C4),
            ("attr4".to_string(), AttributeType::N15(32767)),
        ]);
        let reports = vec![
            RawReport {
                attributes: vec![
                    Attribute::C2(0b10),
                    Attribute::N3(5, 6),
                    Attribute::C4(0b0101),
                    Attribute::N15(32766, 32767),
                ],
            },
            RawReport {
                attributes: vec![
                    Attribute::C2(0b00),
                    Attribute::N3(0, 6),
                    Attribute::C4(0b1101),
                    Attribute::N15(0, 32767),
                ],
            },
            RawReport {
                attributes: vec![
                    Attribute::C2(0b1),
                    Attribute::N3(3, 6),
                    Attribute::C4(0b0101),
                    Attribute::N15(16384, 32767),
                ],
            },
        ];
        let raw_report_vector = RawReportVector { schema, reports };
        let json = serde_json::to_string(&raw_report_vector).unwrap();
        let raw_report_vector2: RawReportVector = serde_json::from_str(&json).unwrap();
        assert_eq!(raw_report_vector, raw_report_vector2);

        let json = r#"
            {
                "schema":[["attr1","c2"],["attr2",{"n3":6}],["attr3","c4"],["attr4",{"n15":32767}]],
                "reports":[
                    {"attributes":[{"c2":2},{"n3":[5,6]},{"c4":5},{"n15": [32766,32767]}]},
                    {"attributes":[{"c2":0},{"n3":[0,6]},{"c4":13},{"n15":[0,32767]}]},
                    {"attributes":[{"c2":1},{"n3":[3,6]},{"c4":5},{"n15":[16384,32767]}]}
                ]
            }"#;
        let raw_report_vector: RawReportVector = serde_json::from_str(&json).unwrap();
        assert_eq!(raw_report_vector, raw_report_vector2);
    }

    #[test]
    fn test_categorical_create_report_shares() {
        let json = r#"
            {
                "schema":[["attr1","c2"],["attr2","c3"],["attr3","c4"],["attr4","c15"]],
                "reports":[
                    {"attributes":[{"c2":2},{"c3":3},{"c4":5},{"c15":28086}]},
                    {"attributes":[{"c2":0},{"c3":2},{"c4":13},{"c15":3893}]},
                    {"attributes":[{"c2":1},{"c3":4},{"c4":5},{"c15":8962}]}
                ]
            }"#;

        let mut rng = OsRng;
        let (schema, _, mut share1, share2) = create_report_shares::<1>(&mut rng, json).unwrap();
        assert!(schema.is_valid());
        assert_ne!(share1, share2);
        assert_eq!(share1.len(), share2.len());
        share1.reveal(share2);

        let report_handler = share1.report_handler();
        assert_eq!(
            *share1.get(0).unwrap(),
            report_handler.create_report(&[2, 3, 5, 28086])
        );
        assert_eq!(
            *share1.get(1).unwrap(),
            report_handler.create_report(&[0, 2, 13, 3893])
        );
        assert_eq!(
            *share1.get(2).unwrap(),
            report_handler.create_report(&[1, 4, 5, 8962])
        );
    }

    #[test]
    fn test_mixed_create_report_shares() {
        let json = r#"
            {
                "schema":[["attr1","c2"],["attr2",{"n3":6}],["attr3","c4"],["attr4",{"n15":32767}]],
                "reports":[
                    {"attributes":[{"c2":2},{"n3":[5,6]},{"c4":5},{"n15": [32766,32767]}]},
                    {"attributes":[{"c2":0},{"n3":[0,6]},{"c4":13},{"n15":[0,32767]}]},
                    {"attributes":[{"c2":1},{"n3":[3,6]},{"c4":5},{"n15":[16384,32767]}]}
                ]
            }"#;

        let mut rng = OsRng;
        let (schema, _, mut share1, share2) = create_report_shares::<1>(&mut rng, json).unwrap();
        assert!(schema.is_valid());
        assert_ne!(share1, share2);
        assert_eq!(share1.len(), share2.len());
        share1.reveal(share2);

        let report_handler = share1.report_handler();
        assert_eq!(
            *share1.get(0).unwrap(),
            report_handler.create_report(&[2, 5, 5, 32766])
        );
        assert_eq!(
            *share1.get(1).unwrap(),
            report_handler.create_report(&[0, 0, 13, 0])
        );
        assert_eq!(
            *share1.get(2).unwrap(),
            report_handler.create_report(&[1, 3, 5, 16384])
        );
    }
}
