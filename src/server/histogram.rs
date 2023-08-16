// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use crate::report::attr::AttrValueType;
use crate::schema::*;
use std::cell::RefCell;
use std::cmp::max;
use std::collections::BTreeMap;
use std::iter::Iterator;
use std::rc::Rc;

pub struct Histogram {
    schema: Schema,
    attr_name: String,
    attr_type: AttributeType,
    values: BTreeMap<Attribute, (usize, Option<Rc<RefCell<Histogram>>>)>,
}

impl Histogram {
    /// Create a new `Histogram` from a `Schema`, an attribute name, and
    /// a vector of attribute values. Each attribute count is corrected by
    /// adding `m` to it.
    pub fn new(
        schema: &Schema,
        attr_name: &str,
        attr_values_iter: impl ExactSizeIterator<Item = AttrValueType>,
        m: i64,
        threshold: usize,
    ) -> Result<Rc<RefCell<Self>>, String> {
        // Check that the schema is valid.
        if !schema.is_valid() {
            return Err(format!("The schema {:?} is invalid.", schema));
        }

        // Get the attribute type and check that the attribute is categorical.
        let attr_type = attr_type_from_attr_name(schema, attr_name)?;
        if !attr_type.is_categorical() {
            return Err(format!("Attribute {} (type {:?}) is not categorical and cannot be used to create a histogram.", attr_name, attr_type));
        }

        // The histogram `Schema` contains all of the attributes, minus
        // the current one.
        let mut new_schema = schema.clone();
        new_schema.remove_attr(attr_name);

        // Add values to the histogram.
        let mut new_histogram = Self {
            schema: new_schema,
            attr_name: attr_name.to_string(),
            attr_type,
            values: BTreeMap::new(),
        };

        let attrs_iter =
            attr_values_iter.map(|attr_value| attr_from_attr_value(attr_type, attr_value));
        for attr in attrs_iter {
            // Check for an error value input.
            let attr = attr?;

            // Increment the count for this attribute in `values`.
            new_histogram
                .values
                .entry(attr)
                .and_modify(|(count, sub_histogram)| {
                    *count += 1;
                    *sub_histogram = None;
                })
                .or_insert((1, None));
        }

        // Correct the counts by adding `m` to each.
        // Retain only those attribute values that are positive.
        new_histogram.values.retain(|_, (count, _)| {
            *count = (max::<i64>(*count as i64 + 2 * m, 0) as u64) as usize;

            // Even if threshold is zero, we have to prune empty nodes.
            *count >= max::<usize>(threshold, 1)
        });

        Ok(Rc::new(RefCell::new(new_histogram)))
    }

    /// Return the name of the attribute that this histogram is for.
    pub fn get_attr_name(&self) -> &str {
        &self.attr_name
    }

    /// Return the type of the attribute that this histogram is for.
    pub fn get_attr_type(&self) -> AttributeType {
        self.attr_type
    }

    /// Return the `Schema` for this histogram.
    pub fn get_schema(&self) -> Schema {
        self.schema.clone()
    }

    /// Get the count for a given attribute value.
    pub fn get_count(&self, attr_value: AttrValueType) -> usize {
        // If the given value is invalid, simply return 0.
        let attr = attr_from_attr_value(self.attr_type, attr_value);
        if attr.is_err() {
            return 0usize;
        }

        // Otherwise, we have an `Attribute` and can get the count.
        let attr_details = self.values.get(&attr.unwrap());
        if attr_details.is_some() {
            attr_details.unwrap().0
        } else {
            0usize
        }
    }

    /// Get all counts in this histogram.
    pub fn get_all_counts(&self) -> BTreeMap<AttrValueType, usize> {
        self.values
            .iter()
            .map(|(attr, (count, _))| (attr.get_value(), *count))
            .collect()
    }

    /// Get the sum of all counts in this histogram.
    pub fn get_total_count(&self) -> usize {
        self.values.iter().map(|(_, (count, _))| count).sum()
    }

    /// Get the sub-histogram for a given attribute value.
    pub fn filter(&self, attr_value: AttrValueType) -> Option<Rc<RefCell<Histogram>>> {
        // If the given value is invalid, simply return `None`.
        let attr = attr_from_attr_value(self.attr_type, attr_value);
        if attr.is_err() {
            return None;
        }

        // In this case, we know `attr_value` is valid.
        let attr = attr.unwrap();

        // Check whether we have any data for this attribute value?
        let attr_details = self.values.get(&attr);
        if attr_details.is_none() {
            return None;
        }

        // Check whether we have a sub-histogram?
        let attr_details = attr_details.unwrap();
        let sub_histogram = &attr_details.1;
        if sub_histogram.is_none() {
            return None;
        }

        // If we have a sub-histogram, return it.
        Some(Rc::clone(sub_histogram.as_ref().unwrap()))
    }

    /// Add data to the histogram.
    fn add<I>(&mut self, attr_values: I) -> Result<(), String>
    where
        I: Iterator<Item = Result<Attribute, String>>,
    {
        for attr in attr_values {
            // Check for an error value input.
            let attr = attr?;

            // Increment the count for this attribute in `values`.
            self.values
                .entry(attr)
                .and_modify(|(count, sub_histogram)| {
                    *count += 1;
                    *sub_histogram = None;
                })
                .or_insert((1, None));
        }

        Ok(())
    }

    pub fn join_at(
        &mut self,
        join_at: AttrValueType,
        histogram: Rc<RefCell<Histogram>>,
    ) -> Result<Rc<RefCell<Histogram>>, String> {
        // Does this attribute value appear at all, i.e., is there anything to put in a sub-histogram?
        let attr_count = self.get_count(join_at);
        if attr_count == 0 {
            return Err("Attribute value does not appear in the histogram.".to_string());
        }

        let sub_histogram = histogram.borrow_mut();

        // For a noiseless recursive histogram, the sub-histogram count should match with `attr_count`.
        // However, since our histograms are noisy, the counts will never match exactly. Instead, we
        // omit this check.
        // let sub_histogram_count = histogram.get_total_count();
        // if sub_histogram_count != attr_count {
        //     return Err(format!("The sub-histogram total count {} does not match the count {} for the given attribute.", sub_histogram_count, attr_count));
        // }

        // If we remove the sub-histogram attribute from the current `Schema`, do we get the sub-histogram `Schema`?
        let mut sub_schema = self.schema.clone();
        sub_schema.remove_attr(&sub_histogram.attr_name);
        if sub_schema != sub_histogram.schema {
            return Err(format!(
                "The sub-histogram schema {:?} is incompatible with the current schema {:?}.",
                sub_histogram.schema, self.schema
            ));
        }

        // In this case, we know `split_at` is valid.
        let attr = attr_from_attr_value(self.attr_type, join_at).unwrap();

        // Set the sub-histogram for this attribute.
        self.values.entry(attr).and_modify(|(_, sub_histogram)| {
            *sub_histogram = Some(histogram.clone());
        });

        Ok(self.filter(join_at).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::Schema;

    #[test]
    fn test_new() {
        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
        ]);

        let attr_name = "attr1";
        let attr_values: [u32; 10] = [0, 1, 2, 1, 1, 0, 0, 1, 3, 1];
        let histogram =
            Histogram::new(&schema, attr_name, attr_values.iter().copied(), 0, 0).unwrap();

        assert_eq!(histogram.borrow().get_attr_name(), attr_name);
        assert_eq!(histogram.borrow().get_attr_type(), AttributeType::C2);
        assert_eq!(histogram.borrow().get_count(0), 3);
        assert_eq!(histogram.borrow().get_count(1), 5);
        assert_eq!(histogram.borrow().get_count(2), 1);
        assert_eq!(histogram.borrow().get_count(3), 1);

        let all_counts_map = histogram.borrow().get_all_counts();
        assert_eq!(all_counts_map.len(), 4);
        assert_eq!(all_counts_map.get(&0), Some(&3));
        assert_eq!(all_counts_map.get(&1), Some(&5));
        assert_eq!(all_counts_map.get(&2), Some(&1));
        assert_eq!(all_counts_map.get(&3), Some(&1));
    }

    #[test]
    #[should_panic(expected = "Given value 4 is invalid for the attribute type C2.")]
    fn test_new_panic() {
        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
        ]);

        let attr_name = "attr1";
        let attr_values: [u32; 10] = [0, 1, 2, 4, 1, 0, 0, 1, 3, 1];
        #[allow(unused_variables)]
        let histogram =
            Histogram::new(&schema, attr_name, attr_values.iter().copied(), 0, 0).unwrap();
    }

    #[test]
    fn test_split_at() {
        let mut schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
            ("attr3".to_string(), AttributeType::C3),
        ]);

        let attr_values10: [u32; 10] = [0, 1, 2, 1, 1, 0, 0, 1, 3, 1];
        let attr_values3: [u32; 3] = [4, 5, 5];

        let histogram =
            Histogram::new(&schema, "attr1", attr_values10.iter().copied(), 0, 0).unwrap();
        assert_eq!(histogram.borrow().get_attr_name(), "attr1");

        // Create an attach a sub-histogram at "attr1" == 0.
        schema.remove_attr("attr1");
        let sub_histogram =
            Histogram::new(&schema, "attr2", attr_values3.iter().copied(), 0, 0).unwrap();
        histogram.borrow_mut().join_at(0, sub_histogram).unwrap();
    }
}
