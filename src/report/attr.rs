// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use crate::random::RandomMod;
use rand::Rng;
use std::cmp::min;

/// Represents the value for an attribute.
pub type AttrValueType = u32;
pub const MAX_ATTR_BIT_SIZE: usize = 32;

/// Create a random attribute value. A zero `modulus` indicates
/// a categorical attribute value. Otherwise the returned value is
/// reduced modulo `modulus`.
pub(crate) fn random_attr_value<R: Rng + ?Sized>(
    rng: &mut R,
    bit_size: usize,
    modulus: u32,
) -> AttrValueType {
    assert!(
        0 < bit_size && bit_size <= MAX_ATTR_BIT_SIZE,
        "random_attr_value: Attribute size cannot be 0 and must be at most MAX_ATTR_BIT_SIZE."
    );
    assert!(
        modulus >> min(31, bit_size) == 0,
        "random_attr_value: Modulus is too large."
    );

    if modulus == 0 {
        rng.gen::<AttrValueType>() >> (MAX_ATTR_BIT_SIZE - bit_size)
    } else {
        modulus.random_mod(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn test_random_attr_value() {
        let mut rng = OsRng;

        let attr_value = random_attr_value(&mut rng, 1, 0);
        assert!((attr_value >> 1) == 0);

        let attr_value1 = random_attr_value(&mut rng, 32, 0);
        let attr_value2 = random_attr_value(&mut rng, 32, 0);
        assert_ne!(attr_value1, attr_value2);

        let attr_value = random_attr_value(&mut rng, 1, 1);
        assert!(attr_value == 0);

        let attr_value = random_attr_value(&mut rng, 2, 3);
        assert!(attr_value <= 2);

        let attr_value = random_attr_value(&mut rng, 10, 128);
        assert!(attr_value <= 128);
    }

    #[test]
    #[should_panic(
        expected = "random_attr_value: Attribute size cannot be 0 and must be at most MAX_ATTR_BIT_SIZE."
    )]
    fn test_random_attr_value_panic1() {
        let mut rng = OsRng;
        random_attr_value(&mut rng, 0, 0);
    }

    #[test]
    #[should_panic(
        expected = "random_attr_value: Attribute size cannot be 0 and must be at most MAX_ATTR_BIT_SIZE."
    )]
    fn test_random_attr_value_panic2() {
        let mut rng = OsRng;
        random_attr_value(&mut rng, MAX_ATTR_BIT_SIZE + 1, 0);
    }

    #[test]
    #[should_panic(expected = "random_attr_value: Modulus is too large.")]
    fn test_random_attr_value_panic3() {
        let mut rng = OsRng;
        random_attr_value(&mut rng, 5, 32);
    }
}
