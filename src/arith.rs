// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use num_modular::{ModularCoreOps, ModularUnaryOps};
use num_traits::{One, WrappingAdd, WrappingSub, Zero};
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::ops::BitAnd;

pub trait Modulus:
    Sized
    + PartialOrd
    + Copy
    + Zero
    + One
    + WrappingAdd
    + WrappingSub
    + From<bool>
    + Debug
    + BitAnd<Output = Self>
    + for<'t> ModularCoreOps<Self, &'t Self>
    + for<'t> ModularUnaryOps<&'t Self>
{
    /// Modular subtraction, assuming the inputs are already modulo the modulus.
    fn sub_mod(&self, a: Self, b: Self) -> Self {
        assert!(a < *self);
        assert!(b < *self);
        if b == Self::zero() {
            a
        } else {
            let diff = a.wrapping_sub(&b);
            if diff > a {
                diff.wrapping_add(self)
            } else {
                diff
            }
        }
    }

    /// Modular addition, assuming the inputs are already modulo the modulus.
    fn add_mod(&self, a: Self, b: Self) -> Self {
        assert!(a < *self);
        assert!(b < *self);
        if b == Self::zero() {
            a
        } else {
            self.sub_mod(a, *self - b)
        }
    }

    /// Modular multiplication, assuming the inputs are already modulo the modulus.
    fn mul_mod(&self, a: Self, b: Self) -> <Self as ModularCoreOps<Self, &'_ Self>>::Output {
        assert!(a < *self);
        assert!(b < *self);
        a.mulm(b, &self)
    }

    /// Modular inversion, assuming the input is already modulo the modulus.
    fn inv_mod(&self, a: Self) -> <Self as ModularUnaryOps<&'_ Self>>::Output {
        assert!(a < *self && a != Self::zero());
        a.invm(self).unwrap()
    }
}

impl Modulus for u8 {}
impl Modulus for u16 {}
impl Modulus for u32 {}
impl Modulus for u64 {}
impl Modulus for u128 {}
impl Modulus for usize {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_mod_b_zero() {
        assert_eq!(10u32.sub_mod(5, 0), 5);
    }

    #[test]
    fn test_sub_mod_diff_greater_than_a() {
        assert_eq!(20u32.sub_mod(5, 10), 15);
    }

    #[test]
    fn test_sub_mod_diff_less_than_or_equal_to_a() {
        assert_eq!(20u32.sub_mod(10, 5), 5);
    }

    #[test]
    fn test_add_mod_b_zero() {
        assert_eq!(10u32.add_mod(5, 0), 5);
    }

    #[test]
    fn test_add_mod_b_not_zero() {
        assert_eq!(10u32.add_mod(5, 2), 10u32.sub_mod(5, 10 - 2));
    }

    #[test]
    fn test_mod_mul() {
        assert_eq!(10u32.mul_mod(5, 2), 0u32);
    }

    #[test]
    fn test_mod_mul_minus_one_sq() {
        assert_eq!(65536u32.mul_mod(65535, 65535), 1u32);
    }

    #[test]
    fn test_mod_mul_one() {
        assert_eq!(65536u32.mul_mod(1, 34343), 34343u32);
    }
}
