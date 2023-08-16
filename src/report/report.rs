// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use rand::Rng;
use serde::{Deserialize, Serialize};

/// A `Report` represents data for a single report from a client. It consists
/// of a bit-field encoding multiple attributes of different sizes. The attribute
/// sizes are not encoded in the `Report` itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Report<const U32_SIZE: usize> {
    #[serde(with = "serde_arrays")]
    data: [u32; U32_SIZE],
}

impl<const U32_SIZE: usize> Report<U32_SIZE> {
    /// Creates a new `Report` with all bits set to 0.
    pub(crate) fn new() -> Self {
        Self {
            data: [0; U32_SIZE],
        }
    }

    /// Creates a new random `Report`.
    pub(crate) fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut random_report = Self::new();
        rng.fill(random_report.data.as_mut_slice());
        random_report
    }

    /// Creates a new `Report` by setting its u32 parts to given values.
    pub fn from_u32_slice(value: &[u32]) -> Self {
        assert!(
            value.len() <= U32_SIZE,
            "Report::from_u32_slice: slice length is greater than the report size"
        );

        // Copy as many u32s as possible to a new Report.
        let mut ret = Self::new();
        ret.data
            .iter_mut()
            .zip(value.iter())
            .for_each(|(ret, value)| *ret = *value);
        ret
    }

    /// Creates a new `Report` by setting its lowest u32 part to given value.
    pub(crate) fn from_u32(value: u32) -> Self {
        Self::from_u32_slice(&[value])
    }

    /// Return the `Report` as a u32 slice.
    pub fn as_u32_slice(&self) -> &[u32] {
        &self.data
    }
}

/// Creates a new `Report` with all bits set to 0.
impl<const U32_SIZE: usize> Default for Report<U32_SIZE> {
    fn default() -> Self {
        Self::new()
    }
}

/// Compares the `Report` to a u32 slice.
impl<const U32_SIZE: usize> PartialEq<&[u32]> for Report<U32_SIZE> {
    fn eq(&self, other: &&[u32]) -> bool {
        let lengths_ok = other.len() <= U32_SIZE;
        let data_ok = self.data[..other.len()]
            .iter()
            .zip(other.iter())
            .all(|(x, y)| *x == *y);
        let trailing_zeros_ok = self.data[other.len()..].iter().all(|x| *x == 0);
        lengths_ok && data_ok && trailing_zeros_ok
    }
}

/// ANDs the `Report` bits.
impl<const U32_SIZE: usize> std::ops::BitAndAssign for Report<U32_SIZE> {
    fn bitand_assign(&mut self, rhs: Self) {
        for i in 0..U32_SIZE {
            self.data[i] &= rhs.data[i];
        }
    }
}

/// ANDs the `Report` bits.
impl<const U32_SIZE: usize> std::ops::BitAnd for Report<U32_SIZE> {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        let mut ret = self;
        ret &= rhs;
        ret
    }
}

/// ORs the `Report` bits.
impl<const U32_SIZE: usize> std::ops::BitOrAssign for Report<U32_SIZE> {
    fn bitor_assign(&mut self, rhs: Self) {
        for i in 0..U32_SIZE {
            self.data[i] |= rhs.data[i];
        }
    }
}

/// ORs the `Report` bits.
impl<const U32_SIZE: usize> std::ops::BitOr for Report<U32_SIZE> {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        let mut ret = self;
        ret |= rhs;
        ret
    }
}

/// XORs the `Report` bits.
impl<const U32_SIZE: usize> std::ops::BitXorAssign for Report<U32_SIZE> {
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..U32_SIZE {
            self.data[i] ^= rhs.data[i];
        }
    }
}

/// XORs the `Report` bits.
impl<const U32_SIZE: usize> std::ops::BitXor for Report<U32_SIZE> {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        let mut ret = self;
        ret ^= rhs;
        ret
    }
}

/// Inverts the `Report` bits.
impl<const U32_SIZE: usize> std::ops::Not for Report<U32_SIZE> {
    type Output = Self;

    fn not(self) -> Self::Output {
        let mut ret = self;
        for i in 0..U32_SIZE {
            ret.data[i] = !ret.data[i];
        }
        ret
    }
}

/// Shifts the `Report` bits to the left and assigns the result to the `Report`.
impl<const U32_SIZE: usize> std::ops::ShlAssign<usize> for Report<U32_SIZE> {
    fn shl_assign(&mut self, shift: usize) {
        let shift_u32 = shift / 32;
        let shift_bits = shift % 32;

        // Shift u32s first.
        for i in (shift_u32..U32_SIZE).rev() {
            self.data[i] = self.data[i - shift_u32];
        }
        for i in 0..shift_u32 {
            self.data[i] = 0;
        }

        // Shift bits within u32s.
        if shift_bits != 0 {
            let neg_shift_bits = 32usize - shift_bits;
            for i in (1..U32_SIZE).rev() {
                self.data[i] = (self.data[i] << shift_bits) | (self.data[i - 1] >> neg_shift_bits);
            }
            self.data[0] <<= shift_bits;
        }
    }
}

/// Shifts the `Report` bits to the left.
impl<const U32_SIZE: usize> std::ops::Shl<usize> for Report<U32_SIZE> {
    type Output = Self;

    fn shl(self, shift: usize) -> Self::Output {
        let mut ret = self;
        ret <<= shift;
        ret
    }
}

/// Shift the `Report` bits to the right and assigns the result to the `Report`.
impl<const U32_SIZE: usize> std::ops::ShrAssign<usize> for Report<U32_SIZE> {
    fn shr_assign(&mut self, shift: usize) {
        let shift_u32 = shift / 32;
        let shift_bits = shift % 32;

        // Shift u32s first.
        for i in 0..(U32_SIZE - shift_u32) {
            self.data[i] = self.data[i + shift_u32];
        }
        for i in (U32_SIZE - shift_u32)..U32_SIZE {
            self.data[i] = 0;
        }

        // Shift bits within u32s.
        if shift_bits != 0 {
            let neg_shift_bits = 32usize - shift_bits;
            for i in 0..(U32_SIZE - 1) {
                self.data[i] = (self.data[i] >> shift_bits) | (self.data[i + 1] << neg_shift_bits);
            }
            self.data[U32_SIZE - 1] >>= shift_bits;
        }
    }
}

/// Shifts the `Report` bits to the right.
impl<const U32_SIZE: usize> std::ops::Shr<usize> for Report<U32_SIZE> {
    type Output = Self;

    fn shr(self, shift: usize) -> Self::Output {
        let mut ret = self;
        ret >>= shift;
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::Report;
    use bincode;
    use rand::rngs::OsRng;

    #[test]
    fn test_new() {
        let report = Report::<1>::new();
        assert_eq!(report, [0].as_slice());

        let report = Report::<2>::new();
        assert_eq!(report, [0, 0].as_slice());
    }

    #[test]
    fn test_from_u32_slice() {
        let report = Report::<1>::from_u32_slice(&[0x12345678]);
        assert_eq!(report, [0x12345678].as_slice());

        let report = Report::<2>::from_u32_slice(&[0x12345678, 0x87654321]);
        assert_eq!(report.data, [0x12345678, 0x87654321].as_slice());

        let report = Report::<3>::from_u32_slice(&[0x12345678, 0x87654321, 0x12345678]);
        assert_eq!(report.data, [0x12345678, 0x87654321, 0x12345678].as_slice());
    }

    #[test]
    #[should_panic(
        expected = "Report::from_u32_slice: slice length is greater than the report size"
    )]
    fn test_from_u32_slice_panic() {
        let _report = Report::<1>::from_u32_slice(&[0x12345678, 0x87654321]);
    }

    #[test]
    fn test_from_u32() {
        let report = Report::<1>::from_u32(0x12345678);
        assert_eq!(report, [0x12345678].as_slice());
    }

    #[test]
    fn test_shift_left() {
        let report = Report::<1>::from_u32(0x12345678);
        assert_eq!(report << 0, [0x12345678].as_slice());
        assert_eq!(report << 1, [0x2468ACF0].as_slice());
        assert_eq!(report << 32, [0].as_slice());
        assert_eq!(report << 4, [0x23456780].as_slice());

        let report = Report::<2>::from_u32_slice([0x12345678, 0x87654321].as_slice());
        assert_eq!(report << 0, [0x12345678, 0x87654321].as_slice());
        assert_eq!(report << 1, [0x2468ACF0, 0xECA8642].as_slice());
        assert_eq!(report << 32, [0, 0x12345678].as_slice());
        assert_eq!(report << 4, [0x23456780, 0x76543211].as_slice());
    }

    #[test]
    fn test_shift_right() {
        let report = Report::<1>::from_u32(0x12345678);
        assert_eq!(report >> 0, [0x12345678].as_slice());
        assert_eq!(report >> 1, [0x91A2B3C].as_slice());
        assert_eq!(report >> 32, [0].as_slice());
        assert_eq!(report >> 4, [0x01234567].as_slice());

        let report = Report::<2>::from_u32_slice([0x12345678, 0x87654321].as_slice());
        assert_eq!(report >> 0, [0x12345678, 0x87654321].as_slice());
        assert_eq!(report >> 1, [0x891A2B3C, 0x43B2A190].as_slice());
        assert_eq!(report >> 32, [0x87654321, 0].as_slice());
        assert_eq!(report >> 4, [0x11234567, 0x8765432].as_slice());
    }

    #[test]
    fn test_bitand() {
        let report1 = Report::<1>::from_u32(0x12345678);
        let report2 = Report::<1>::from_u32(0x87654321);
        let report3 = Report::<1>::from_u32(0x2244220);
        assert_eq!(report3, report1 & report2);

        let report1 = Report::<2>::from_u32_slice([0x12345678, 0x87654321].as_slice());
        let report2 = Report::<2>::from_u32_slice([0x87654321, 0x12345678].as_slice());
        let report3 = Report::<2>::from_u32_slice([0x2244220, 0x2244220].as_slice());
        assert_eq!(report3, report1 & report2);

        let report3 = Report::<2>::from_u32_slice([0x87644321, 0x87654321].as_slice());
        assert_ne!(report3, report1 & report2);

        let report3 = Report::<2>::from_u32_slice([0x87654321, 0x87644321].as_slice());
        assert_ne!(report3, report1 & report2);
    }

    #[test]
    fn test_bitor() {
        let report1 = Report::<1>::from_u32(0x12345678);
        let report2 = Report::<1>::from_u32(0x87654321);
        let report3 = Report::<1>::from_u32(0x97755779);
        assert_eq!(report3, report1 | report2);

        let report1 = Report::<2>::from_u32_slice([0x12345678, 0x87654321].as_slice());
        let report2 = Report::<2>::from_u32_slice([0x87654321, 0x12345678].as_slice());
        let report3 = Report::<2>::from_u32_slice([0x97755779, 0x97755779].as_slice());
        assert_eq!(report3, report1 | report2);

        let report3 = Report::<2>::from_u32_slice([0x97745779, 0x97755779].as_slice());
        assert_ne!(report3, report1 | report2);

        let report3 = Report::<2>::from_u32_slice([0x97755779, 0x97745779].as_slice());
        assert_ne!(report3, report1 | report2);
    }

    #[test]
    fn test_bitxor() {
        let report1 = Report::<1>::from_u32(0x12345678);
        let report2 = Report::<1>::from_u32(0x87654321);
        let report3 = Report::<1>::from_u32(0x95511559);
        assert_eq!(report3, report1 ^ report2);

        let report1 = Report::<2>::from_u32_slice([0x12345678, 0x87654321].as_slice());
        let report2 = Report::<2>::from_u32_slice([0x87654321, 0x12345678].as_slice());
        let report3 = Report::<2>::from_u32_slice([0x95511559, 0x95511559].as_slice());
        assert_eq!(report3, report1 ^ report2);

        let report3 = Report::<2>::from_u32_slice([0x95510559, 0x95511559].as_slice());
        assert_ne!(report3, report1 ^ report2);

        let report3 = Report::<2>::from_u32_slice([0x95511559, 0x95510559].as_slice());
        assert_ne!(report3, report1 ^ report2);
    }

    #[test]
    fn test_not() {
        let report1 = Report::<1>::from_u32(0x12345678);
        let report2 = Report::<1>::from_u32(0xEDCBA987);
        assert_eq!(report2, !report1);

        let report1 = Report::<2>::from_u32_slice([0x12345678, 0x87654321].as_slice());
        let report2 = Report::<2>::from_u32_slice([0xEDCBA987, 0x789ABCDE].as_slice());
        assert_eq!(report2, !report1);
    }

    #[test]
    fn test_serialize_deserialize() {
        let mut rng = OsRng;

        let report = Report::<1>::random(&mut rng);
        let encoded = bincode::serialize(&report).unwrap();
        let report_copy: Report<1> = bincode::deserialize(&encoded).unwrap();
        assert_eq!(report, report_copy);

        let report = Report::<10>::random(&mut rng);
        let encoded = bincode::serialize(&report).unwrap();
        let report_copy: Report<10> = bincode::deserialize(&encoded).unwrap();
        assert_eq!(report, report_copy);
    }
}
