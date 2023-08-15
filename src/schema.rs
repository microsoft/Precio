// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use crate::report::attr::AttrValueType;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub(crate) enum AttributeType {
    C2,
    C3,
    C4,
    C5,
    C6,
    C7,
    C8,
    C9,
    C10,
    C11,
    C12,
    C13,
    C14,
    C15,
    C16,
    C17,
    C18,
    C19,
    C20,
    C21,
    C22,
    C23,
    C24,
    C25,
    C26,
    C27,
    C28,
    C29,
    C30,
    C31,
    C32,
    N2(u8),
    N3(u8),
    N4(u8),
    N5(u8),
    N6(u8),
    N7(u8),
    N8(u8),
    N9(u16),
    N10(u16),
    N11(u16),
    N12(u16),
    N13(u16),
    N14(u16),
    N15(u16),
    N16(u16),
    N17(u32),
    N18(u32),
    N19(u32),
    N20(u32),
    N21(u32),
    N22(u32),
    N23(u32),
    N24(u32),
    N25(u32),
    N26(u32),
    N27(u32),
    N28(u32),
    N29(u32),
    N30(u32),
    N31(u32),
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Attribute {
    C2(u8),
    C3(u8),
    C4(u8),
    C5(u8),
    C6(u8),
    C7(u8),
    C8(u8),
    C9(u16),
    C10(u16),
    C11(u16),
    C12(u16),
    C13(u16),
    C14(u16),
    C15(u16),
    C16(u16),
    C17(u32),
    C18(u32),
    C19(u32),
    C20(u32),
    C21(u32),
    C22(u32),
    C23(u32),
    C24(u32),
    C25(u32),
    C26(u32),
    C27(u32),
    C28(u32),
    C29(u32),
    C30(u32),
    C31(u32),
    C32(u32),
    N2(u8, u8),
    N3(u8, u8),
    N4(u8, u8),
    N5(u8, u8),
    N6(u8, u8),
    N7(u8, u8),
    N8(u8, u8),
    N9(u16, u16),
    N10(u16, u16),
    N11(u16, u16),
    N12(u16, u16),
    N13(u16, u16),
    N14(u16, u16),
    N15(u16, u16),
    N16(u16, u16),
    N17(u32, u32),
    N18(u32, u32),
    N19(u32, u32),
    N20(u32, u32),
    N21(u32, u32),
    N22(u32, u32),
    N23(u32, u32),
    N24(u32, u32),
    N25(u32, u32),
    N26(u32, u32),
    N27(u32, u32),
    N28(u32, u32),
    N29(u32, u32),
    N30(u32, u32),
    N31(u32, u32),
}

impl AttributeType {
    /// Returns the bit-size of the attribute.
    pub(crate) fn get_size(&self) -> usize {
        use AttributeType::*;

        match self {
            C2 => 2,
            C3 => 3,
            C4 => 4,
            C5 => 5,
            C6 => 6,
            C7 => 7,
            C8 => 8,
            C9 => 9,
            C10 => 10,
            C11 => 11,
            C12 => 12,
            C13 => 13,
            C14 => 14,
            C15 => 15,
            C16 => 16,
            C17 => 17,
            C18 => 18,
            C19 => 19,
            C20 => 20,
            C21 => 21,
            C22 => 22,
            C23 => 23,
            C24 => 24,
            C25 => 25,
            C26 => 26,
            C27 => 27,
            C28 => 28,
            C29 => 29,
            C30 => 30,
            C31 => 31,
            C32 => 32,
            N2(_) => 2,
            N3(_) => 3,
            N4(_) => 4,
            N5(_) => 5,
            N6(_) => 6,
            N7(_) => 7,
            N8(_) => 8,
            N9(_) => 9,
            N10(_) => 10,
            N11(_) => 11,
            N12(_) => 12,
            N13(_) => 13,
            N14(_) => 14,
            N15(_) => 15,
            N16(_) => 16,
            N17(_) => 17,
            N18(_) => 18,
            N19(_) => 19,
            N20(_) => 20,
            N21(_) => 21,
            N22(_) => 22,
            N23(_) => 23,
            N24(_) => 24,
            N25(_) => 25,
            N26(_) => 26,
            N27(_) => 27,
            N28(_) => 28,
            N29(_) => 29,
            N30(_) => 30,
            N31(_) => 31,
        }
    }

    /// Return whether the attribute is numerical.
    pub(crate) fn is_numerical(&self) -> bool {
        use AttributeType::*;

        match self {
            N2(_) => true,
            N3(_) => true,
            N4(_) => true,
            N5(_) => true,
            N6(_) => true,
            N7(_) => true,
            N8(_) => true,
            N9(_) => true,
            N10(_) => true,
            N11(_) => true,
            N12(_) => true,
            N13(_) => true,
            N14(_) => true,
            N15(_) => true,
            N16(_) => true,
            N17(_) => true,
            N18(_) => true,
            N19(_) => true,
            N20(_) => true,
            N21(_) => true,
            N22(_) => true,
            N23(_) => true,
            N24(_) => true,
            N25(_) => true,
            N26(_) => true,
            N27(_) => true,
            N28(_) => true,
            N29(_) => true,
            N30(_) => true,
            N31(_) => true,
            _ => false,
        }
    }

    /// Return whether the attribute is categorical.
    pub(crate) fn is_categorical(&self) -> bool {
        !self.is_numerical()
    }

    /// Get the bit mask for this attribute.
    fn get_bit_mask(&self) -> u32 {
        use AttributeType::*;

        match self {
            C2 | N2(_) => 0b0000_0000_0000_0000_0000_0000_0000_0011u32,
            C3 | N3(_) => 0b0000_0000_0000_0000_0000_0000_0000_0111u32,
            C4 | N4(_) => 0b0000_0000_0000_0000_0000_0000_0000_1111u32,
            C5 | N5(_) => 0b0000_0000_0000_0000_0000_0000_0001_1111u32,
            C6 | N6(_) => 0b0000_0000_0000_0000_0000_0000_0011_1111u32,
            C7 | N7(_) => 0b0000_0000_0000_0000_0000_0000_0111_1111u32,
            C8 | N8(_) => 0b0000_0000_0000_0000_0000_0000_1111_1111u32,
            C9 | N9(_) => 0b0000_0000_0000_0000_0000_0001_1111_1111u32,
            C10 | N10(_) => 0b0000_0000_0000_0000_0000_0011_1111_1111u32,
            C11 | N11(_) => 0b0000_0000_0000_0000_0000_0111_1111_1111u32,
            C12 | N12(_) => 0b0000_0000_0000_0000_0000_1111_1111_1111u32,
            C13 | N13(_) => 0b0000_0000_0000_0000_0001_1111_1111_1111u32,
            C14 | N14(_) => 0b0000_0000_0000_0000_0011_1111_1111_1111u32,
            C15 | N15(_) => 0b0000_0000_0000_0000_0111_1111_1111_1111u32,
            C16 | N16(_) => 0b0000_0000_0000_0000_1111_1111_1111_1111u32,
            C17 | N17(_) => 0b0000_0000_0000_0001_1111_1111_1111_1111u32,
            C18 | N18(_) => 0b0000_0000_0000_0011_1111_1111_1111_1111u32,
            C19 | N19(_) => 0b0000_0000_0000_0111_1111_1111_1111_1111u32,
            C20 | N20(_) => 0b0000_0000_0000_1111_1111_1111_1111_1111u32,
            C21 | N21(_) => 0b0000_0000_0001_1111_1111_1111_1111_1111u32,
            C22 | N22(_) => 0b0000_0000_0011_1111_1111_1111_1111_1111u32,
            C23 | N23(_) => 0b0000_0000_0111_1111_1111_1111_1111_1111u32,
            C24 | N24(_) => 0b0000_0000_1111_1111_1111_1111_1111_1111u32,
            C25 | N25(_) => 0b0000_0001_1111_1111_1111_1111_1111_1111u32,
            C26 | N26(_) => 0b0000_0011_1111_1111_1111_1111_1111_1111u32,
            C27 | N27(_) => 0b0000_0111_1111_1111_1111_1111_1111_1111u32,
            C28 | N28(_) => 0b0000_1111_1111_1111_1111_1111_1111_1111u32,
            C29 | N29(_) => 0b0001_1111_1111_1111_1111_1111_1111_1111u32,
            C30 | N30(_) => 0b0011_1111_1111_1111_1111_1111_1111_1111u32,
            C31 | N31(_) => 0b0111_1111_1111_1111_1111_1111_1111_1111u32,
            C32 => 0b1111_1111_1111_1111_1111_1111_1111_1111u32,
        }
    }

    /// Check if the `AttributeType` is valid. All categorical attribute types
    /// are automatically valid, but numerical attribute types may have a modulus
    /// that is larger than the intended bit size.
    pub(crate) fn is_valid(&self) -> bool {
        let inv_bit_mask = !self.get_bit_mask();

        use AttributeType::*;

        match self {
            N2(m) => (*m as u32) & inv_bit_mask == 0,
            N3(m) => (*m as u32) & inv_bit_mask == 0,
            N4(m) => (*m as u32) & inv_bit_mask == 0,
            N5(m) => (*m as u32) & inv_bit_mask == 0,
            N6(m) => (*m as u32) & inv_bit_mask == 0,
            N7(m) => (*m as u32) & inv_bit_mask == 0,
            N8(m) => (*m as u32) & inv_bit_mask == 0,
            N9(m) => (*m as u32) & inv_bit_mask == 0,
            N10(m) => (*m as u32) & inv_bit_mask == 0,
            N11(m) => (*m as u32) & inv_bit_mask == 0,
            N12(m) => (*m as u32) & inv_bit_mask == 0,
            N13(m) => (*m as u32) & inv_bit_mask == 0,
            N14(m) => (*m as u32) & inv_bit_mask == 0,
            N15(m) => (*m as u32) & inv_bit_mask == 0,
            N16(m) => (*m as u32) & inv_bit_mask == 0,
            N17(m) => (*m as u32) & inv_bit_mask == 0,
            N18(m) => (*m as u32) & inv_bit_mask == 0,
            N19(m) => (*m as u32) & inv_bit_mask == 0,
            N20(m) => (*m as u32) & inv_bit_mask == 0,
            N21(m) => (*m as u32) & inv_bit_mask == 0,
            N22(m) => (*m as u32) & inv_bit_mask == 0,
            N23(m) => (*m as u32) & inv_bit_mask == 0,
            N24(m) => (*m as u32) & inv_bit_mask == 0,
            N25(m) => (*m as u32) & inv_bit_mask == 0,
            N26(m) => (*m as u32) & inv_bit_mask == 0,
            N27(m) => (*m as u32) & inv_bit_mask == 0,
            N28(m) => (*m as u32) & inv_bit_mask == 0,
            N29(m) => (*m as u32) & inv_bit_mask == 0,
            N30(m) => (*m as u32) & inv_bit_mask == 0,
            N31(m) => (*m as u32) & inv_bit_mask == 0,
            _ => true,
        }
    }

    /// Return whether a given value is valid for this `AttributeType`.
    pub(crate) fn is_valid_value(&self, attr_value: u32) -> bool {
        let bit_mask = self.get_bit_mask();
        if self.is_numerical() {
            (attr_value & !bit_mask == 0) && self.make_attr(attr_value).is_valid()
        } else {
            attr_value & !bit_mask == 0
        }
    }

    /// Return the modulus for numerical attributes. For categorical
    /// attributes this function returns 0.
    pub(crate) fn get_modulus(&self) -> u32 {
        let bit_mask = self.get_bit_mask();

        use AttributeType::*;

        match self {
            N2(m) => (*m as u32) & bit_mask,
            N3(m) => (*m as u32) & bit_mask,
            N4(m) => (*m as u32) & bit_mask,
            N5(m) => (*m as u32) & bit_mask,
            N6(m) => (*m as u32) & bit_mask,
            N7(m) => (*m as u32) & bit_mask,
            N8(m) => (*m as u32) & bit_mask,
            N9(m) => (*m as u32) & bit_mask,
            N10(m) => (*m as u32) & bit_mask,
            N11(m) => (*m as u32) & bit_mask,
            N12(m) => (*m as u32) & bit_mask,
            N13(m) => (*m as u32) & bit_mask,
            N14(m) => (*m as u32) & bit_mask,
            N15(m) => (*m as u32) & bit_mask,
            N16(m) => (*m as u32) & bit_mask,
            N17(m) => (*m as u32) & bit_mask,
            N18(m) => (*m as u32) & bit_mask,
            N19(m) => (*m as u32) & bit_mask,
            N20(m) => (*m as u32) & bit_mask,
            N21(m) => (*m as u32) & bit_mask,
            N22(m) => (*m as u32) & bit_mask,
            N23(m) => (*m as u32) & bit_mask,
            N24(m) => (*m as u32) & bit_mask,
            N25(m) => (*m as u32) & bit_mask,
            N26(m) => (*m as u32) & bit_mask,
            N27(m) => (*m as u32) & bit_mask,
            N28(m) => (*m as u32) & bit_mask,
            N29(m) => (*m as u32) & bit_mask,
            N30(m) => (*m as u32) & bit_mask,
            N31(m) => (*m as u32) & bit_mask,
            _ => 0u32,
        }
    }

    /// Create a new `Attribute` with given value. The created `Attribute`
    /// may not be valid, and its validity should be checked with the
    /// `Attribute::is_valid` function. Alternatively, the value can be
    /// checked with `AttributeType::is_valid_value` before calling this
    /// function.
    fn make_attr(&self, attr_value: u32) -> Attribute {
        let modulus = self.get_modulus();

        use AttributeType::*;

        match self {
            C2 => Attribute::C2(attr_value as u8),
            C3 => Attribute::C3(attr_value as u8),
            C4 => Attribute::C4(attr_value as u8),
            C5 => Attribute::C5(attr_value as u8),
            C6 => Attribute::C6(attr_value as u8),
            C7 => Attribute::C7(attr_value as u8),
            C8 => Attribute::C8(attr_value as u8),
            C9 => Attribute::C9(attr_value as u16),
            C10 => Attribute::C10(attr_value as u16),
            C11 => Attribute::C11(attr_value as u16),
            C12 => Attribute::C12(attr_value as u16),
            C13 => Attribute::C13(attr_value as u16),
            C14 => Attribute::C14(attr_value as u16),
            C15 => Attribute::C15(attr_value as u16),
            C16 => Attribute::C16(attr_value as u16),
            C17 => Attribute::C17(attr_value),
            C18 => Attribute::C18(attr_value),
            C19 => Attribute::C19(attr_value),
            C20 => Attribute::C20(attr_value),
            C21 => Attribute::C21(attr_value),
            C22 => Attribute::C22(attr_value),
            C23 => Attribute::C23(attr_value),
            C24 => Attribute::C24(attr_value),
            C25 => Attribute::C25(attr_value),
            C26 => Attribute::C26(attr_value),
            C27 => Attribute::C27(attr_value),
            C28 => Attribute::C28(attr_value),
            C29 => Attribute::C29(attr_value),
            C30 => Attribute::C30(attr_value),
            C31 => Attribute::C31(attr_value),
            C32 => Attribute::C32(attr_value),
            N2(_) => Attribute::N2(attr_value as u8, modulus as u8),
            N3(_) => Attribute::N3(attr_value as u8, modulus as u8),
            N4(_) => Attribute::N4(attr_value as u8, modulus as u8),
            N5(_) => Attribute::N5(attr_value as u8, modulus as u8),
            N6(_) => Attribute::N6(attr_value as u8, modulus as u8),
            N7(_) => Attribute::N7(attr_value as u8, modulus as u8),
            N8(_) => Attribute::N8(attr_value as u8, modulus as u8),
            N9(_) => Attribute::N9(attr_value as u16, modulus as u16),
            N10(_) => Attribute::N10(attr_value as u16, modulus as u16),
            N11(_) => Attribute::N11(attr_value as u16, modulus as u16),
            N12(_) => Attribute::N12(attr_value as u16, modulus as u16),
            N13(_) => Attribute::N13(attr_value as u16, modulus as u16),
            N14(_) => Attribute::N14(attr_value as u16, modulus as u16),
            N15(_) => Attribute::N15(attr_value as u16, modulus as u16),
            N16(_) => Attribute::N16(attr_value as u16, modulus as u16),
            N17(_) => Attribute::N17(attr_value, modulus),
            N18(_) => Attribute::N18(attr_value, modulus),
            N19(_) => Attribute::N19(attr_value, modulus),
            N20(_) => Attribute::N20(attr_value, modulus),
            N21(_) => Attribute::N21(attr_value, modulus),
            N22(_) => Attribute::N22(attr_value, modulus),
            N23(_) => Attribute::N23(attr_value, modulus),
            N24(_) => Attribute::N24(attr_value, modulus),
            N25(_) => Attribute::N25(attr_value, modulus),
            N26(_) => Attribute::N26(attr_value, modulus),
            N27(_) => Attribute::N27(attr_value, modulus),
            N28(_) => Attribute::N28(attr_value, modulus),
            N29(_) => Attribute::N29(attr_value, modulus),
            N30(_) => Attribute::N30(attr_value, modulus),
            N31(_) => Attribute::N31(attr_value, modulus),
        }
    }
}

impl Attribute {
    /// Return the `AttributeType`.
    pub(crate) fn get_type(&self) -> AttributeType {
        use Attribute::*;

        match self {
            C2(_) => AttributeType::C2,
            C3(_) => AttributeType::C3,
            C4(_) => AttributeType::C4,
            C5(_) => AttributeType::C5,
            C6(_) => AttributeType::C6,
            C7(_) => AttributeType::C7,
            C8(_) => AttributeType::C8,
            C9(_) => AttributeType::C9,
            C10(_) => AttributeType::C10,
            C11(_) => AttributeType::C11,
            C12(_) => AttributeType::C12,
            C13(_) => AttributeType::C13,
            C14(_) => AttributeType::C14,
            C15(_) => AttributeType::C15,
            C16(_) => AttributeType::C16,
            C17(_) => AttributeType::C17,
            C18(_) => AttributeType::C18,
            C19(_) => AttributeType::C19,
            C20(_) => AttributeType::C20,
            C21(_) => AttributeType::C21,
            C22(_) => AttributeType::C22,
            C23(_) => AttributeType::C23,
            C24(_) => AttributeType::C24,
            C25(_) => AttributeType::C25,
            C26(_) => AttributeType::C26,
            C27(_) => AttributeType::C27,
            C28(_) => AttributeType::C28,
            C29(_) => AttributeType::C29,
            C30(_) => AttributeType::C30,
            C31(_) => AttributeType::C31,
            C32(_) => AttributeType::C32,
            N2(_, m) => AttributeType::N2(*m as u8),
            N3(_, m) => AttributeType::N3(*m as u8),
            N4(_, m) => AttributeType::N4(*m as u8),
            N5(_, m) => AttributeType::N5(*m as u8),
            N6(_, m) => AttributeType::N6(*m as u8),
            N7(_, m) => AttributeType::N7(*m as u8),
            N8(_, m) => AttributeType::N8(*m as u8),
            N9(_, m) => AttributeType::N9(*m as u16),
            N10(_, m) => AttributeType::N10(*m as u16),
            N11(_, m) => AttributeType::N11(*m as u16),
            N12(_, m) => AttributeType::N12(*m as u16),
            N13(_, m) => AttributeType::N13(*m as u16),
            N14(_, m) => AttributeType::N14(*m as u16),
            N15(_, m) => AttributeType::N15(*m as u16),
            N16(_, m) => AttributeType::N16(*m as u16),
            N17(_, m) => AttributeType::N17(*m as u32),
            N18(_, m) => AttributeType::N18(*m as u32),
            N19(_, m) => AttributeType::N19(*m as u32),
            N20(_, m) => AttributeType::N20(*m as u32),
            N21(_, m) => AttributeType::N21(*m as u32),
            N22(_, m) => AttributeType::N22(*m as u32),
            N23(_, m) => AttributeType::N23(*m as u32),
            N24(_, m) => AttributeType::N24(*m as u32),
            N25(_, m) => AttributeType::N25(*m as u32),
            N26(_, m) => AttributeType::N26(*m as u32),
            N27(_, m) => AttributeType::N27(*m as u32),
            N28(_, m) => AttributeType::N28(*m as u32),
            N29(_, m) => AttributeType::N29(*m as u32),
            N30(_, m) => AttributeType::N30(*m as u32),
            N31(_, m) => AttributeType::N31(*m as u32),
        }
    }

    /// Check if the `Attribute` value is valid.
    fn is_valid(&self) -> bool {
        let this_type = self.get_type();
        let inv_bit_mask = !this_type.get_bit_mask();

        use Attribute::*;

        match self {
            C2(v) => (*v as u32) & inv_bit_mask == 0,
            C3(v) => (*v as u32) & inv_bit_mask == 0,
            C4(v) => (*v as u32) & inv_bit_mask == 0,
            C5(v) => (*v as u32) & inv_bit_mask == 0,
            C6(v) => (*v as u32) & inv_bit_mask == 0,
            C7(v) => (*v as u32) & inv_bit_mask == 0,
            C8(v) => (*v as u32) & inv_bit_mask == 0,
            C9(v) => (*v as u32) & inv_bit_mask == 0,
            C10(v) => (*v as u32) & inv_bit_mask == 0,
            C11(v) => (*v as u32) & inv_bit_mask == 0,
            C12(v) => (*v as u32) & inv_bit_mask == 0,
            C13(v) => (*v as u32) & inv_bit_mask == 0,
            C14(v) => (*v as u32) & inv_bit_mask == 0,
            C15(v) => (*v as u32) & inv_bit_mask == 0,
            C16(v) => (*v as u32) & inv_bit_mask == 0,
            C17(v) => (*v as u32) & inv_bit_mask == 0,
            C18(v) => (*v as u32) & inv_bit_mask == 0,
            C19(v) => (*v as u32) & inv_bit_mask == 0,
            C20(v) => (*v as u32) & inv_bit_mask == 0,
            C21(v) => (*v as u32) & inv_bit_mask == 0,
            C22(v) => (*v as u32) & inv_bit_mask == 0,
            C23(v) => (*v as u32) & inv_bit_mask == 0,
            C24(v) => (*v as u32) & inv_bit_mask == 0,
            C25(v) => (*v as u32) & inv_bit_mask == 0,
            C26(v) => (*v as u32) & inv_bit_mask == 0,
            C27(v) => (*v as u32) & inv_bit_mask == 0,
            C28(v) => (*v as u32) & inv_bit_mask == 0,
            C29(v) => (*v as u32) & inv_bit_mask == 0,
            C30(v) => (*v as u32) & inv_bit_mask == 0,
            C31(v) => (*v as u32) & inv_bit_mask == 0,
            C32(v) => (*v as u32) & inv_bit_mask == 0,
            N2(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N3(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N4(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N5(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N6(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N7(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N8(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N9(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N10(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N11(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N12(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N13(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N14(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N15(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N16(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N17(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N18(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N19(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N20(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N21(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N22(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N23(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N24(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N25(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N26(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N27(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N28(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N29(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N30(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
            N31(v, m) => ((*m as u32) & inv_bit_mask == 0) && *v < *m,
        }
    }

    /// Return the `Attribute` value as a `u32`.
    pub(crate) fn get_value(&self) -> u32 {
        let bit_mask = self.get_type().get_bit_mask();

        use Attribute::*;

        match self {
            C2(v) | N2(v, _) => (*v as u32) & bit_mask,
            C3(v) | N3(v, _) => (*v as u32) & bit_mask,
            C4(v) | N4(v, _) => (*v as u32) & bit_mask,
            C5(v) | N5(v, _) => (*v as u32) & bit_mask,
            C6(v) | N6(v, _) => (*v as u32) & bit_mask,
            C7(v) | N7(v, _) => (*v as u32) & bit_mask,
            C8(v) | N8(v, _) => (*v as u32) & bit_mask,
            C9(v) | N9(v, _) => (*v as u32) & bit_mask,
            C10(v) | N10(v, _) => (*v as u32) & bit_mask,
            C11(v) | N11(v, _) => (*v as u32) & bit_mask,
            C12(v) | N12(v, _) => (*v as u32) & bit_mask,
            C13(v) | N13(v, _) => (*v as u32) & bit_mask,
            C14(v) | N14(v, _) => (*v as u32) & bit_mask,
            C15(v) | N15(v, _) => (*v as u32) & bit_mask,
            C16(v) | N16(v, _) => (*v as u32) & bit_mask,
            C17(v) | N17(v, _) => (*v as u32) & bit_mask,
            C18(v) | N18(v, _) => (*v as u32) & bit_mask,
            C19(v) | N19(v, _) => (*v as u32) & bit_mask,
            C20(v) | N20(v, _) => (*v as u32) & bit_mask,
            C21(v) | N21(v, _) => (*v as u32) & bit_mask,
            C22(v) | N22(v, _) => (*v as u32) & bit_mask,
            C23(v) | N23(v, _) => (*v as u32) & bit_mask,
            C24(v) | N24(v, _) => (*v as u32) & bit_mask,
            C25(v) | N25(v, _) => (*v as u32) & bit_mask,
            C26(v) | N26(v, _) => (*v as u32) & bit_mask,
            C27(v) | N27(v, _) => (*v as u32) & bit_mask,
            C28(v) | N28(v, _) => (*v as u32) & bit_mask,
            C29(v) | N29(v, _) => (*v as u32) & bit_mask,
            C30(v) | N30(v, _) => (*v as u32) & bit_mask,
            C31(v) | N31(v, _) => (*v as u32) & bit_mask,
            C32(v) => (*v as u32) & bit_mask,
        }
    }
}

/// Convert an `AttrValueType` to an `Attribute` if it has a valid value.
pub(crate) fn attr_from_attr_value(
    attr_type: AttributeType,
    attr_value: AttrValueType,
) -> Result<Attribute, String> {
    // Check that `attr_value` is a valid attribute value. Otherwise return an error.
    let attr = attr_type.make_attr(attr_value);
    if attr.is_valid() {
        Ok(attr)
    } else {
        Err(format!(
            "Given value {} is invalid for the attribute type {:?}.",
            attr_value, attr_type
        ))
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub struct Schema(pub(crate) Vec<(String, AttributeType)>);

impl Schema {
    /// Return the number of attributes in this `Schema`.
    pub(crate) fn len(&self) -> usize {
        if !self.is_valid() {
            panic!("Invalid schema");
        }

        self.0.len()
    }

    /// Return a vector of attribute names.
    pub(crate) fn get_attr_names(&self) -> Vec<String> {
        if !self.is_valid() {
            panic!("Invalid schema");
        }

        let mut attr_names = Vec::new();
        for (attr_name, _) in self.0.iter() {
            attr_names.push(attr_name.clone());
        }
        attr_names
    }

    /// If the schema has an attribute with the given name, returns the index of
    /// that attribute. Otherwise returns `None`.
    pub(crate) fn get_attr_index(&self, attr_name: &str) -> Option<usize> {
        if !self.is_valid() {
            panic!("Invalid schema");
        }

        for (i, (name, _)) in self.0.iter().enumerate() {
            if name == attr_name {
                return Some(i);
            }
        }
        None
    }

    /// Returns a `Vec` of the `AttributeType`s.
    pub(crate) fn get_attr_types(&self) -> Vec<AttributeType> {
        if !self.is_valid() {
            panic!("Invalid schema");
        }

        let mut attr_types = Vec::new();
        for (_, attr_type) in self.0.iter() {
            attr_types.push(*attr_type);
        }
        attr_types
    }

    /// Returns a `HashMap` mapping the attribute names to their `AttributeType`s.
    pub(crate) fn get_attr_name_to_type_map(&self) -> HashMap<String, AttributeType> {
        if !self.is_valid() {
            panic!("Invalid schema");
        }

        let mut attr_types = HashMap::new();
        for (attr_name, attr_type) in self.0.iter() {
            attr_types.insert(attr_name.clone(), *attr_type);
        }
        attr_types
    }

    /// Return a vector of attribute sizes.
    pub(crate) fn get_attr_sizes(&self) -> Vec<usize> {
        if !self.is_valid() {
            panic!("Invalid schema");
        }
        self.0
            .iter()
            .map(|(_, attr_type)| attr_type.get_size())
            .collect()
    }

    /// Validate the `Schema`. The function returns `false` if the attribute names are not
    /// unique or if any of the attribute types are invalid. Otherwise returns `true`.
    pub(crate) fn is_valid(&self) -> bool {
        let mut attr_names = HashSet::<String>::new();
        self.0
            .iter()
            .all(|(name, attr_type)| attr_names.insert(name.clone()) && attr_type.is_valid())
    }

    /// Return whether a given array of `Attribute`s is compatible with the `Schema`.
    pub(crate) fn is_compatible_attr_array(&self, attributes: &[Attribute]) -> bool {
        if !self.is_valid() {
            panic!("Invalid schema");
        }

        if self.0.len() != attributes.len() {
            return false;
        }
        for (i, attr) in attributes.iter().enumerate() {
            if attr.get_type() != self.0[i].1 {
                return false;
            }
            if !attr.is_valid() {
                return false;
            }
        }
        true
    }

    /// Return whether a given array of `AttributeType`s is compatible with the `Schema`.
    pub(crate) fn is_compatible_attr_type_array(&self, attribute_types: &[AttributeType]) -> bool {
        if !self.is_valid() {
            panic!("Invalid schema");
        }

        if self.0.len() != attribute_types.len() {
            return false;
        }
        for (i, &attr_type) in attribute_types.iter().enumerate() {
            if attr_type != self.0[i].1 {
                return false;
            }
            if !attr_type.is_valid() {
                return false;
            }
        }
        true
    }

    /// Remove one of the `AttributeType`s from the `Schema`.
    pub(crate) fn remove_attr(&mut self, attr_name: &str) -> &mut Self {
        if !self.is_valid() {
            panic!("Invalid schema");
        }

        let attr_index = self.get_attr_index(attr_name);
        assert!(attr_index.is_some(), "Attribute not found");
        self.0.remove(attr_index.unwrap());

        self
    }
}

impl TryFrom<&str> for Schema {
    type Error = String;

    fn try_from(input_json: &str) -> Result<Self, Self::Error> {
        // Load the input JSON.
        let schema_json: Result<Schema, _> = serde_json::from_str(input_json);
        match schema_json {
            Ok(schema) => Ok(schema),
            Err(e) => Err(format!("Error parsing input JSON: {}.", e)),
        }
    }
}

/// Get the attribute type for a named attribute.
pub(crate) fn attr_type_from_attr_name(
    schema: &Schema,
    attr_name: &str,
) -> Result<AttributeType, String> {
    // Check that `attr_name` is a valid attribute name. Otherwise return an error.
    let attr_type = schema.get_attr_name_to_type_map().get(attr_name).copied();
    if attr_type.is_some() {
        Ok(attr_type.unwrap())
    } else {
        Err(format!("Attribute name {} is invalid.", attr_name))
    }
}

/// Get the attribute index for a named attribute.
pub(crate) fn attr_index_from_attr_name(schema: &Schema, attr_name: &str) -> Result<usize, String> {
    // Check that `attr_name` is a valid attribute name. Otherwise return an error.
    schema
        .get_attr_index(attr_name)
        .ok_or(format!("Attribute name {} is invalid.", attr_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_attr_sizes() {
        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
            ("attr3".to_string(), AttributeType::C4),
        ]);
        assert_eq!(schema.get_attr_sizes(), vec![2, 3, 4]);

        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C12),
            ("attr2".to_string(), AttributeType::C13),
            ("attr3".to_string(), AttributeType::C14),
        ]);
        assert_eq!(schema.get_attr_sizes(), vec![12, 13, 14]);

        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::N2(3)),
            ("attr2".to_string(), AttributeType::N3(4)),
            ("attr3".to_string(), AttributeType::N4(5)),
        ]);
        assert_eq!(schema.get_attr_sizes(), vec![2, 3, 4]);

        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::N3(5)),
            ("attr2".to_string(), AttributeType::C10),
            ("attr3".to_string(), AttributeType::N17(65537)),
        ]);
        assert_eq!(schema.get_attr_sizes(), vec![3, 10, 17]);
    }

    #[test]
    fn test_is_valid() {
        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
            ("attr3".to_string(), AttributeType::C4),
        ]);
        assert!(schema.is_valid());

        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
            ("attr3".to_string(), AttributeType::C4),
            ("attr2".to_string(), AttributeType::C5),
        ]);
        assert!(!schema.is_valid());

        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::N3(5)),
            ("attr3".to_string(), AttributeType::C4),
            ("attr4".to_string(), AttributeType::N5(20)),
        ]);
        assert!(schema.is_valid());

        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::N3(5)),
            ("attr3".to_string(), AttributeType::C4),
            ("attr4".to_string(), AttributeType::N5(33)),
        ]);
        assert!(!schema.is_valid());
    }

    #[test]
    fn test_is_compatible() {
        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
            ("attr3".to_string(), AttributeType::C4),
            ("attr4".to_string(), AttributeType::C8),
            ("attr5".to_string(), AttributeType::C15),
            ("attr6".to_string(), AttributeType::C16),
        ]);
        let attributes = vec![
            Attribute::C2(0b11),
            Attribute::C3(0b111),
            Attribute::C4(0b1111),
            Attribute::C8(0b1111_1111),
            Attribute::C15(0b0111_1111_1111_1111),
            Attribute::C16(0b1111_1111_1111_1111),
        ];
        assert!(schema.is_compatible_attr_array(&attributes));

        let attributes = vec![
            Attribute::C2(0b11),
            Attribute::C3(0b1111),
            Attribute::C4(0b1111),
            Attribute::C8(0b1111_1111),
            Attribute::C15(0b0111_1111_1111_1111),
            Attribute::C16(0b1111_1111_1111_1111),
        ];
        assert!(!schema.is_compatible_attr_array(&attributes));

        let attributes = vec![
            Attribute::C2(0b11),
            Attribute::C3(0b111),
            Attribute::C4(0b1111),
            Attribute::C8(0b1111_1111),
            Attribute::C15(0b1111_1111_1111_1111),
            Attribute::C16(0b1111_1111_1111_1111),
        ];
        assert!(!schema.is_compatible_attr_array(&attributes));

        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::N2(3)),
            ("attr2".to_string(), AttributeType::C3),
            ("attr3".to_string(), AttributeType::N4(10)),
            ("attr4".to_string(), AttributeType::N8(255)),
            ("attr5".to_string(), AttributeType::C15),
            ("attr6".to_string(), AttributeType::C16),
        ]);
        let attributes = vec![
            Attribute::N2(0b10, 0b11),
            Attribute::C3(0b111),
            Attribute::N4(0b11, 10),
            Attribute::N8(0b1111_1110, 255),
            Attribute::C15(0b0111_1111_1111_1111),
            Attribute::C16(0b1111_1111_1111_1111),
        ];
        assert!(schema.is_compatible_attr_array(&attributes));

        let attributes = vec![
            Attribute::N2(0b01, 0b10),
            Attribute::C3(0b111),
            Attribute::N4(0b11, 10),
            Attribute::N8(0b1111_1110, 255),
            Attribute::C15(0b0111_1111_1111_1111),
            Attribute::C16(0b1111_1111_1111_1111),
        ];
        assert!(!schema.is_compatible_attr_array(&attributes));

        let attributes = vec![
            Attribute::C2(0b11),
            Attribute::C3(0b111),
            Attribute::N4(0b11, 11),
            Attribute::N8(0b1111_1110, 255),
            Attribute::C15(0b0111_1111_1111_1111),
            Attribute::C16(0b1111_1111_1111_1111),
        ];
        assert!(!schema.is_compatible_attr_array(&attributes));

        let attributes = vec![
            Attribute::C2(0b11),
            Attribute::C3(0b111),
            Attribute::N4(0b11, 10),
            Attribute::N8(0b1111_1111, 255),
            Attribute::C15(0b0111_1111_1111_1111),
            Attribute::C16(0b1111_1111_1111_1111),
        ];
        assert!(!schema.is_compatible_attr_array(&attributes));
    }

    #[test]
    fn test_get_value() {
        let attributes = vec![
            Attribute::C2(0b1111_1111),
            Attribute::C3(0b1111_1111),
            Attribute::C4(0b1111_1111),
            Attribute::C8(0b1111_1111),
            Attribute::C15(0b1111_1111_1111_1111),
            Attribute::C16(0b1111_1111_1111_1111),
        ];
        assert_eq!(attributes[0].get_value(), 0b11);
        assert_eq!(attributes[1].get_value(), 0b111);
        assert_eq!(attributes[2].get_value(), 0b1111);
        assert_eq!(attributes[3].get_value(), 0b1111_1111);
        assert_eq!(attributes[4].get_value(), 0b0111_1111_1111_1111);
        assert_eq!(attributes[5].get_value(), 0b1111_1111_1111_1111);

        let attributes = vec![
            // Note that some of these are not actually valid attributes.
            Attribute::N3(0b1111_0101, 0b1111_1111),
            Attribute::N4(0b1111_0110, 0b1111_1111),
            Attribute::N8(0b1001_0110, 0b1111_1111),
            Attribute::N15(0b1110_1010_0101_1001, 0b1111_1111_1111_1111),
            Attribute::N16(0b1110_1010_0101_1001, 0b1111_1111_1111_1111),
        ];
        assert_eq!(attributes[0].get_value(), 0b101);
        assert_eq!(attributes[1].get_value(), 0b0110);
        assert_eq!(attributes[2].get_value(), 0b1001_0110);
        assert_eq!(attributes[3].get_value(), 0b0110_1010_0101_1001);
        assert_eq!(attributes[4].get_value(), 0b1110_1010_0101_1001);
        assert_eq!(attributes[0].get_type().get_modulus(), 0b111);
        assert_eq!(attributes[1].get_type().get_modulus(), 0b1111);
        assert_eq!(attributes[2].get_type().get_modulus(), 0b1111_1111);
        assert_eq!(
            attributes[3].get_type().get_modulus(),
            0b0111_1111_1111_1111
        );
        assert_eq!(
            attributes[4].get_type().get_modulus(),
            0b1111_1111_1111_1111
        );
        assert!(!attributes[0].is_valid());
        assert!(!attributes[1].is_valid());
        assert!(attributes[2].is_valid());
        assert!(!attributes[3].is_valid());
        assert!(attributes[4].is_valid());
    }

    #[test]
    fn test_serialize_deserialize_json() {
        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
            ("attr3".to_string(), AttributeType::C4),
            ("attr4".to_string(), AttributeType::C8),
            ("attr5".to_string(), AttributeType::C15),
            ("attr6".to_string(), AttributeType::C16),
        ]);
        let json = serde_json::to_string(&schema).unwrap();
        let schema2: Schema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, schema2);

        let json = r#"[["myattr1","c2"], ["myattr2","c4"], ["myattr3","c15"]]"#;
        let schema: Schema = serde_json::from_str(&json).unwrap();
        let schema2 = Schema(vec![
            ("myattr1".to_string(), AttributeType::C2),
            ("myattr2".to_string(), AttributeType::C4),
            ("myattr3".to_string(), AttributeType::C15),
        ]);
        assert_eq!(schema, schema2);

        let schema = Schema(vec![
            ("attr1".to_string(), AttributeType::C2),
            ("attr2".to_string(), AttributeType::C3),
            ("attr3".to_string(), AttributeType::N4(10)),
            ("attr4".to_string(), AttributeType::C8),
            ("attr5".to_string(), AttributeType::N15(34567)),
            ("attr6".to_string(), AttributeType::C16),
        ]);
        let json = serde_json::to_string(&schema).unwrap();
        let schema2: Schema = serde_json::from_str(&json).unwrap();
        assert_eq!(schema, schema2);

        let json =
            r#"[["myattr1","c2"], ["myattr2","c4"], ["myattr3",{"n4": 10}], ["myattr4","c15"]]"#;
        let schema: Schema = serde_json::from_str(&json).unwrap();
        let schema2 = Schema(vec![
            ("myattr1".to_string(), AttributeType::C2),
            ("myattr2".to_string(), AttributeType::C4),
            ("myattr3".to_string(), AttributeType::N4(10)),
            ("myattr4".to_string(), AttributeType::C15),
        ]);
        assert_eq!(schema, schema2);
    }
}
