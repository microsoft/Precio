// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use super::role::Role;
use crate::arith::Modulus;
use crate::random::prf::*;
use crate::random::RandomMod;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake256,
};

#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub enum OTMessage<M: Modulus> {
    Seed(bool, PRFValue),
    Key(PRFKey),
    MaskedBit(bool),
    Reveal(M, M),
}

/// In this 3-party OT protocol two parties (`Role::First`, `Role::Second`) hold bits a and b.
/// The third server (`Role::Third`) helps them compute the product of a and b, secret shared
/// in a ring Z_p. The protocol uses the messages defined above. A message of type `OTMessage::Key`
/// is created by `Role::Third` and sent to `Role::Second`, who uses it throughout the protocol.
/// Then, for each iteration of the protocol (each report), `Role::Third` sends a message of type
/// `OTMessage::Seed` to `Role::First`. After processing the message, `Role::First` sends
/// a message of type `OTMessage::MaskedBit` to `Role::Second`, who processes it to produce
/// its own output and a message of type `OTMessage::Reveal` to `Role::First`, which is finally
/// processed to produce the output for `Role::First`.
///
/// The `OTProtocolHandler` struct implements the protocol steps. It processes the inputs and
/// generates outputs for each step.
#[derive(Clone)]
pub(crate) struct OTProtocolHandler<M: Modulus + RandomMod> {
    /// Indicates the role of the party.
    role: Role,

    /// The modulus for the output.
    modulus: M,

    /// The PRF key. Only used by `Role::Third` and `Role::Second`.
    key: Option<PRFKey>,

    /// The PRF. Only used by `Role::Third` and `Role::Second`.
    prf: Option<PRF>,

    /// Internal values used by `Role::First`.
    c: Option<Vec<bool>>,

    /// Internal values used by `Role::First`.
    r_c: Option<Vec<M>>,
}

impl<M: Modulus + RandomMod> OTProtocolHandler<M> {
    /// Create a new instance of the protocol handler. The handler is valid only for the
    /// specified role, which can be changed with `rotate_role`.
    pub(crate) fn new<R: Rng + ?Sized>(
        role: Role,
        rng: &mut R,
        summation_modulus: M,
    ) -> Result<Self, String> {
        let three = M::one() + M::one() + M::one();
        if summation_modulus & M::one() == M::zero() && summation_modulus >= three {
            return Err("Summation modulus must be odd and at least 3.".to_string());
        }

        let mut new_otph = Self {
            role,
            modulus: summation_modulus,
            key: None,
            prf: None,
            c: None,
            r_c: None,
        };
        new_otph.init(rng);
        Ok(new_otph)
    }

    /// Get the current `Role`.
    pub(crate) fn get_role(&self) -> Role {
        self.role
    }

    /// Get the `modulus`.
    pub(crate) fn get_modulus(&self) -> M {
        self.modulus
    }

    /// Only used by `Role::Third`. Initialize the protocol.
    fn init<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        if self.role == Role::Third {
            // Generate a new random PRF key and associated PRF.
            let mut key: PRFKey = [0; 32];
            rng.fill_bytes(&mut key);
            self.key = Some(key);
            self.prf = Some(PRF::new(&key));
        }
    }

    /// Only used by `Role::Third`. Create an `OTMessage::Key` to send to `Role::Second`.
    pub(crate) fn create_key_message(&self) -> Result<OTMessage<M>, String> {
        if self.role != Role::Third {
            Err(format!("OTProtocolHandler::create_key_message called by incorrect role {:?} (expected {:?}).", self.role, Role::Third))
        } else {
            Ok(OTMessage::Key(self.key.unwrap()))
        }
    }

    /// Only used by `Role::Third`. Create an `OTMessage::Seed` to send to `Role::First`.
    pub(crate) fn create_seed_messages<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        count: usize,
    ) -> Result<Vec<OTMessage<M>>, String> {
        if self.role != Role::Third {
            return Err(format!("OTProtocolHandler::create_seed_messages called by incorrect role {:?} (expected {:?}).", self.role, Role::Third));
        }

        // Ensure `init` has been called.
        if self.key.is_none() || self.prf.is_none() {
            return Err("OTProtocolHandler::init must be called before OTProtocolHandler::create_seed_messages.".to_string());
        }

        // Check that `count` fits in `u32`.
        let count = u32::try_from(count).map_err(|_| {
            format!(
                "OTProtocolHandler::create_seed_messages called with too large count {}.",
                count
            )
        })?;

        let mut ret = Vec::<OTMessage<M>>::with_capacity(count as usize);
        for index in 0..count {
            // Create the PRF input as a concatenation of the index and c.
            let c = rng.gen::<u32>() & 1u32;
            let mut prf_in = [0u8; 8];
            prf_in[..4].copy_from_slice(&index.to_le_bytes());
            prf_in[4..8].copy_from_slice(&c.to_le_bytes());

            // Evaluate the PRF.
            let seed_c = self.prf.as_ref().unwrap().eval(&prf_in);

            // Add to the return value.
            ret.push(OTMessage::Seed(c == 1, seed_c));
        }

        Ok(ret)
    }

    /// Only used by `Role::First`. Process an `OTMessage::Seed` from `Role::Third` to
    /// output an `OTMessage::MaskedBit`. This takes as input this party's input bit as well.
    pub(crate) fn create_masked_bit_messages<R: Rng + SeedableRng>(
        &mut self,
        msgs: Vec<OTMessage<M>>,
        input_bits: &Vec<bool>,
    ) -> Result<Vec<OTMessage<M>>, String> {
        if self.role != Role::First {
            return Err(format!("OTProtocolHandler::create_masked_bit_messages called by incorrect role {:?} (expected {:?}).", self.role, Role::First));
        }

        if msgs.len() != input_bits.len() {
            return Err(format!("OTProtocolHandler::create_masked_bit_messages called with incorrect number of messages ({} messages, {} input bits).", msgs.len(), input_bits.len()));
        }
        let count = msgs.len();

        // These vectors store internal data for the protocol.
        self.c = Some(Vec::<bool>::with_capacity(count));
        self.r_c = Some(Vec::<M>::with_capacity(count));

        let mut ret = Vec::<OTMessage<M>>::with_capacity(count);
        for index in 0..count {
            let msg = msgs[index];

            // Ensure `msg` is of type `OTMessage::Seed`.
            if !matches!(msg, OTMessage::Seed(_, _)) {
                return Err(format!("OTProtocolHandler::create_masked_bit_messages called with incorrect message type {:?}.", msg));
            }

            // Extract the seed from the message.
            let (c, seed_c) = match msg {
                OTMessage::Seed(c, seed_c) => (c, seed_c),
                _ => unreachable!(),
            };

            // Set up the SeedableRng from seed_c.
            let mut prg = Shake256::default();
            prg.update(seed_c.as_slice());
            let mut xof_reader = prg.finalize_xof();
            let mut seed_c = R::Seed::default();
            xof_reader.read(seed_c.as_mut());
            let mut seeded_rng = R::from_seed(seed_c);

            // Store c and a random value r_c in Z_m.
            self.c.as_mut().unwrap().push(c);
            self.r_c
                .as_mut()
                .unwrap()
                .push(self.modulus.random_mod(&mut seeded_rng));

            ret.push(OTMessage::MaskedBit(input_bits[index] ^ c));
        }

        Ok(ret)
    }

    /// Only used by `Role::Second`. Process an `OTMessage::Key` from `Role::Third`.
    pub(crate) fn process_key_message(&mut self, msg: OTMessage<M>) -> Result<(), String> {
        if self.role != Role::Second {
            return Err(format!("OTProtocolHandler::process_key_message called by incorrect role {:?} (expected {:?}).", self.role, Role::Second));
        }

        // Ensure `msg` is of type `OTMessage::Key`.
        if !matches!(msg, OTMessage::Key(_)) {
            return Err(format!(
                "OTProtocolHandler::process_key_message called with incorrect message type {:?}.",
                msg
            ));
        }

        // Extract the key from the message.
        let key = match msg {
            OTMessage::Key(key) => key,
            _ => unreachable!(),
        };

        // Store the key and create the PRF.
        self.key = Some(key);
        self.prf = Some(PRF::new(&key));

        Ok(())
    }

    /// Only used by `Role::Second`. Process an `OTMessage::MaskedBit` from `Role::First`
    /// and output `OTMessage::Reveal` to `Role::First`.
    pub(crate) fn create_reveal_messages<R: Rng + SeedableRng>(
        &self,
        rng: &mut (impl Rng + ?Sized),
        msgs: Vec<OTMessage<M>>,
        input_bits: &Vec<bool>,
    ) -> Result<(Vec<OTMessage<M>>, Vec<M>), String> {
        if self.role != Role::Second {
            return Err(format!("OTProtocolHandler::create_reveal_messages called by incorrect role {:?} (expected {:?}).", self.role, Role::Second));
        }

        // Ensure `self.key` and `self.prf` are set.
        if self.key.is_none() || self.prf.is_none() {
            return Err("OTProtocolHandler::process_key_message must be called before OTProtocolHandler::create_reveal_messages.".to_string());
        }

        if msgs.len() != input_bits.len() {
            return Err(format!("OTProtocolHandler::create_reveal_messages called with incorrect number of messages ({} messages, {} input bits).", msgs.len(), input_bits.len()));
        }
        let count = msgs.len();

        let mut ret_to_s1 = Vec::<OTMessage<M>>::with_capacity(count);
        let mut ret_to_self = Vec::<M>::with_capacity(count);
        for index in 0..count {
            let index32 = index as u32;
            let msg = msgs[index];

            // Ensure `msg` is of type `OTMessage::MaskedBit`.
            if !matches!(msg, OTMessage::MaskedBit(_)) {
                return Err(format!(
                    "OTProtocolHandler::create_reveal_messages called with incorrect message type {:?}.",
                    msg
                ));
            }

            // Extract the masked bit from the message.
            let masked_bit = match msg {
                OTMessage::MaskedBit(masked_bit) => masked_bit,
                _ => unreachable!(),
            };

            // Compute seed_0 and seed_1 with the PRF.
            let mut prf_in = [0u8; 8];
            prf_in[..4].copy_from_slice(&index32.to_le_bytes());
            prf_in[4..8].copy_from_slice(&[0, 0, 0, 0]);
            let seed_0 = self.prf.as_ref().unwrap().eval(&prf_in);
            prf_in[4..8].copy_from_slice(&[1, 0, 0, 0]);
            let seed_1 = self.prf.as_ref().unwrap().eval(&prf_in);

            // Set up PRGs from seed_0 and seed_1.
            let mut prg = Shake256::default();
            prg.update(seed_0.as_slice());
            let mut xof_reader = prg.finalize_xof();
            let mut seed_0 = R::Seed::default();
            xof_reader.read(seed_0.as_mut());
            let mut seeded_rng_0 = R::from_seed(seed_0);

            let mut prg = Shake256::default();
            prg.update(seed_1.as_slice());
            let mut xof_reader = prg.finalize_xof();
            let mut seed_1 = R::Seed::default();
            xof_reader.read(seed_1.as_mut());
            let mut seeded_rng_1 = R::from_seed(seed_1);

            // Compute r_0 and r_1 in Z_m.
            let r_0 = self.modulus.random_mod(&mut seeded_rng_0);
            let r_1 = self.modulus.random_mod(&mut seeded_rng_1);

            // Sample a random number modulo `modulus` as this role's output.
            let m_2 = self.modulus.random_mod(rng);

            // Compute the `OTMessage::Reveal` to send to `Role::First`.
            let bit_prod_0: M = (input_bits[index] && masked_bit).into();
            let y_0 = self
                .modulus
                .sub_mod(bit_prod_0, self.modulus.add_mod(r_0, m_2));
            let bit_prod_1: M = (input_bits[index] && !masked_bit).into();
            let y_1 = self
                .modulus
                .sub_mod(bit_prod_1, self.modulus.add_mod(r_1, m_2));

            ret_to_s1.push(OTMessage::Reveal(y_0, y_1));
            ret_to_self.push(m_2);
        }

        Ok((ret_to_s1, ret_to_self))
    }

    /// Only used by `Role::First`. Process an `OTMessage::Reveal` from `Role::Second`.
    pub(crate) fn process_reveal_messages(
        &mut self,
        msgs: Vec<OTMessage<M>>,
    ) -> Result<Vec<M>, String> {
        if self.role != Role::First {
            return Err(format!("OTProtocolHandler::process_reveal_message called by incorrect role {:?} (expected {:?}).", self.role, Role::First));
        }

        // Ensure `c` and `r_c` are set.
        if self.c.is_none() || self.r_c.is_none() {
            return Err("OTProtocolHandler::create_masked_bit_messages must be called before OTProtocolHandler::process_reveal_messages.".to_string());
        }

        // Ensure the sizes of `c` and `r_c` match the number of messages.
        let count = msgs.len();
        if self.c.as_ref().unwrap().len() != count || self.r_c.as_ref().unwrap().len() != count {
            return Err(format!("OTProtocolHandler::process_reveal_messages called with incorrect number of messages ({} messages, {} c values, {} r_c values).", count, self.c.as_ref().unwrap().len(), self.r_c.as_ref().unwrap().len()));
        }

        let mut ret = Vec::<M>::with_capacity(msgs.len());
        for index in 0..count {
            let msg = msgs[index];

            // Ensure `msg` is of type `OTMessage::Reveal`.
            if !matches!(msg, OTMessage::Reveal(_, _)) {
                return Err(format!(
                    "OTProtocolHandler::process_reveal_messages called with incorrect message type {:?}.",
                    msg
                ));
            }

            // Extract the reveal from the message.
            let (y_0, y_1) = match msg {
                OTMessage::Reveal(y_0, y_1) => (y_0, y_1),
                _ => unreachable!(),
            };

            // Compute the output.
            let y = if self.c.as_ref().unwrap()[index] {
                y_1
            } else {
                y_0
            };
            let m_1 = self.modulus.add_mod(y, self.r_c.as_ref().unwrap()[index]);

            ret.push(m_1);
        }

        Ok(ret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::{OsRng, StdRng};

    fn test_ot<M: Modulus + RandomMod>(modulus: M, count: usize) {
        let mut rng = OsRng;

        // Create the protocol handlers.
        let mut ot_handler_1 = OTProtocolHandler::new(Role::First, &mut rng, modulus).unwrap();
        let mut ot_handler_2 = OTProtocolHandler::new(Role::Second, &mut rng, modulus).unwrap();
        let ot_handler_3 = OTProtocolHandler::new(Role::Third, &mut rng, modulus).unwrap();

        let key_msg = ot_handler_3.create_key_message().unwrap();
        ot_handler_2.process_key_message(key_msg).unwrap();

        let mut input_a = Vec::<bool>::with_capacity(count);
        let mut input_b = Vec::<bool>::with_capacity(count);
        for _ in 0..count {
            input_a.push(rng.gen::<bool>());
            input_b.push(rng.gen::<bool>());
        }

        let seed_msgs = ot_handler_3.create_seed_messages(&mut rng, count).unwrap();
        let masked_bit_msgs = ot_handler_1
            .create_masked_bit_messages::<StdRng>(seed_msgs, &input_a)
            .unwrap();
        let (reveal_msgs, m_2) = ot_handler_2
            .create_reveal_messages::<StdRng>(&mut rng, masked_bit_msgs, &input_b)
            .unwrap();
        let m_1 = ot_handler_1.process_reveal_messages(reveal_msgs).unwrap();

        assert_eq!(m_1.len(), count);
        assert_eq!(m_2.len(), count);
        for i in 0..count {
            assert_eq!(
                modulus.add_mod(m_1[i], m_2[i]),
                M::from(input_a[i] && input_b[i])
            );
        }
    }

    #[test]
    fn test_ot_mod3() {
        test_ot(3u32, 1000);
    }

    #[test]
    fn test_ot_mod101() {
        test_ot(101u32, 1000);
    }

    #[test]
    fn test_ot_mod_large() {
        test_ot((1u64 << 48) + 1, 1000);
    }
}
