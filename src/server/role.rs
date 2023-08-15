// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

/// The role of a server. Servers with roles First and Second
/// hold shares of `ReportVector`s. Servers with role Third
/// assist in the oblivious shuffle protocol only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    First,
    Second,
    Third,
}

impl Role {
    /// Rotate the role after an oblivious permutation.
    pub(crate) fn rotate(&mut self) {
        *self = match *self {
            Role::First => Role::First,
            Role::Second => Role::Third,
            Role::Third => Role::Second,
        };
    }
}
