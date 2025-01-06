// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

use crate::schema::{Attribute, Schema};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub(crate) struct PlainReport {
    pub attributes: Vec<Attribute>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Eq, Clone)]
pub(crate) struct PlainReportVector {
    pub schema: Schema,
    pub reports: Vec<PlainReport>,
}
