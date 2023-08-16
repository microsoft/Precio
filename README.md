## Overview of Precio
Precio is a Rust implementation of the protocol described in [eprint.iacr.org/2021/1490](https://eprint.iacr.org/2021/1490).
The goal of the protocol is to enable an *analyst* to compute histograms on *reports*, consisting on multiple *categorical attributes*, that are secret-shared among two *helper servers*.
The histograms can be computed in a layered manner, where the analyst may first want to compute a histogram on one of the attributes, then narrow down on a subset of reports where that attributes takes a particular value, and subsequently compute a histogram on another attribute on this subset.
The reports can also include *numerical attributes*, which the analyst can sum at any point while building the layered histogram.

### Clients
The reports are created by *clients*.
Each client is assumed to submit one or more reports to the system.
The clients may be malicious and submit false data in the reports, but they cannot do anything else to break the protocol or impact the results.
Therefore, as long as the majority of clients behave correctly, the results will be largely unaffected by a few reports with false data.
The clients get no output from the protocol.
Precio provides a minimal API for the clients to read in reports formatted as JSON and output secret-shares that the client will need to communicate to the helper servers.

### Helper Servers
Precio leverages three helper servers that are assumed to be non-colluding and semi-honest, *i.e.*, they follow the protocol execution but may try to infer additional information from the data they see.
We label the servers as `Server1`, `Server2`, and `Server3`.
During the protocol execution, `Server2` and `Server3` switch roles: the machine executing the role of `Server2` switches to instead execute the role of `Server3`, and *vice versa*.
The machine executing the role of `Server1` never changes its role.
More accurately, Precio contains two protocols: the sum protocol and the histogram protocol, which are executed repeatedly, as requested by the analyst who eventually gets the results.

#### Histogram Protocol
One layer of the histogram protocol proceeds in the following steps. Here we have omitted the details of the oblivious shuffle sub-protocol:
1. `Server1` and `Server2` hold secret-shares of a vector of reports.
1. All three servers agree on a categorical attribute for which they want to compute the histogram.
1. `Server1` and `Server2` add *noise reports* to their vector of (secret-shared) reports (see [Differential Privacy](#differential-privacy) below).
1. `Server1` and `Server2` communicate to each other how many reports they added in the previous step and pad their report vectors to have the same length.
1. `Server1`, `Server2`, and `Server3` jointly perform the *oblivious shuffle sub-protocol*, at the end of which `Server1` and `Server3` hold secret-shares of a permuted vector of reports, `Server2` holds nothing, and none of the servers knows the permutation.
1. `Server2` and `Server3` switch roles so that from this point it is `Server1` and `Server2` that hold secret-shared reports and `Server3` holds nothing.
1. `Server1` and `Server2` reveal to each other the secret-shares of the attribute for which they agreed to compute the histogram.
1. With the attribute values revealed, `Server1` and `Server2` compute the counts for each attribute value.
They prune attribute values with counts less than a given *prune threshold*, removing reports with the corresponding value.
1. `Server1` and `Server2` remove the processed categorical attribute from each report in their report vector.

#### Sum Protocol
The sum protocol proceeds in the following steps. Here we have omitted the details of the sum sub-protocol:
1. `Server1` and `Server2` hold secret-shares of a vector of reports.
1. All three servers agree on a numerical attribute for which they want to compute the sum.
1. All three servers execute the *sum sub-protocol*, at the end of which `Server1` and `Server2` learn secret-shares of the sum for the chosen numerical attribute.
1. `Server1` and `Server2` independently sample a noise value to add to their shares of the sum for differential privacy.
1. `Server1` and `Server2` reveal their shares to obtain a differentially private sum.
1. `Server1` and `Server2` remove the processed numerical attribute from each report in their report vector.

### Differential Privacy
Precio provides differential privacy guarantees for the results.
The helper servers can control the noise distribution, *i.e.*, how much noise is added at different steps of the protocol.
We refer the reader to [eprint.iacr.org/2021/1490](https://eprint.iacr.org/2021/1490) for an in-depth explanation of the differential privacy guarantee, including some reference parameters.

## API
In this section, we describe how to use the API for executing the histogram and sum protocols.

### Reports and Attributes
The reports are bit strings represented by a given number of `u32`s.
This number, denoted `REPORT_U32_SIZE` in the library, is a [const generic parameter](https://doc.rust-lang.org/reference/items/generics.html#const-generics) that needs to be specified at compile-time when setting up the client and server objects.
The report itself is broken into an arbitrary number of categorical or numerical attributes, each taking between 2 and 31 bits of the report.
The structure of the report is described by a user-configurable `Schema` object (see [src/schema.rs](src/schema.rs)).

The different attribute types are named as `Cxx` or `Nxx`, depending on whether they denote a categorical or numerical attribute.
The `xx` is a number denoting the number of bits reserved for this attribute.
The numerical attributes are slightly more complicated, because in addition to having a bit-size, they also hold a numerical value denoting a modulus for the input value.

**IMPORTANT: The modulus must be an odd positive integer and the largest valid value for a numerical attribute with modulus `m` is `floor(m / 2)`.
For example, to get a valid input range of 0-100, you use an 8-bit numerical attribute with modulus 201.**

### Serialization
Precio involves a substantial amount of communication between the different parties.
How this communication is achieved is beyond the scope of Precio; we simply provide [Serde](https://serde.rs/) serialization and deserialization for all of the relevant objects.

### Client
The client's API is in [src/client.rs](src/client.rs).
It consists of a single function: `client::create_report_shares`.
This function reads a schema and a report (or many reports) from a given JSON string and secret-shares the report(s) into two parts.
One of these parts needs to be sent to `Server1` and another to `Server2` (it does not matter which one gets which part).

As an example of the input, consider the following:
```json
{
    "schema":[["attr1","c2"],["attr2",{"n3":7}],["attr3","c4"],["attr4",{"n15":20001}]],
    "reports":[
        {"attributes":[{"c2":2},{"n3":[2,7]},{"c4":5},{"n15": [6107,20001]}]},
        {"attributes":[{"c2":0},{"n3":[1,7]},{"c4":13},{"n15":[139,20001]}]},
        {"attributes":[{"c2":1},{"n3":[3,7]},{"c4":5},{"n15":[9800,20001]}]}
    ]
}
```
This specifies a schema with four attributes, unimaginatively named `attr1`, `attr2`, `attr3`, and `attr4` here.
In practice, the names should be descriptive of the meaning of the attribute.
Here `attr1` and `attr3` are categorical, of sizes 2 and 4 bits, respectively, whereas `attr2` and `attr4` are numerical, of sizes 3 and 15 bits, respectively.
The numerical attributes specify moduli 6 and 32767, which means that the value given to `attr2` must be in the range `[0, 3]` and the value given to `attr4` must be in the range `[0, 10000]`, as the largest valid input value is half of the modulus (rounded down).

In this example, the client specifies three reports with the following attribute values:
| attr1 | attr2 | attr3 | attr4 |
|-------|-------|-------|-------|
|   2   |   2   |   5   |  6107 |
|   0   |   1   |   13  |   139 |
|   1   |   3   |   5   |  9800 |

### Histogram API
The server's API is much more complex and makes up the vast majority of the library.
First, they do not already exist, *e.g.*, from prior runs of the histogram protocol, new `server::Server` instances are created by calling `Server::<REPORT_U32_SIZE>::new(role: Role, schema: Schema)`.
The servers simply specify their desired role (`Role::First`, `Role::Second`, or `Role::Third`) and the expected schema for the data that will be loaded in this server.

All three servers need to decide on an attribute name from the schema for which they want to compute a histogram; we call this attribute the *current attribute*.
Most of the server functions discussed below require the name of the current attribute to be given as a parameter.

Both `Server1` and `Server2`, *i.e.*, servers with `Role::First` and `Role::Second`, respectively, call `Server::add_noise_reports`.
The function also expects two noise distribution objects, which can be instances of either `hist_noise::Laplace` or `hist_noise::Gaussian`, depending on the kind of differential privacy mechanism the user wants to use.
Setting up the noise distributions is in principle straightforward, but requires the user to specify a truncation point `m` in addition to the distribution parameters.
We refer the reader to [eprint.iacr.org/2021/1490](https://eprint.iacr.org/2021/1490) for more information on how to choose these distributions.
The `add_noise_reports` functions returns the number of noise reports that were added; both servers need to exchange these numbers for the next step.

Both `Server1` and `Server2` need to pad their report vectors to have the same length using the `Server::add_empty_reports` function.
This requires as an input the number of padding reports to be added, *i.e.*, the number received from the other server.

Now the three servers need to execute the oblivious shuffle protocol.
In this protocol, the three servers need to call, in order `Server2`, `Server3`, `Server1`, the function `Server::oblivious_permute::<R>(...)`, where `R` is a random number generator type that implements the `rand::SeedableRng` trait.
In addition, each pair of servers needs to agree on a random seed compatible with `R` and pass those to the function.
The function output needs to be sent to the next server as input to their call of the function (`Server2`'s call takes `None` as input and `Server3`'s output can be ignored).
Finally, `Server2` and `Server3` call the `Server::rotate_role()` function to switch their roles.

Concretely, the functions must be called in the following sequence. Here we use `StdRng` as an example of a `SeedableRng`.
```rust
// Random seed chosen and shared between Server1 and Server2.
let seed12 = rng.gen::<[u8; 32]>();

// Random seed chosen and shared between Server2 and Server3.
let seed23 = rng.gen::<[u8; 32]>();

// Random seed chosen and shared between Server1 and Server3.
let seed13 = rng.gen::<[u8; 32]>();

// First, Server2 calls oblivious_permute. It sends the output to Server1.
let to_server1 = server2
    .oblivious_permute::<StdRng>(seed23.clone(), seed12.clone(), None)
    .unwrap();

// Next, Server1 calls oblivious_permute with the output received from
// Server2. It sends the output to Server3.
let to_server3 = server1
    .oblivious_permute::<StdRng>(seed12.clone(), seed13.clone(), to_server1)
    .unwrap();

// Finally, Server3 calls oblivious_permute with the output received from
// Server1. It receives no output.
server3
    .oblivious_permute::<StdRng>(seed13.clone(), seed23.clone(), to_server3)
    .unwrap();

// Server2 and Server3 call rotate_role independently in any order.
server2.rotate_role().unwrap();
server3.rotate_role().unwrap();
```

After the oblivious shuffle protocol, the machine that earlier played the role of `Server3` now plays the role of `Server2`, and *vice versa*.
Both `Server1` and (the new) `Server2` now hold secret shares of the reports.
They extract and exchange the shares for their chosen attribute by calling the `Server::get_attr_values(&self, attr_name: &str)` function.
To reveal the shares, they call `Server::reveal_attr(&mut self, attr_name: &str, attr_shares: Vec<u32>)`, where `attr_shares` is the vector of shares the servers receive from each other.

It is often not meaningful to maintain all reports, especially if the attribute value distribution has a long tail.
In this case, both `Server1` and `Server2` can at this point call `Server::prune(&mut self, attr_name: &str, threshold: usize)`, which removes all reports with an attribute value such that it appear with a count of less than `threshold`.

Now `Server1` is ready to create a `server::Histogram` object representing a histogram (for this attribute) for their reports.
They do this by calling `Server::make_histogram(&self, attr_name: &str, m: i64, prune_threshold: usize)`.
Here `m` is the same truncation point that was used in creating the noise distribution earlier and `prune_threshold` should be the same as the `threshold` specified earlier when calling the `Server::prune` function.
The functions returns an `Rc<RefCell<Histogram>>`, wrapped in a `Result`.

If `Server1` already holds a histogram from an earlier layer of the protocol, *i.e.*, from processing another attribute, it can now attach the new sub-histogram to the earlier histogram by calling `Histogram::join_at(&mut self, join_at: u32, histogram: Rc<RefCell<Histogram>>`, where `join_at` specifies the attribute value for the previous attribute at which to join `histogram` as a sub-histogram.
For example, if the user first computed a histogram on `attr1` and then filtered (see below) to exploring only those reports where `attr1` has a value of 3, then `Histogram::join_at` would be called for the histogram for `attr1` and the `join_at` value would be set to 3.

At this point the user (analyst) needs to decide what to do.
They may want to filter down by computing a sub-histogram for another attribute when the current attribute takes a particular value, say 5.
In this case, they can call `Server::split_at(&mut self, attr_name: &str, attr_value: u32)`, with `attr_value` set to 5.
The function returns another `Server` object that contains exactly those reports where the current attribute has value 5, whereas the current server (`&self`) retains all but those reports.
The function also removes the current attribute from all of the reports.
Finally, `Server3` needs to adjust its schema to match the updated schema of `Server1` and `Server2` (*i.e.*, with the current attribute removed).
It can do this by simply calling `Server::remove_attr(&mut self, attr_name: &str)`.

This completes a single layer of the histogram protocol.
There are multiple examples in the test section of [src/server/server.rs](src/server/server.rs) showing the entire process.
Note that all of these examples run all three servers, which in a real-world execution would not be the case.

### Sum API
The sum protocol is executed between runs of the layered histogram protocol.
It is much simpler, consisting of just a few back-and-forth communications between the servers.

First, the three servers need to agree upon a *summation modulus*; an odd positive integer modulo which the sum is computed.
In practice, this should be either a 32-bit or 64-bit integer, depending on which one is needed to hold the expected aggregate value.
By default, the summation modulus is a `u32`.
To change it into a `u64`, enable the feature `wide_summation_modulus`.

Next the servers must perform the following sequence of operations:
```rust
// Each server independently calls the Server::summation_initialize
// function with the chosen numerical attribute name and the chosen
// summation modulus.
server1
    .summation_initialize(&mut rng, attr_name, summation_modulus)
    .unwrap();
server2
    .summation_initialize(&mut rng, attr_name, summation_modulus)
    .unwrap();
server3
    .summation_initialize(&mut rng, attr_name, summation_modulus)
    .unwrap();

// Server3 calls Server::summation_create_key and sends the output to
// Server2. It also calls Server::summation_create_seeds and sends the
// result to Server1.
let from_s3_to_s2 = server3.summation_create_key().unwrap();
let report_count = server3.get_report_count().unwrap();
let from_s3_to_s1 = server3
    .summation_create_seeds(&mut rng, report_count)
    .unwrap();

// Server1 uses the data it received from Server3 to call
// Server::summation_create_masked_bits and sends the result to Server2.
let from_s1_to_s2 = server1
    .summation_create_masked_bits::<StdRng>(from_s3_to_s1)
    .unwrap();

// Server2 uses the data it received from Server3 to call
// Server::summation_receive_key. The function outputs nothing.
server2.summation_receive_key(from_s3_to_s2).unwrap();

// Server2 uses the data it received from Server1 to call
// Server::summation_create_reveal_msgs. The function outputs a pair
// of objects: a message that is sent back to Server1 and final output
// for Server2.
let (from_s2_to_s1, s2_output) = server2
    .summation_create_reveal_msgs::<StdRng>(&mut rng, from_s1_to_s2)
    .unwrap();

// Server1 uses the data from Server2 to compute its own final output
// by calling Server::summation_receive_reveal_msgs.
let s1_output = server1
    .summation_receive_reveal_msgs(from_s2_to_s1)
    .unwrap();

// Each server must call Server::summation_finalize at this point.
server1.summation_finalize();
server2.summation_finalize();
server3.summation_finalize();
```
In the end, `Server1` and `Server2` output numbers modulo the summation modulus,
whereas `Server3` outputs nothing.
Both `Server1` and `Server2` need to use standard differential privacy techniques to add noise to their result to achieve differential privacy for the output.
Note that they should not use the distributions `hist_noise::Laplace` or `hist_noise::Gaussian`, as those are special truncated shifted noise distributions meant to be used by the histogram protocol.
Finally, `Server2` sends it differentially private output to `Server1`, who reveals the result by adding the two numbers modulo the summation modulus.
For the modular addition, you can use `u32::add_mod` or `u64::add_mod` by using `arith::Modulus` (see [src/arith.rs](src/arith.rs)).

## Examples
The above explanation of the core APIs omits many details.
However, each `.rs` file includes tests for the functionalities defined in it, and [tests/](tests/) includes multiple complete examples of using Precio.

**IMPORTANT: The hard-coded protocol parameters in these examples are not intended to be secure.
The examples merely demonstrate how to use the library.
To learn how to set secure parameters, please see [eprint.iacr.org/2021/1490](https://eprint.iacr.org/2021/1490).**

The following examples are the most relevant ones, all in [tests/](tests/):
1. `layered_histogram` demonstrates building a few layers of a layered histogram for artificially generated test data.
1. `full_histogram_exploration` computes a full private histogram, *i.e.*, all possible paths, for artificially generated test data.
The test can easily be modified; for example, one may want to change the number of clients (`report_count`), the test data distribution (`zipf_parameter`), the noise distibutions (`noise_distr` and `noise_distr_dummy`).
The schema can be changed as well, but make sure to set the `U32_COUNT` value to be such that the sum of the attribute bit-sizes in the schema is no larger than `32 * U32_COUNT`.
Note that this test only handles categorical attributes.
Finally, you can change `reports_prune_threshold` and `histogram_prune_threshold` to change the pruning behavior.
Setting these to zero results in an exact (up to differential privacy) histogram.
1. `full_histogram_exploration_from_file` is the same as `full_histogram_exploration`, but it reads the client reports from a JSON file called `full_histogram.json` instead of generating test data.
The format of this input file was described in above in [Client](#client).
1. `heavy_hitter` demonstrates finding the most frequently occurring report, the *heavy hitter* from a set of reports.
1. `heavy_hitter_from_file` is the same as `heavy_hitter`, but it reads the client reports from a JSON file called `heavy_hitter.json` instead of generating test data.
1. `histogram_with_summation` demonstrates a few runs of the sum protocol interleaved with layered histogram exploration.

To run these examples, just use `cargo test <example_name>`, but remember to use the `--release` switch if you hope to measure timings.
The examples also print some information to stdout, including communication and timing measurements, which you can see by running the tests with `--nocapture`.
The `_from_file` examples are marked as `ignored`, so if you want to run them you need to additionally use the `--ignored` switch.
All in all, to run the examples, use
```
cargo test <example_name> --release -- --nocapture [--ignored]
```