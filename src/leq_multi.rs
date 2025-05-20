use fancy_garbling::{
    AllWire, BinaryBundle, Bundle, BinaryGadgets, Fancy, FancyArithmetic, FancyBinary, FancyInput,
    FancyReveal,
    twopac::semihonest::{Evaluator, Garbler},
    util,
};
use ocelot::{ot::AlszReceiver as OtReceiver, ot::AlszSender as OtSender};
use scuttlebutt::{AbstractChannel, AesRng, Channel};
use std::{fmt::Debug, io::{BufReader, BufWriter}, os::unix::net::UnixStream};

// We assume these constants and types are already defined:
use crate::field::FieldElm;
const MODULUS_64: u64 = 9223372036854775783u64;
const CONSTANT: u128 = 11 + 1; // per your code

/// Structure to hold both parties’ wires for the “sum” circuit.
struct SUMInputs<F> {
    pub garbler_wires: BinaryBundle<F>,
    pub evaluator_wires: BinaryBundle<F>,
}

/// Multi‑input version of the garbler’s wire exchange.
/// It encodes each u128 input into 128 bits and concatenates their wires.
fn gb_set_fancy_inputs_multi<F, E>(gb: &mut F, inputs: &[u128]) -> SUMInputs<F::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    let nbits = 128;
    let mut garbler_bits = Vec::new();
    for &input in inputs {
        // For each test, encode the 128‑bit number.
        let bundle = gb.bin_encode(input, nbits).unwrap();
        garbler_bits.extend(bundle.wires());
    }
    let garbler_bundle = BinaryBundle::from(Bundle::new(garbler_bits));
    // Receive evaluator wires for all tests at once.
    let evaluator_bundle = gb.bin_receive(nbits * inputs.len()).unwrap();
    SUMInputs {
        garbler_wires: garbler_bundle,
        evaluator_wires: evaluator_bundle,
    }
}

/// Multi‑input version of the evaluator’s wire exchange.
/// Here the evaluator receives the garbler’s wires and encodes its own inputs.
fn ev_set_fancy_inputs_multi<F, E>(ev: &mut F, inputs: &[u128]) -> SUMInputs<F::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    let nbits = 128;
    let garbler_bundle = ev.bin_receive(nbits * inputs.len()).unwrap();
    let mut evaluator_bits = Vec::new();
    for &input in inputs {
        let bundle = ev.bin_encode(input, nbits).unwrap();
        evaluator_bits.extend(bundle.wires());
    }
    let evaluator_bundle = BinaryBundle::from(Bundle::new(evaluator_bits));
    SUMInputs {
        garbler_wires: garbler_bundle,
        evaluator_wires: evaluator_bundle,
    }
}

/// A gadget function that, given a bundle of wires for multiple tests,
/// performs the binary addition (no carry) and then for each block of 128 bits:
///   - converts it to a u128 (via util::u128_from_bits),
///   - reduces modulo MODULUS_64,
///   - subtracts CONSTANT, and
///   - extracts the MSB (by shifting right 127 bits).
/// Returns a Vec<bool> with one result per test.
fn multiple_fancy_less_than<F>(
    f: &mut F,
    wire_inputs: SUMInputs<F::Item>,
    num_tests: usize,
) -> Result<Vec<bool>, F::Error>
where
    F: FancyReveal + Fancy + BinaryGadgets + FancyBinary + FancyArithmetic,
    F::Item: Clone,
{
    let nbits = 128;
    // Perform the binary addition over all the concatenated wires.
    let sum_bundle = f.bin_addition_no_carry(&wire_inputs.garbler_wires, &wire_inputs.evaluator_wires)?;
    // Reveal the output bits for all tests; this should be a vector of length num_tests * 128.
    let out = f.outputs(sum_bundle.wires())?;
    let out = out.expect("Outputs should be produced");
    let mut results = Vec::with_capacity(num_tests);
    for i in 0..num_tests {
        // Extract the i-th block of 128 bits.
        let bits = &out[i * nbits..(i + 1) * nbits];
        let raw_sum = util::u128_from_bits(bits) % (MODULUS_64 as u128);
        let final_sum = raw_sum.wrapping_sub(CONSTANT);
        // The MSB is at position 127.
        let msb = final_sum >> 127;
        results.push(msb == 1);
    }
    Ok(results)
}

/// The garbler’s multi‑input less‑than test --- analogous to multiple_gb_equality_test.
pub fn multiple_gb_less_test<C>(
    rng: &mut AesRng,
    channel: &mut C,
    inputs: &[u128],
) -> Vec<bool>
where
    C: AbstractChannel + Clone,
{
    let num_tests = inputs.len();
    let mut gb = Garbler::<C, AesRng, OtSender, AllWire>::new(channel.clone(), rng.clone())
        .unwrap();
    // Prepare the wires for all tests.
    let wire_inputs = gb_set_fancy_inputs_multi(&mut gb, inputs);
    // Compute the less-than (i.e. negative) gadget over all tests.
    let results = multiple_fancy_less_than(&mut gb, wire_inputs, num_tests).unwrap();
    channel.flush().unwrap();
    // Wait for acknowledgment.
    let mut ack = [0u8; 1];
    channel.read_bytes(&mut ack).unwrap();
    results
}

/// The evaluator’s multi‑input less‑than test --- analogous to multiple_ev_equality_test.
pub fn multiple_ev_less_test<C>(
    rng: &mut AesRng,
    channel: &mut C,
    inputs: &[u128],
) -> Vec<bool>
where
    C: AbstractChannel + Clone,
{
    let num_tests = inputs.len();
    let mut ev = Evaluator::<C, AesRng, OtReceiver, AllWire>::new(channel.clone(), rng.clone())
        .unwrap();
    let wire_inputs = ev_set_fancy_inputs_multi(&mut ev, inputs);
    let results = multiple_fancy_less_than(&mut ev, wire_inputs, num_tests).unwrap();
    // Send an acknowledgment.
    channel.write_bytes(&[1u8]).unwrap();
    channel.flush().unwrap();
    results
}
