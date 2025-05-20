// // less_than_gc.rs

// use fancy_garbling::{
//     twopac::semihonest::{Evaluator, Garbler},
//     util,
//     AllWire,
//     BinaryBundle,
//     BundleGadgets,
//     Fancy,
//     FancyArithmetic,
//     FancyBinary,
//     FancyInput,
//     FancyReveal,
// };
// use ocelot::{
//     ot::AlszReceiver as OtReceiver,
//     ot::AlszSender as OtSender,
// };
// use scuttlebutt::{
//     AbstractChannel, AesRng, Channel, SyncChannel,
// };
// use std::{
//     fmt::Debug,
//     io::{BufReader, BufWriter, Read, Write},
//     os::unix::net::UnixStream,
//     time::Instant,
// };
// use rayon::prelude::*;

// /// A structure that contains both the garbler’s and the evaluator’s wire bundles.
// /// In the equality test repo this is named `EQInputs`; here we call it `SUMInputs`
// struct SUMInputs<F> {
//     pub garbler_wires: BinaryBundle<F>,
//     pub evaluator_wires: BinaryBundle<F>,
// }

// // Our modulus and constant values.
// const MODULUS_64: u64 = 9223372036854775783u64;
// // The constant to subtract—for example, (delta^P + 1)
// const CONSTANT: u128 = 11 + 1;

// // --------------------------------------------------------------------------
// // Wire exchange functions (same structure as the equality version)
// // --------------------------------------------------------------------------

// /// The garbler's wire exchange method.
// /// Encodes the garbler’s input (a slice of u16) into binary wires,
// /// and receives the evaluator’s corresponding wires.
// /// `num_tests` is the number of independent tests (each test should use a fixed number of bits).
// fn gb_set_fancy_inputs<F, E>(gb: &mut F, input: &[u16], num_tests: usize) -> SUMInputs<F::Item>
// where
//     F: FancyInput<Item = AllWire, Error = E>,
//     E: Debug,
// {
//     // The garbler encodes its input into binary wires.
//     let garbler_wires: BinaryBundle<F::Item> =
//         gb.encode_bundle(&input, &vec![2; input.len()]).map(BinaryBundle::from).unwrap();
//     // The evaluator’s labels are received via OT.
//     let evaluator_wires: BinaryBundle<F::Item> =
//         gb.bin_receive(input.len() - num_tests).unwrap();
//     SUMInputs {
//         garbler_wires,
//         evaluator_wires,
//     }
// }

// /// The evaluator's wire exchange method.
// /// Receives the garbler’s wires and encodes its own input.
// fn ev_set_fancy_inputs<F, E>(ev: &mut F, input: &[u16], num_tests: usize) -> SUMInputs<F::Item>
// where
//     F: FancyInput<Item = AllWire, Error = E>,
//     E: Debug,
// {
//     let nwires = input.len();
//     let garbler_wires: BinaryBundle<F::Item> =
//         ev.bin_receive(nwires + num_tests).unwrap();
//     let evaluator_wires: BinaryBundle<F::Item> =
//         ev.encode_bundle(input, &vec![2; nwires]).map(BinaryBundle::from).unwrap();
//     SUMInputs {
//         garbler_wires,
//         evaluator_wires,
//     }
// }

// // --------------------------------------------------------------------------
// // Extension trait for wire-level operations.
// // (In the repo they add some helper functions for equality; you can add more if needed.)
// // --------------------------------------------------------------------------
// pub trait BinaryGadgets: FancyBinary + BundleGadgets {
//     // (You can include equality gadgets here if needed; we assume the necessary ones 
//     // are already implemented by the garbler/evaluator.)
// }

// impl<C, R, S, W> BinaryGadgets for Garbler<C, R, S, W>
// where
//     Self: FancyBinary + BundleGadgets,
// {
// }

// impl<C, R, S, W> BinaryGadgets for Evaluator<C, R, S, W>
// where
//     Self: FancyBinary + BundleGadgets,
// {
// }

// // --------------------------------------------------------------------------
// // Gadget function: compute the "is negative" (or less-than zero) result.
// // This is analogous to fancy_equality but performs arithmetic.
// // --------------------------------------------------------------------------
// fn fancy_is_negative<F>(
//     f: &mut F,
//     wire_inputs: SUMInputs<F::Item>,
//     num_tests: usize,
// ) -> Result<BinaryBundle<F::Item>, F::Error>
// where
//     F: FancyReveal + Fancy + BinaryGadgets + FancyBinary + FancyArithmetic,
//     F::Item: Clone,
// {
//     // First, add the two sets of wires (without carry).
//     let sum_bundle = f.bin_addition_no_carry(&wire_inputs.garbler_wires, &wire_inputs.evaluator_wires)?;
//     // Reveal the sum wires. We expect for each test 128 output bits.
//     let out = f.outputs(sum_bundle.wires())?
//         .expect("Garbler/Evaluator should produce outputs");
//     let nbits = 128;
//     let mut results = Vec::with_capacity(num_tests);
//     for i in 0..num_tests {
//         // Extract the i-th block (128 bits) corresponding to one test.
//         let bits = &out[i * nbits..(i + 1) * nbits];
//         // Convert the bits to a u128.
//         let raw_sum = util::u128_from_bits(bits) % (MODULUS_64 as u128);
//         // Subtract the constant.
//         let final_sum = raw_sum.wrapping_sub(CONSTANT);
//         // The MSB (bit 127) shows whether the value is negative.
//         let msb = final_sum >> 127;
//         // We consider msb == 1 as true.
//         results.push(msb == 1);
//     }
//     // Option: pack the resulting booleans back into wires using the circuit’s “one” or “zero”.
//     // Here we map each boolean to a corresponding wire.
//     let wires = results.into_iter()
//         .map(|b| if b { f.one() } else { f.zero() })
//         .collect::<Result<Vec<_>, _>>()?;
//     Ok(BinaryBundle::from(Bundle::new(wires)))
// }

// // --------------------------------------------------------------------------
// // Multi-input functions analogous to multiple_{gb,ev}_equality_test,
// // but for our less-than (is negative) circuit.
// // --------------------------------------------------------------------------

// /// Multiple garbler less-than test:
// /// The garbler prepares its inputs, calls fancy_is_negative, and outputs the result.
// pub fn multiple_gb_less_test<C>(
//     rng: &mut AesRng,
//     channel: &mut C,
//     inputs: &[Vec<u16>], // each test's input as a vector of u16 bits
// ) -> Vec<bool>
// where
//     C: AbstractChannel + Clone,
// {
//     let num_tests = inputs.len();
//     let mut gb = Garbler::<C, AesRng, OtSender, AllWire>::new(channel.clone(), rng.clone())
//         .unwrap();
//     // Here we assume that the inputs are already prepared (possibly masked)
//     // and concatenated as in the repository code.
//     let wire_inputs = inputs.iter()
//                               .flatten()
//                               .cloned()
//                               .collect::<Vec<u16>>();
//     let wires = gb_set_fancy_inputs(&mut gb, wire_inputs.as_slice(), num_tests);
//     let less_bundle = fancy_is_negative(&mut gb, wires, num_tests).unwrap();
//     // Have the garbler output the wires.
//     gb.outputs(less_bundle.wires()).unwrap();
//     channel.flush().unwrap();
//     let mut ack = [0u8; 1];
//     channel.read_bytes(&mut ack).unwrap();
//     // For simplicity, we call outputs() again to get clear bits (in practice, you may store these).
//     let out = gb.outputs(less_bundle.wires()).unwrap()
//         .expect("Outputs expected");
//     // Interpret each output bit as a boolean.
//     out.iter().map(|&r| r == 1).collect()
// }

// /// Multiple evaluator less-than test:
// /// The evaluator prepares its inputs, calls fancy_is_negative, and returns the result.
// pub fn multiple_ev_less_test<C>(
//     rng: &mut AesRng,
//     channel: &mut C,
//     inputs: &[Vec<u16>],
// ) -> Vec<bool>
// where
//     C: AbstractChannel + Clone,
// {
//     let num_tests = inputs.len();
//     let mut ev = Evaluator::<C, AesRng, OtReceiver, AllWire>::new(channel.clone(), rng.clone())
//         .unwrap();
//     let input_vec = inputs.to_vec()
//                           .into_iter()
//                           .flatten()
//                           .collect::<Vec<u16>>();
//     let wires = ev_set_fancy_inputs(&mut ev, input_vec.as_slice(), num_tests);
//     let less_bundle = fancy_is_negative(&mut ev, wires, num_tests).unwrap();
//     let output = ev.outputs(less_bundle.wires()).unwrap().unwrap();
//     let results: Vec<bool> = output.iter().map(|r| *r == 1).collect();
//     channel.write_bytes(&[1u8]).unwrap();
//     channel.flush().unwrap();
//     results
// }

// // --------------------------------------------------------------------------
// // (Optional) A test function to run the circuit locally using UnixStream
// // --------------------------------------------------------------------------
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::sync::mpsc;
    
//     #[test]
//     fn less_than_gc() {
//         // For testing, we simulate inputs.
//         // Here each input is represented as a Vec<u16>; 
//         // in practice these would be the binary representation of a 128-bit number.
//         // For simplicity, we assume each input vector is already the correct length.
//         let gb_inputs = vec![
//             vec![0, 1, 1, 0], // example bit-vectors for test 0
//             vec![0, 0, 0, 0], // test 1
//             vec![1, 1, 1, 0], // test 2
//         ];
//         let ev_inputs = gb_inputs.clone(); // suppose evaluator uses same inputs
        
//         // Expected output could be computed in the clear.
//         // Here we simply simulate expected booleans.
//         // (Replace with your expected value computation.)
//         let expected: Vec<bool> = vec![false, false, false]; 
        
//         let (sender, receiver) = UnixStream::pair().unwrap();
//         let (result_sender, result_receiver) = mpsc::channel();

//         let x = std::thread::spawn(move || {
//             let mut rng_gb = AesRng::new();
//             let reader = BufReader::new(sender.try_clone().unwrap());
//             let writer = BufWriter::new(sender);
//             let mut channel = Channel::new(reader, writer);
//             let res = multiple_gb_less_test(&mut rng_gb, &mut channel, gb_inputs.as_slice());
//             result_sender.send(res).unwrap();
//         });

//         let mut rng_ev = AesRng::new();
//         let reader = BufReader::new(receiver.try_clone().unwrap());
//         let writer = BufWriter::new(receiver);
//         let mut channel = Channel::new(reader, writer);
//         let results = multiple_ev_less_test(&mut rng_ev, &mut channel, ev_inputs.as_slice());

//         let masks = result_receiver.recv().unwrap();
//         x.join().unwrap();

//         // For testing, simply compare lengths and dummy expected values.
//         assert_eq!(masks.len(), results.len(), "Results length mismatch");
//         // You can add further assertions comparing 'masks' and 'results' to the expected outcomes.
//         for (i, (m, r)) in masks.iter().zip(results.iter()).enumerate() {
//             println!("Test {}: garbler returned {}, evaluator returned {}", i, m, r);
//             // For example, ensure they agree:
//             assert_eq!(m, r, "Mismatch at test {}", i);
//         }
//     }
// }


// less_than_gc_standard_addition.rs

use fancy_garbling::{
    twopac::semihonest::{Evaluator, Garbler},
    util,
    AllWire,
    BinaryBundle,
    BundleGadgets,
    Fancy,
    FancyArithmetic,
    FancyBinary,
    FancyInput,
    FancyReveal,
};
use ocelot::{
    ot::AlszReceiver as OtReceiver,
    ot::AlszSender as OtSender,
};
use scuttlebutt::{
    AbstractChannel, AesRng, Channel, SyncChannel,
};
use std::{
    fmt::Debug,
    io::{BufReader, BufWriter, Read, Write},
    os::unix::net::UnixStream,
    time::Instant,
};
use rayon::prelude::*;
use num_traits::{Zero, One};
use fancy_garbling::WireLabel;
use fancy_garbling::Bundle;


/// A structure that contains both the garbler’s and evaluator’s wire bundles.
/// (Analogous to EQInputs in the equality test.)
struct SUMInputs<F> {
    pub garbler_wires: BinaryBundle<F>,
    pub evaluator_wires: BinaryBundle<F>,
}

// Our modulus and constant parameters.
const MODULUS_64: u64 = 9223372036854775783u64;
// The constant to subtract—for example, (delta^P + 1)
const CONSTANT: u128 = 11 + 1; // Adjust as needed

// --------------------------------------------------------------------------
// Helper gadget: Standard addition (with carry) for binary wires
// --------------------------------------------------------------------------

/// Compute OR of two wires using available operations.
/// We define: a OR b = NOT( NOT(a) AND NOT(b) ).
fn f_or<F>(f: &mut F, a: F::Item, b: F::Item) -> Result<F::Item, F::Error>
where
    F: FancyBinary,
{
    let not_a = f.negate(&a)?;
    let not_b = f.negate(&b)?;
    let and_val = f.and(&not_a, &not_b)?;
    f.negate(&and_val)
}

/// Full-adder gadget: Given bits a, b, and carry-in c (each F::Item),
/// returns (sum, carry_out) representing the full adder output.
fn full_adder<F>(f: &mut F, a: &F::Item, b: &F::Item, c: &F::Item) 
    -> Result<(F::Item, F::Item), F::Error>
where
    F: FancyBinary,
{
    // sum = a XOR b XOR c
    let ab_xor = f.xor(a, b)?;
    let sum = f.xor(&ab_xor, c)?;
    // carry_out = (a AND b) OR (a AND c) OR (b AND c)
    let ab_and = f.and(a, b)?;
    let ac_and = f.and(a, c)?;
    let bc_and = f.and(b, c)?;
    let temp = f_or(f, ab_and, ac_and)?;
    let carry_out = f_or(f, temp, bc_and)?;
    Ok((sum, carry_out))
}

/// Given two slices of bits (each of length nbits, assumed to be in MSB-first order),
/// perform standard (ripple-carry) addition and return a vector of nbits wires representing the sum.
/// (We discard the final carry-out.)
fn bin_addition_with_carry<F>(
    f: &mut F,
    bits_x: &[F::Item],
    bits_y: &[F::Item],
) -> Result<Vec<F::Item>, F::Error>
where
    F: FancyBinary,
    F::Item: Clone + WireLabel, // require WireLabel
{
    let nbits = bits_x.len();
    // Use the associated function from WireLabel:
    let mut carry = <F::Item as WireLabel>::zero(2);
    let mut result_rev = Vec::with_capacity(nbits);
    for i in (0..nbits).rev() {
        let (sum, new_carry) = full_adder(f, &bits_x[i], &bits_y[i], &carry)?;
        result_rev.push(sum);
        carry = new_carry;
    }
    result_rev.reverse();
    Ok(result_rev)
}


/// A gadget function that – given two bundles of wires (from garbler and evaluator)
/// – adds them using standard addition (with carry), then for each test:
///   - converts the resulting bits into a u128,
///   - reduces modulo MODULUS_64, subtracts CONSTANT, and
///   - extracts the MSB (bit at position nbits-1) as the output.
/// Returns a vector of booleans (true if MSB==1).
fn fancy_is_negative<F>(
    f: &mut F,
    wire_inputs: SUMInputs<F::Item>,
    num_tests: usize,
) -> Result<BinaryBundle<F::Item>, F::Error>
where
    F: FancyReveal + Fancy + BundleGadgets + FancyBinary + FancyArithmetic,
    F::Item: Clone + WireLabel,  // require WireLabel so that zero(q) is available
{
    let nbits = 128; // the bit-width used for each test
    let sum_bits = bin_addition_with_carry(
        f,
        wire_inputs.garbler_wires.wires(),
        wire_inputs.evaluator_wires.wires(),
    )?;
    let out = f.outputs(&sum_bits)?
        .expect("Both parties should produce outputs");
    let mut results = Vec::with_capacity(num_tests);
    for i in 0..num_tests {
        let block = &out[i * nbits..(i + 1) * nbits];
        let raw_sum = util::u128_from_bits(block) % (MODULUS_64 as u128);
        let final_sum = raw_sum.wrapping_sub(CONSTANT);
        // Since the bits are in MSB-first order, the most significant bit is at index 0.
        // Here we shift right by 127 to extract bit 127.
        let msb = final_sum >> 127;
        results.push(msb == 1);
    }
    // Map booleans back into wires.
    // Since WireLabel does not provide a one() method, we define "one" as the negation of zero.
    let wires = results.into_iter()
        .map(|b| {
             if b {
                 f.negate(&<F::Item as WireLabel>::zero(2))
             } else {
                 Ok(<F::Item as WireLabel>::zero(2))
             }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(BinaryBundle::from(Bundle::new(wires)))
}



// --------------------------------------------------------------------------
// Wire exchange functions (using the same structure as the original repo)
// --------------------------------------------------------------------------

/// The garbler's wire exchange method.
/// Encodes the garbler’s input (a slice of u16) into binary wires,
/// and receives the evaluator’s corresponding wires.
/// `num_tests` is the number of independent tests.
fn gb_set_fancy_inputs<F, E>(gb: &mut F, input: &[u16], num_tests: usize) -> SUMInputs<F::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    let garbler_wires: BinaryBundle<F::Item> =
        gb.encode_bundle(input, &vec![2; input.len()])
            .map(BinaryBundle::from)
            .unwrap();
    let evaluator_wires: BinaryBundle<F::Item> =
        gb.bin_receive(input.len() - num_tests).unwrap();
    SUMInputs {
        garbler_wires,
        evaluator_wires,
    }
}

/// The evaluator's wire exchange method.
/// Receives the garbler’s wires and encodes its own inputs.
fn ev_set_fancy_inputs<F, E>(ev: &mut F, input: &[u16], num_tests: usize) -> SUMInputs<F::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    let nwires = input.len();
    let garbler_wires: BinaryBundle<F::Item> =
        ev.bin_receive(nwires + num_tests).unwrap();
    let evaluator_wires: BinaryBundle<F::Item> =
        ev.encode_bundle(input, &vec![2; nwires]).map(BinaryBundle::from).unwrap();
    SUMInputs {
        garbler_wires,
        evaluator_wires,
    }
}

// --------------------------------------------------------------------------
// Top-level functions for multiple tests (garbler and evaluator sides)
// --------------------------------------------------------------------------

/// Multiple garbler less-than test using standard addition with carry.
/// The garbler prepares its inputs, calls fancy_is_negative,
/// and outputs the result.
pub fn multiple_gb_less_test<C>(
    rng: &mut AesRng,
    channel: &mut C,
    inputs: &[Vec<u16>], // each test's input represented as a vector of u16 bits
) -> Vec<bool>
where
    C: AbstractChannel + Clone,
{
    let num_tests = inputs.len();
    let mut gb = Garbler::<C, AesRng, OtSender, AllWire>::new(channel.clone(), rng.clone())
        .unwrap();
    let wire_inputs = inputs.iter()
                              .flatten()
                              .cloned()
                              .collect::<Vec<u16>>();
    let wires = gb_set_fancy_inputs(&mut gb, &wire_inputs, num_tests);
    let less_bundle = fancy_is_negative(&mut gb, wires, num_tests).unwrap();
    gb.outputs(less_bundle.wires()).unwrap();
    channel.flush().unwrap();
    let mut ack = [0u8; 1];
    channel.read_bytes(&mut ack).unwrap();
    let out = gb.outputs(less_bundle.wires()).unwrap()
        .expect("Outputs expected");
    out.iter().map(|&r| r == 1).collect()
}

/// Multiple evaluator less-than test using standard addition with carry.
/// The evaluator prepares its inputs, calls fancy_is_negative,
/// and returns the result.
pub fn multiple_ev_less_test<C>(
    rng: &mut AesRng,
    channel: &mut C,
    inputs: &[Vec<u16>],
) -> Vec<bool>
where
    C: AbstractChannel + Clone,
{
    let num_tests = inputs.len();
    let mut ev = Evaluator::<C, AesRng, OtReceiver, AllWire>::new(channel.clone(), rng.clone())
        .unwrap();
    let input_vec = inputs.to_vec().into_iter().flatten().collect::<Vec<u16>>();
    let wires = ev_set_fancy_inputs(&mut ev, &input_vec, num_tests);
    let less_bundle = fancy_is_negative(&mut ev, wires, num_tests).unwrap();
    let output = ev.outputs(less_bundle.wires()).unwrap().unwrap();
    let results: Vec<bool> = output.iter().map(|r| *r == 1).collect();
    channel.write_bytes(&[1u8]).unwrap();
    channel.flush().unwrap();
    results
}

// --------------------------------------------------------------------------
// (Optional) Test function (using UnixStream)

// cargo test less_than_gc_standard_add -- --nocapture

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    
    #[test]
    fn less_than_gc_standard_add() {
        // For testing purposes, we simulate inputs.
        // Each input is represented as a Vec<u16> that encodes the binary number.
        // (In practice these vectors should have length equal to the bit-width, e.g. 128.)
        let gb_inputs = vec![
            vec![0, 1, 1, 0], // example for test 0
            vec![0, 0, 0, 0], // test 1
            vec![1, 1, 1, 0], // test 2
        ];
        let ev_inputs = gb_inputs.clone(); // for simplicity, both parties use the same inputs
        
        let expected: Vec<bool> = vec![false, true, false];
        
        let (sender, receiver) = UnixStream::pair().unwrap();
        let (result_sender, result_receiver) = std::sync::mpsc::channel();
        
        let x = std::thread::spawn(move || {
            let mut rng_gb = AesRng::new();
            let reader = BufReader::new(sender.try_clone().unwrap());
            let writer = BufWriter::new(sender);
            let mut channel = Channel::new(reader, writer);
            let res = multiple_gb_less_test(&mut rng_gb, &mut channel, gb_inputs.as_slice());
            result_sender.send(res).unwrap();
        });
        
        let mut rng_ev = AesRng::new();
        let reader = BufReader::new(receiver.try_clone().unwrap());
        let writer = BufWriter::new(receiver);
        let mut channel = Channel::new(reader, writer);
        let results = multiple_ev_less_test(&mut rng_ev, &mut channel, ev_inputs.as_slice());
        
        let masks = result_receiver.recv().unwrap();
        x.join().unwrap();
        
        // For testing let’s print the output
        for (i, (mask, r)) in masks.iter().zip(results.iter()).enumerate() {
            println!("Test {}: Garbler = {}, Evaluator = {}", i, mask, r);
        }
        // In practice, you would compare with expected values.
    }
}

