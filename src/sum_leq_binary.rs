// use fancy_garbling::{
//     AllWire, BinaryBundle, Bundle, BinaryGadgets, Fancy, FancyArithmetic, FancyBinary, FancyInput,
//     FancyReveal,
//     twopac::semihonest::{Evaluator, Garbler},
//     util,
// };
// use ocelot::{ot::AlszReceiver as OtReceiver, ot::AlszSender as OtSender};
// use scuttlebutt::{AbstractChannel, AesRng, Channel};
// use std::fmt::Debug;
// use std::{
//     io::{BufReader, BufWriter},
//     os::unix::net::UnixStream,
// };

// use crate::lagrange::{compute_polynomials, print_polynomial, evaluate_polynomial};
// use crate::field::FieldElm;
// const MODULUS_64: u64 = 9223372036854775783u64;

// // The constant to subtract.
// // clear-text adjusted sum is computed as:
// //    (gb_value + ev_value) mod MODULUS_64  - CONSTANT
// const CONSTANT: u128 = 11 + 1; // this will be delta^P (need + 1 to simulate leq 0)

// // A structure that contains both the garbler and the evaluator's wires.
// struct SUMInputs<F> {
//     pub garbler_wires: BinaryBundle<F>,
//     pub evaluator_wires: BinaryBundle<F>,
// }

// // Extracts `num_bits` starting at position `start_bit` from a `BinaryBundle`
// pub fn extract<F>(
//     bundle: &BinaryBundle<F::Item>,
//     start_bit: usize,
//     num_bits: usize,
// ) -> Result<BinaryBundle<F::Item>, F::Error>
// where
//     F: Fancy,
//     F::Item: Clone,
// {
//     let wires = bundle.wires();
//     let end_bit = start_bit + num_bits;
//     let sliced = wires[start_bit..end_bit].to_vec();
//     Ok(BinaryBundle::from(Bundle::new(sliced)))
// }

// /// The garbler's main method. (Modified to return a u128 result.)
// pub fn gb_sum<C>(rng: &mut AesRng, channel: &mut C, input: u128) -> u128
// where
//     C: AbstractChannel + Clone,
// {
//     let mut gb = Garbler::<C, AesRng, OtSender, AllWire>::new(channel.clone(), rng.clone())
//         .unwrap();
//     // Use raw input
//     let circuit_wires = gb_set_fancy_inputs(&mut gb, input);
    
//     // Compute the binary addition of garbler's and evaluator's inputs.
//     let sum_bundle = gb
//         .bin_addition_no_carry(&circuit_wires.garbler_wires, &circuit_wires.evaluator_wires)
//         .unwrap();
//     let out = gb
//         .outputs(sum_bundle.wires())
//         .unwrap()
//         .expect("garbler should produce outputs");
    
//     // Reduce the output modulo MODULUS_64.
//     let raw_sum = util::u128_from_bits(&out) % (MODULUS_64 as u128);
    
//     // Now subtract CONSTANT.
//     let final_sum = raw_sum.wrapping_sub(CONSTANT);
    
//     // Extract the MSB (most significant bit) from final_sum.
//     final_sum >> 127
// }

// /// Processes multiple inputs in batch for the garbler.
// /// 
// /// For each u128 found in each vector of the slice, this function calls your existing
// /// `gb_sum` function and pushes the returned bit (assumed to be 0 or 1) into a result vector.
// /// 
// /// The input is a slice of vectors of u128. The order is preserved (first process all entries
// /// in the first vector, then the next, etc.) and a vector of result bits is returned.
// pub fn multiple_gb_sum<C>(rng: &mut AesRng, channel: &mut C, inputs: &[u128]) -> Vec<u128>
// where
//     C: AbstractChannel + Clone,
// {
//     let mut results = Vec::new();
//     // Iterate directly over each u128 value in the slice.
//     for &value in inputs.iter() {
//         // Call the single input garbler function.
//         let bit = gb_sum(rng, channel, value);
//         results.push(bit);
//     }
//     results
// }

// /// The garbler's wire exchange method: now uses the raw input.
// fn gb_set_fancy_inputs<F, E>(gb: &mut F, input: u128) -> SUMInputs<F::Item>
// where
//     F: FancyInput<Item = AllWire, Error = E>,
//     E: Debug,
// {
//     let nbits = 128;
//     // Use the raw input, not adjusted.
//     let garbler_wires: BinaryBundle<F::Item> = gb.bin_encode(input, nbits).unwrap();
//     let evaluator_wires: BinaryBundle<F::Item> = gb.bin_receive(nbits).unwrap();
//     SUMInputs {
//         garbler_wires,
//         evaluator_wires,
//     }
// }

// /// The evaluator's main method.
// pub fn ev_sum<C>(rng: &mut AesRng, channel: &mut C, input: u128) -> u128
// where
//     C: AbstractChannel + Clone,
// {
//     let mut ev = Evaluator::<C, AesRng, OtReceiver, AllWire>::new(channel.clone(), rng.clone())
//         .unwrap();
//     let circuit_wires = ev_set_fancy_inputs(&mut ev, input);

//     let sum_bundle = ev
//         .bin_addition_no_carry(&circuit_wires.garbler_wires, &circuit_wires.evaluator_wires)
//         .unwrap();
//     let out = ev
//         .outputs(sum_bundle.wires())
//         .unwrap()
//         .expect("evaluator should produce outputs");
    
//     let raw_sum = util::u128_from_bits(&out) % (MODULUS_64 as u128);
//     let final_sum = raw_sum.wrapping_sub(CONSTANT);
//     final_sum >> 127
// }

// /// Processes multiple inputs in batch for the evaluator.
// /// 
// /// This function behaves analogously to `multiple_gb_sum` but calls the evaluator's version (`ev_sum`)
// /// on each u128 input and collects the resulting bits.
// pub fn multiple_ev_sum<C>(rng: &mut AesRng, channel: &mut C, inputs: &[u128]) -> Vec<u128>
// where
//     C: AbstractChannel + Clone,
// {
//     let mut results = Vec::new();
//     // Iterate directly over each u128 value in the slice.
//     for &value in inputs.iter() {
//         let bit = ev_sum(rng, channel, value);
//         results.push(bit);
//     }
//     results
// }

// /// The evaluator's wire exchange method.
// fn ev_set_fancy_inputs<F, E>(ev: &mut F, input: u128) -> SUMInputs<F::Item>
// where
//     F: FancyInput<Item = AllWire, Error = E>,
//     E: Debug,
// {
//     let nbits = 128;
//     let garbler_wires: BinaryBundle<F::Item> = ev.bin_receive(nbits).unwrap();
//     let evaluator_wires: BinaryBundle<F::Item> = ev.bin_encode(input, nbits).unwrap();
//     SUMInputs {
//         garbler_wires,
//         evaluator_wires,
//     }
// }

// /// Computes the garbled circuit function (the underlying binary addition) to extract the MSB.
// fn fancy_sum_is_negative<F>(
//     f: &mut F,
//     wire_inputs: SUMInputs<F::Item>,
// ) -> Result<BinaryBundle<F::Item>, F::Error>
// where
//     F: FancyReveal + Fancy + BinaryGadgets + FancyBinary + FancyArithmetic,
//     F::Item: Clone,
// {
//     let sum = f.bin_addition_no_carry(&wire_inputs.garbler_wires, &wire_inputs.evaluator_wires)?;
//     let msb = extract::<F>(&sum, 127, 1)?;
//     Ok(msb)
// }

// /// Computes the clear-text adjusted sum as (gb_value + ev_value) mod MODULUS_64 - CONSTANT.
// fn sum_in_clear(gb_value: u128, ev_value: u128) -> u128 {
//     (gb_value.wrapping_add(ev_value) % (MODULUS_64 as u128))
//         .wrapping_sub(CONSTANT)
// }


// // use clap::Parser;
// // #[derive(Parser)]

// // struct Cli {
// //     /// The garbler's value.
// //     gb_value: u128,
// //     /// The evaluator's value.
// //     ev_value: u128,
// // }

// // fn main() {
// //     let cli = Cli::parse();
// //     let gb_value: u128 = cli.gb_value;
// //     let ev_value: u128 = cli.ev_value;

// //     let (sender, receiver) = UnixStream::pair().unwrap();

// //     std::thread::spawn(move || {
// //         let rng_gb = AesRng::new();
// //         let reader = BufReader::new(sender.try_clone().unwrap());
// //         let writer = BufWriter::new(sender);
// //         let mut channel = Channel::new(reader, writer);
// //         gb_sum(&mut rng_gb.clone(), &mut channel, gb_value);
// //     });

// //     let rng_ev = AesRng::new();
// //     let reader = BufReader::new(receiver.try_clone().unwrap());
// //     let writer = BufWriter::new(receiver);
// //     let mut channel = Channel::new(reader, writer);

// //     let result = ev_sum(&mut rng_ev.clone(), &mut channel, ev_value);

// //     // Compute the expected clear-text result.
// //     let sum = gb_value.wrapping_sub(CONSTANT).wrapping_add(ev_value);
// //     let expected_msb = (sum >> 127) & 1;

// //     println!(
// //         "Garbled Circuit result is : MSB((( {} - {} ) + {})) = {}",
// //         gb_value, CONSTANT, ev_value, result
// //     );
// //     println!("Clear computed sum (adjusted) = {}", sum);
// //     println!("Clear computed sum (interpreted as signed i128) = {}", sum as i128);
// //     println!("Expected MSB (negative flag): {}", expected_msb);

// //     assert!(
// //         result == expected_msb,
// //         "The garbled circuit result is incorrect. Expected MSB = {}",
// //         expected_msb
// //     );
// // }

// // To run test: cargo test test_add_leq_binary_gc -- --nocapture
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use std::io::{BufReader, BufWriter};

//     #[test]
//     fn test_add_leq_binary_gc_custom() {
//         // Replace these values with the ones observed from your servers:
//         let gb_value: u128 = 5212451119783949583;
//         let ev_value: u128 = 4010920917070826200;
    
//         println!("--- Running test_add_leq_gc_custom ---");
//         println!("Garbler value: {}", gb_value);
//         println!("Evaluator value: {}", ev_value);
//         println!("Constant: {}", CONSTANT);
    
//         let clear_sum = sum_in_clear(gb_value, ev_value);
//         println!("Clear computed sum (adjusted) = {}", clear_sum);
//         println!("Clear computed sum (interpreted as i128) = {}", clear_sum as i128);
        
//         // Compute expected MSB.
//         let expected_msb = (clear_sum >> 127) & 1;
//         println!("Expected MSB (negative flag): {}", expected_msb);
    
//         // Set up the channel for the MPC simulation.
//         let (sender, receiver) = UnixStream::pair().unwrap();
    
//         std::thread::spawn(move || {
//             let rng_gb = AesRng::new();
//             let reader = BufReader::new(sender.try_clone().unwrap());
//             let writer = BufWriter::new(sender);
//             let mut channel = Channel::new(reader, writer);
//             let _ = gb_sum(&mut rng_gb.clone(), &mut channel, gb_value);
//         });
    
//         let rng_ev = AesRng::new();
//         let reader = BufReader::new(receiver.try_clone().unwrap());
//         let writer = BufWriter::new(receiver);
//         let mut channel = Channel::new(reader, writer);
    
//         let result = ev_sum(&mut rng_ev.clone(), &mut channel, ev_value);
//         println!("Garbled Circuit computed MSB: {}", result);
    
//         assert_eq!(
//             result, expected_msb,
//             "Expected MSB = {}, but got {}",
//             expected_msb, result
//         );
//     }
//     //add another test to fix MSB bug... tests by using same input to compute polynomial and evaluate and then
//     // checks MSB bit is 1 (negative)
//     #[test]
//     fn test_mpc_with_polynomial_sums() {
//         // For an exact-match query: client vector equals the server's dictionary.
//         let q: Vec<u64> = vec![5, 10, 15, 20];
        
//         // Compute polynomials for each dimension.
//         let (polys_A, polys_B) = compute_polynomials(&q);
        
//         // For debugging: Print the polynomials.
//         println!("Polynomials for Server A:");
//         for (i, poly) in polys_A.iter().enumerate() {
//             let poly_name = format!("E_A{}", i);
//             print_polynomial(poly, &poly_name);
//         }
//         println!("Polynomials for Server B:");
//         for (i, poly) in polys_B.iter().enumerate() {
//             let poly_name = format!("E_B{}", i);
//             print_polynomial(poly, &poly_name);
//         }
        
//         // Evaluate each polynomial at x = FieldElm::from(q[i]) (i.e. offset 0).
//         let mut gb_total: u128 = 0;
//         let mut ev_total: u128 = 0;
//         for i in 0..q.len() {
//             let x_eval = FieldElm::from(q[i]);
//             let eval_A = evaluate_polynomial(&polys_A[i], &x_eval);
//             let eval_B = evaluate_polynomial(&polys_B[i], &x_eval);
//             println!(
//                 "Dimension {}: eval_A = {}, eval_B = {}",
//                 i, eval_A.value, eval_B.value
//             );
//             // Add up the evaluations for each server modulo MODULUS_64.
//             gb_total = (gb_total + eval_A.value as u128) % (MODULUS_64 as u128);
//             ev_total = (ev_total + eval_B.value as u128) % (MODULUS_64 as u128);
//         }
//         println!("Total sum for Server A evaluations = {}", gb_total);
//         println!("Total sum for Server B evaluations = {}", ev_total);
        
//         // When q exactly matches, each distance should be 0,
//         // so the reconstructed (combined) sum (gb_total + ev_total) mod MODULUS_64 is 0.
//         // Then MPC circuit will compute:
//         //   adjusted_input = (0 - CONSTANT) mod 2^128,
//         // and the expected MSB (the negative flag) is computed from that.
//         let clear_sum = sum_in_clear(gb_total, ev_total);
//         println!("Clear computed sum (adjusted) = {}", clear_sum);
//         println!(
//             "Clear computed sum (interpreted as signed i128) = {}",
//             clear_sum as i128
//         );
//         let expected_msb = (clear_sum >> 127) & 1;
//         println!("Expected MSB (negative flag): {}", expected_msb);
        
//         // Now simulate MPC by feeding aggregated sums to garbled circuit.
//         // Set up pair of connected UnixStream channels.
//         let (sender, receiver) = UnixStream::pair().unwrap();
        
//         // Spawn a thread to simulate the garbler.
//         std::thread::spawn(move || {
//             let mut rng_gb = AesRng::new();
//             let reader = BufReader::new(sender.try_clone().unwrap());
//             let writer = BufWriter::new(sender);
//             let mut channel = Channel::new(reader, writer);
//             let result_gb = gb_sum(&mut rng_gb, &mut channel, gb_total);
//             println!("Garbled Circuit (garbler) computed MSB: {}", result_gb);
//             // Typically the garbler does not output a value to the leader in protocol.
//             // Rely on the evaluator’s output for comparison.
//         });
        
//         // In the main thread, simulate the evaluator.
//         let mut rng_ev = AesRng::new();
//         let reader = BufReader::new(receiver.try_clone().unwrap());
//         let writer = BufWriter::new(receiver);
//         let mut channel = Channel::new(reader, writer);
//         let result_ev = ev_sum(&mut rng_ev, &mut channel, ev_total);
//         println!("Garbled Circuit (evaluator) computed MSB: {}", result_ev);
        
//         // For protocol, the final output is taken from the evaluator.
//         // The expected MSB is computed from the clear sum.
//         assert_eq!(
//             result_ev, expected_msb,
//             "Expected MSB = {}, but got {}",
//             expected_msb, result_ev
//         );
//     }
// }

// sum_binary_leq.rs

// sum_binary_leq.rs

// sum_binary_leq.rs

// sum_binary_leq.rs

// use fancy_garbling::{
//     twopac::semihonest::{Evaluator, Garbler},
//     util, AllWire, BinaryBundle, BundleGadgets, Fancy, FancyArithmetic, FancyBinary, FancyInput,
//     FancyReveal,
// };
// use ocelot::{ot::AlszReceiver as OtReceiver, ot::AlszSender as OtSender};
// use scuttlebutt::{AbstractChannel, AesRng, Channel, SyncChannel};
// use std::{
//     io::{BufReader, BufWriter},
//     os::unix::net::UnixStream,
// };
// use std::io::{Read, Write};
// use std::time::Instant;
// use fancy_garbling::util::RngExt;
// use ocelot::ot::Sender;
// use rayon::prelude::*;

// /// A structure that contains both the garbler's and evaluator's wires.
// struct EQInputs<F> {
//     pub garbler_wires: BinaryBundle<F>,
//     pub evaluator_wires: BinaryBundle<F>,
// }

// /// For the garbler: set up the fancy inputs.
// pub fn gb_set_fancy_inputs<F, E>(
//     gb: &mut F,
//     input: &[u16],
//     num_tests: usize,
// ) -> EQInputs<F::Item>
// where
//     F: FancyInput<Item = AllWire, Error = E>,
//     E: std::fmt::Debug,
// {
//     let garbler_wires: BinaryBundle<F::Item> = gb
//         .encode_bundle(&input, &vec![2; input.len()])
//         .map(BinaryBundle::from)
//         .unwrap();
//     let evaluator_wires: BinaryBundle<F::Item> =
//         gb.bin_receive(input.len() - num_tests).unwrap();
//     EQInputs {
//         garbler_wires,
//         evaluator_wires,
//     }
// }

// /// For the evaluator: set up the fancy inputs.
// pub fn ev_set_fancy_inputs<F, E>(
//     ev: &mut F,
//     input: &[u16],
//     num_tests: usize,
// ) -> EQInputs<F::Item>
// where
//     F: FancyInput<Item = AllWire, Error = E>,
//     E: std::fmt::Debug,
// {
//     let nwires = input.len();
//     let garbler_wires: BinaryBundle<F::Item> =
//         ev.bin_receive(nwires + num_tests).unwrap();
//     let evaluator_wires: BinaryBundle<F::Item> = ev
//         .encode_bundle(input, &vec![2; nwires])
//         .map(BinaryBundle::from)
//         .unwrap();
//     EQInputs {
//         garbler_wires,
//         evaluator_wires,
//     }
// }

// /// Extension trait that provides additional arithmetic gadgets.
// pub trait FancyArithmeticExt: FancyArithmetic {
//     /// Returns the arithmetic value (represented by a wire) that is the input modulo `modulus`.
//     fn modulo(&mut self, input: &Self::Item, modulus: u16) -> Result<Self::Item, Self::Error>;
//     /// Extracts the most significant bit from an `input` known to have `bit_length` bits.
//     fn msb(&mut self, input: &Self::Item, bit_length: usize) -> Result<Self::Item, Self::Error>;
// }

// impl<T: FancyArithmetic> FancyArithmeticExt for T {
//     fn modulo(&mut self, _input: &Self::Item, _modulus: u16) -> Result<Self::Item, Self::Error> {
//          unimplemented!("modulo gadget is not implemented")
//     }
//     fn msb(&mut self, _input: &Self::Item, _bit_length: usize) -> Result<Self::Item, Self::Error> {
//          unimplemented!("msb gadget is not implemented")
//     }
// }

// /// Extension trait for `FancyBinary` providing extra binary (bundle) gadgets.
// pub trait BinaryGadgets: FancyBinary + BundleGadgets + FancyArithmetic {
//     fn bin_eq_bundles(
//         &mut self,
//         x: &BinaryBundle<Self::Item>,
//         y: &BinaryBundle<Self::Item>,
//     ) -> Result<Self::Item, Self::Error> {
//         let zs = x
//             .wires()
//             .iter()
//             .zip(y.wires().iter())
//             .map(|(x_bit, y_bit)| {
//                 let xy = self.xor(x_bit, y_bit)?;
//                 self.negate(&xy)
//             })
//             .collect::<Result<Vec<Self::Item>, Self::Error>>()?;
//         self.and_many(&zs)
//     }

//     fn bin_eq_bundles_shared(
//         &mut self,
//         x: &BinaryBundle<Self::Item>,
//         y: &BinaryBundle<Self::Item>,
//     ) -> Result<Self::Item, Self::Error> {
//         assert_eq!(
//             x.wires().len(),
//             y.wires().len() + 1,
//             "x must have one more wire than y"
//         );
//         let (x_wires, mask) = x.wires().split_at(x.wires().len() - 1);
//         let mask = &mask[0]; // Last wire is the mask
//         let eq_result = self.bin_eq_bundles(&BinaryBundle::new(x_wires.to_vec()), y)?;
//         self.xor(&eq_result, mask)
//     }

//     fn multi_bin_eq_bundles_shared(
//         &mut self,
//         x: &BinaryBundle<Self::Item>,
//         y: &BinaryBundle<Self::Item>,
//         num_tests: usize,
//     ) -> Result<BinaryBundle<Self::Item>, Self::Error> {
//         assert_eq!(
//             x.wires().len(),
//             y.wires().len() + num_tests,
//             "each string in x must have one extra mask bit"
//         );
//         assert_eq!(y.wires().len() % num_tests, 0);
//         let string_len = y.wires().len() / num_tests;
//         let mut results = Vec::with_capacity(num_tests);
//         for i in 0..num_tests {
//             let x_start = i * (string_len + 1);
//             let y_start = i * string_len;
//             let eq_result = self.bin_eq_bundles(
//                 &BinaryBundle::new(x.wires()[x_start..x_start + string_len].to_vec()),
//                 &BinaryBundle::new(y.wires()[y_start..y_start + string_len].to_vec()),
//             )?;
//             let masked_result = self.xor(&eq_result, &x.wires()[x_start + string_len])?;
//             results.push(masked_result);
//         }
//         Ok(BinaryBundle::new(results))
//     }

//     /// NEW: Compute bitwise addition with no carry.
//     /// Assumes MSB-first ordering and computes the bitwise XOR of corresponding bits,
//     /// then "packs" the resulting bits into an arithmetic value.
//     fn bin_addition_no_carry(
//         &mut self,
//         x: &[Self::Item],
//         y: &[Self::Item],
//     ) -> Result<Self::Item, Self::Error> {
//         assert_eq!(x.len(), y.len(), "Input slices must have the same length");
//         let mut sum = self.constant(0, 2)?; // Supply two arguments: initial value and field (here, 2)
//         let n = x.len();
//         for (j, (x_bit, y_bit)) in x.iter().zip(y.iter()).enumerate() {
//             let bit_sum = self.xor(x_bit, y_bit)?;
//             let weight: u128 = 1u128 << (n - 1 - j);
//             let weighted = Self::mul_const_helper(self, &bit_sum, weight as u16)?;
//             sum = self.add(&sum, &weighted)?;
//         }
//         Ok(sum)
//     }

//     /// Helper: multiply a wire by a constant.
//     fn mul_const_helper(
//         &mut self,
//         a: &Self::Item,
//         c: u16,
//     ) -> Result<Self::Item, Self::Error> {
//         // Lift the constant into a wire.
//         let c_wire = self.constant(c, 2)?; // Supply two arguments here as well
//         self.mul(a, &c_wire)
//     }
// }

// /// Implement BinaryGadgets for the Garbler.
// impl<C, R, S, W> BinaryGadgets
//     for fancy_garbling::twopac::semihonest::Garbler<C, R, S, W>
// where
//     Self: FancyBinary + BundleGadgets + FancyArithmetic,
// {
// }

// /// Implement BinaryGadgets for the Evaluator.
// impl<C, R, S, W> BinaryGadgets
//     for fancy_garbling::twopac::semihonest::Evaluator<C, R, S, W>
// where
//     Self: FancyBinary + BundleGadgets + FancyArithmetic,
// {
// };

// /// Custom gadget that implements your requested logic:
// /// - Computes "addition with no carry" via bin_addition_no_carry (assuming MSB-first ordering),
// /// - Takes the result modulo 64 using the modulo gadget,
// /// - Subtracts a constant (here, 10) from that result,
// /// - Extracts the most-significant bit (using the msb gadget),
// /// - And finally re-masks the output.
// pub fn fancy_custom<F>(
//     f: &mut F,
//     wire_inputs: EQInputs<F::Item>,
//     num_tests: usize,
// ) -> Result<BinaryBundle<F::Item>, F::Error>
// where
//     F: FancyReveal + Fancy + BinaryGadgets + FancyBinary + FancyArithmetic + FancyArithmeticExt,
// {
//     assert_eq!(
//         wire_inputs.garbler_wires.wires().len(),
//         wire_inputs.evaluator_wires.wires().len() + num_tests,
//         "each string in garbler wires must have one extra mask bit"
//     );
//     let string_len = wire_inputs.evaluator_wires.wires().len() / num_tests;
//     let mut results = Vec::with_capacity(num_tests);
//     for i in 0..num_tests {
//         let x_start = i * (string_len + 1);
//         let y_start = i * string_len;
//         let x_segment = &wire_inputs.garbler_wires.wires()[x_start..x_start + string_len];
//         let mask = &wire_inputs.garbler_wires.wires()[x_start + string_len];
//         let y_segment = &wire_inputs.evaluator_wires.wires()[y_start..y_start + string_len];

//         // Use bin_addition_no_carry to compute the bitwise XOR (as addition with no carry).
//         let sum_wire = f.bin_addition_no_carry(x_segment, y_segment)?;

//         // Compute the result modulo 64.
//         let mod_wire = f.modulo(&sum_wire, 64)?;

//         // Subtract a constant (first lift the constant to a wire).
//         const CONSTANT: u16 = 10;
//         let const_wire = f.constant(CONSTANT, 2)?;
//         let diff_wire = f.sub(&mod_wire, &const_wire)?;

//         // Extract the most-significant bit from the (6-bit) result.
//         let msb_wire = f.msb(&diff_wire, 6)?;

//         // Remove the mask by XORing with the garbler's mask bit.
//         let masked_result = f.xor(&msb_wire, mask)?;
//         results.push(masked_result);
//     }
//     Ok(BinaryBundle::new(results))
// }

// /// Garbler–side function that uses fancy_custom.
// pub fn multiple_gb_custom_test<C>(
//     rng: &mut AesRng,
//     channel: &mut C,
//     inputs: &[Vec<u16>],
// ) -> Vec<bool>
// where
//     C: AbstractChannel + Clone,
// {
//     let num_tests = inputs.len();
//     let mut results = Vec::with_capacity(num_tests);
//     let mut gb =
//         Garbler::<C, AesRng, OtSender, AllWire>::new(channel.clone(), rng.clone()).unwrap();

//     // Obscure inputs with a random mask.
//     let masked_inputs = inputs
//         .iter()
//         .map(|input| {
//             let mask = rng.clone().gen_bool();
//             results.push(mask);
//             [input.as_slice(), &[mask as u16]].concat()
//         })
//         .collect::<Vec<Vec<u16>>>();

//     let wire_inputs = masked_inputs.into_iter().flatten().collect::<Vec<u16>>();
//     let wires = gb_set_fancy_inputs(&mut gb, wire_inputs.as_slice(), inputs.len());
//     let custom = fancy_custom(&mut gb, wires, num_tests).unwrap();
//     gb.outputs(custom.wires()).unwrap();
//     channel.flush().unwrap();
//     // Read acknowledgement from the evaluator.
//     let mut ack = [0u8; 1];
//     channel.read_bytes(&mut ack).unwrap();
//     results
// }

// /// Evaluator–side function that uses fancy_custom.
// pub fn multiple_ev_custom_test<C>(
//     rng: &mut AesRng,
//     channel: &mut C,
//     inputs: &[Vec<u16>],
// ) -> Vec<bool>
// where
//     C: AbstractChannel + Clone,
// {
//     let num_tests = inputs.len();
//     let mut ev =
//         Evaluator::<C, AesRng, OtReceiver, AllWire>::new(channel.clone(), rng.clone()).unwrap();
//     let input_vec = inputs.to_vec().into_iter().flatten().collect::<Vec<u16>>();
//     let ev_in = input_vec.as_slice();
//     let wires = ev_set_fancy_inputs(&mut ev, &ev_in, num_tests);
//     let custom = fancy_custom(&mut ev, wires, num_tests).unwrap();
//     let output = ev.outputs(custom.wires()).unwrap().unwrap();
//     let results = output.iter().map(|r| *r == 1).collect();
//     channel.write_bytes(&[1u8]).unwrap();
//     channel.flush().unwrap();
//     results
// }

// #[test]
// fn custom_gc() {
//     // Example tests using 4-bit inputs.
//     // For test 0:
//     //   Garbler: [0, 1, 1, 0] and Evaluator: [1, 0, 0, 1]
//     //   bin_addition_no_carry yields bitwise XOR: [1, 1, 1, 1] → 15.
//     //   15 mod 64 = 15; 15 - 10 = 5; in 6-bit binary 5 is 000101, whose MSB is 0.
//     let gb_value = vec![vec![0, 1, 1, 0], vec![0, 0, 0, 0], vec![1, 1, 1, 0]];
//     let ev_value = vec![vec![1, 0, 0, 1], vec![0, 0, 0, 0], vec![0, 1, 0, 0]];
//     let expected = vec![false, true, false];
//     let (sender, receiver) = UnixStream::pair().unwrap();
//     let (result_sender, result_receiver) = std::sync::mpsc::channel();

//     let x = std::thread::spawn(move || {
//         let rng_gb = AesRng::new();
//         let reader = BufReader::new(sender.try_clone().unwrap());
//         let writer = BufWriter::new(sender);
//         let mut channel = Channel::new(reader, writer);
//         let masks =
//             multiple_gb_custom_test(&mut rng_gb.clone(), &mut channel, gb_value.as_slice());
//         result_sender.send(masks).unwrap();
//     });

//     let rng_ev = AesRng::new();
//     let reader = BufReader::new(receiver.try_clone().unwrap());
//     let writer = BufWriter::new(receiver);
//     let mut channel = Channel::new(reader, writer);
//     let results =
//         multiple_ev_custom_test(&mut rng_ev.clone(), &mut channel, ev_value.as_slice());
//     let masks = result_receiver.recv().unwrap();
//     x.join().unwrap();

//     assert_eq!(masks.len(), results.len(), "Mismatch in result length");
//     for i in 0..results.len() {
//         // Unmask the output by XORing the two parts and compare to expected.
//         assert_eq!(
//             (masks[i] ^ results[i]) as u16,
//             expected[i] as u16,
//             "Result mismatch at index {}: expected {}",
//             i,
//             expected[i]
//         );
//     }
// }
