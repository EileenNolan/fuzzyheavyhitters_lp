use fancy_garbling::{
    AllWire, BinaryBundle, Bundle, BinaryGadgets, Fancy, FancyArithmetic, FancyBinary, FancyInput,
    FancyReveal,
    twopac::semihonest::{Evaluator, Garbler},
    util,
};
use ocelot::{ot::AlszReceiver as OtReceiver, ot::AlszSender as OtSender};
use scuttlebutt::{AbstractChannel, AesRng, Channel};
use std::fmt::Debug;
use std::{
    io::{BufReader, BufWriter},
    os::unix::net::UnixStream,
};

use crate::lagrange::{compute_polynomials, print_polynomial, evaluate_polynomial};
use crate::field::FieldElm;
const MODULUS_64: u64 = 9223372036854775783u64;

// The constant to subtract.
// clear-text adjusted sum is computed as:
//    (gb_value + ev_value) mod MODULUS_64  - CONSTANT
const CONSTANT: u128 = 11 + 1; // this will be delta^P (need + 1 to simulate leq 0)

// A structure that contains both the garbler and the evaluator's wires.
struct SUMInputs<F> {
    pub garbler_wires: BinaryBundle<F>,
    pub evaluator_wires: BinaryBundle<F>,
}

// Extracts `num_bits` starting at position `start_bit` from a `BinaryBundle`
pub fn extract<F>(
    bundle: &BinaryBundle<F::Item>,
    start_bit: usize,
    num_bits: usize,
) -> Result<BinaryBundle<F::Item>, F::Error>
where
    F: Fancy,
    F::Item: Clone,
{
    let wires = bundle.wires();
    let end_bit = start_bit + num_bits;
    let sliced = wires[start_bit..end_bit].to_vec();
    Ok(BinaryBundle::from(Bundle::new(sliced)))
}

/// The garbler's main method. (Modified to return a u128 result.)
pub fn gb_sum<C>(rng: &mut AesRng, channel: &mut C, input: u128) -> u128
where
    C: AbstractChannel + Clone,
{
    let mut gb = Garbler::<C, AesRng, OtSender, AllWire>::new(channel.clone(), rng.clone())
        .unwrap();
    // Use raw input
    let circuit_wires = gb_set_fancy_inputs(&mut gb, input);
    
    // Compute the binary addition of garbler's and evaluator's inputs.
    let sum_bundle = gb
        .bin_addition_no_carry(&circuit_wires.garbler_wires, &circuit_wires.evaluator_wires)
        .unwrap();
    let out = gb
        .outputs(sum_bundle.wires())
        .unwrap()
        .expect("garbler should produce outputs");
    
    // Reduce the output modulo MODULUS_64.
    let raw_sum = util::u128_from_bits(&out) % (MODULUS_64 as u128);
    
    // Now subtract CONSTANT.
    let final_sum = raw_sum.wrapping_sub(CONSTANT);
    
    // Extract the MSB (most significant bit) from final_sum.
    final_sum >> 127
}

/// Processes multiple inputs in batch for the garbler.
/// 
/// For each u128 found in each vector of the slice, this function calls your existing
/// `gb_sum` function and pushes the returned bit (assumed to be 0 or 1) into a result vector.
/// 
/// The input is a slice of vectors of u128. The order is preserved (first process all entries
/// in the first vector, then the next, etc.) and a vector of result bits is returned.
pub fn multiple_gb_sum<C>(rng: &mut AesRng, channel: &mut C, inputs: &[Vec<u128>]) -> Vec<u128>
where
    C: AbstractChannel + Clone,
{
    let mut results = Vec::new();
    // Iterate over each vector in the given slice.
    for input_vec in inputs.iter() {
        // Process each value in the current vector.
        for &value in input_vec.iter() {
            // Call the single input garbler function.
            let bit = gb_sum(rng, channel, value);
            results.push(bit);
        }
    }
    results
}

/// The garbler's wire exchange method: now uses the raw input.
fn gb_set_fancy_inputs<F, E>(gb: &mut F, input: u128) -> SUMInputs<F::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    let nbits = 128;
    // Use the raw input, not adjusted.
    let garbler_wires: BinaryBundle<F::Item> = gb.bin_encode(input, nbits).unwrap();
    let evaluator_wires: BinaryBundle<F::Item> = gb.bin_receive(nbits).unwrap();
    SUMInputs {
        garbler_wires,
        evaluator_wires,
    }
}

/// The evaluator's main method.
pub fn ev_sum<C>(rng: &mut AesRng, channel: &mut C, input: u128) -> u128
where
    C: AbstractChannel + Clone,
{
    let mut ev = Evaluator::<C, AesRng, OtReceiver, AllWire>::new(channel.clone(), rng.clone())
        .unwrap();
    let circuit_wires = ev_set_fancy_inputs(&mut ev, input);

    let sum_bundle = ev
        .bin_addition_no_carry(&circuit_wires.garbler_wires, &circuit_wires.evaluator_wires)
        .unwrap();
    let out = ev
        .outputs(sum_bundle.wires())
        .unwrap()
        .expect("evaluator should produce outputs");
    
    let raw_sum = util::u128_from_bits(&out) % (MODULUS_64 as u128);
    let final_sum = raw_sum.wrapping_sub(CONSTANT);
    final_sum >> 127
}

/// Processes multiple inputs in batch for the evaluator.
/// 
/// This function behaves analogously to `multiple_gb_sum` but calls the evaluator's version (`ev_sum`)
/// on each u128 input and collects the resulting bits.
pub fn multiple_ev_sum<C>(rng: &mut AesRng, channel: &mut C, inputs: &[Vec<u128>]) -> Vec<u128>
where
    C: AbstractChannel + Clone,
{
    let mut results = Vec::new();
    for input_vec in inputs.iter() {
        for &value in input_vec.iter() {
            let bit = ev_sum(rng, channel, value);
            results.push(bit);
        }
    }
    results
}

/// The evaluator's wire exchange method.
fn ev_set_fancy_inputs<F, E>(ev: &mut F, input: u128) -> SUMInputs<F::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    let nbits = 128;
    let garbler_wires: BinaryBundle<F::Item> = ev.bin_receive(nbits).unwrap();
    let evaluator_wires: BinaryBundle<F::Item> = ev.bin_encode(input, nbits).unwrap();
    SUMInputs {
        garbler_wires,
        evaluator_wires,
    }
}

/// Computes the garbled circuit function (the underlying binary addition) to extract the MSB.
fn fancy_sum_is_negative<F>(
    f: &mut F,
    wire_inputs: SUMInputs<F::Item>,
) -> Result<BinaryBundle<F::Item>, F::Error>
where
    F: FancyReveal + Fancy + BinaryGadgets + FancyBinary + FancyArithmetic,
    F::Item: Clone,
{
    let sum = f.bin_addition_no_carry(&wire_inputs.garbler_wires, &wire_inputs.evaluator_wires)?;
    let msb = extract::<F>(&sum, 127, 1)?;
    Ok(msb)
}

/// Computes the clear-text adjusted sum as (gb_value + ev_value) mod MODULUS_64 - CONSTANT.
fn sum_in_clear(gb_value: u128, ev_value: u128) -> u128 {
    (gb_value.wrapping_add(ev_value) % (MODULUS_64 as u128))
        .wrapping_sub(CONSTANT)
}


// use clap::Parser;
// #[derive(Parser)]

// struct Cli {
//     /// The garbler's value.
//     gb_value: u128,
//     /// The evaluator's value.
//     ev_value: u128,
// }

// fn main() {
//     let cli = Cli::parse();
//     let gb_value: u128 = cli.gb_value;
//     let ev_value: u128 = cli.ev_value;

//     let (sender, receiver) = UnixStream::pair().unwrap();

//     std::thread::spawn(move || {
//         let rng_gb = AesRng::new();
//         let reader = BufReader::new(sender.try_clone().unwrap());
//         let writer = BufWriter::new(sender);
//         let mut channel = Channel::new(reader, writer);
//         gb_sum(&mut rng_gb.clone(), &mut channel, gb_value);
//     });

//     let rng_ev = AesRng::new();
//     let reader = BufReader::new(receiver.try_clone().unwrap());
//     let writer = BufWriter::new(receiver);
//     let mut channel = Channel::new(reader, writer);

//     let result = ev_sum(&mut rng_ev.clone(), &mut channel, ev_value);

//     // Compute the expected clear-text result.
//     let sum = gb_value.wrapping_sub(CONSTANT).wrapping_add(ev_value);
//     let expected_msb = (sum >> 127) & 1;

//     println!(
//         "Garbled Circuit result is : MSB((( {} - {} ) + {})) = {}",
//         gb_value, CONSTANT, ev_value, result
//     );
//     println!("Clear computed sum (adjusted) = {}", sum);
//     println!("Clear computed sum (interpreted as signed i128) = {}", sum as i128);
//     println!("Expected MSB (negative flag): {}", expected_msb);

//     assert!(
//         result == expected_msb,
//         "The garbled circuit result is incorrect. Expected MSB = {}",
//         expected_msb
//     );
// }

// To run test: cargo test test_add_leq_binary_gc -- --nocapture
#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufReader, BufWriter};

    #[test]
    fn test_add_leq_binary_gc_custom() {
        // Replace these values with the ones observed from your servers:
        let gb_value: u128 = 5212451119783949583;
        let ev_value: u128 = 4010920917070826200;
    
        println!("--- Running test_add_leq_gc_custom ---");
        println!("Garbler value: {}", gb_value);
        println!("Evaluator value: {}", ev_value);
        println!("Constant: {}", CONSTANT);
    
        let clear_sum = sum_in_clear(gb_value, ev_value);
        println!("Clear computed sum (adjusted) = {}", clear_sum);
        println!("Clear computed sum (interpreted as i128) = {}", clear_sum as i128);
        
        // Compute expected MSB.
        let expected_msb = (clear_sum >> 127) & 1;
        println!("Expected MSB (negative flag): {}", expected_msb);
    
        // Set up the channel for the MPC simulation.
        let (sender, receiver) = UnixStream::pair().unwrap();
    
        std::thread::spawn(move || {
            let rng_gb = AesRng::new();
            let reader = BufReader::new(sender.try_clone().unwrap());
            let writer = BufWriter::new(sender);
            let mut channel = Channel::new(reader, writer);
            let _ = gb_sum(&mut rng_gb.clone(), &mut channel, gb_value);
        });
    
        let rng_ev = AesRng::new();
        let reader = BufReader::new(receiver.try_clone().unwrap());
        let writer = BufWriter::new(receiver);
        let mut channel = Channel::new(reader, writer);
    
        let result = ev_sum(&mut rng_ev.clone(), &mut channel, ev_value);
        println!("Garbled Circuit computed MSB: {}", result);
    
        assert_eq!(
            result, expected_msb,
            "Expected MSB = {}, but got {}",
            expected_msb, result
        );
    }
    //add another test to fix MSB bug... tests by using same input to compute polynomial and evaluate and then
    // checks MSB bit is 1 (negative)
    #[test]
    fn test_mpc_with_polynomial_sums() {
        // For an exact-match query: client vector equals the server's dictionary.
        let q: Vec<u64> = vec![5, 10, 15, 20];
        
        // Compute polynomials for each dimension.
        let (polys_A, polys_B) = compute_polynomials(&q);
        
        // For debugging: Print the polynomials.
        println!("Polynomials for Server A:");
        for (i, poly) in polys_A.iter().enumerate() {
            let poly_name = format!("E_A{}", i);
            print_polynomial(poly, &poly_name);
        }
        println!("Polynomials for Server B:");
        for (i, poly) in polys_B.iter().enumerate() {
            let poly_name = format!("E_B{}", i);
            print_polynomial(poly, &poly_name);
        }
        
        // Evaluate each polynomial at x = FieldElm::from(q[i]) (i.e. offset 0).
        let mut gb_total: u128 = 0;
        let mut ev_total: u128 = 0;
        for i in 0..q.len() {
            let x_eval = FieldElm::from(q[i]);
            let eval_A = evaluate_polynomial(&polys_A[i], &x_eval);
            let eval_B = evaluate_polynomial(&polys_B[i], &x_eval);
            println!(
                "Dimension {}: eval_A = {}, eval_B = {}",
                i, eval_A.value, eval_B.value
            );
            // Add up the evaluations for each server modulo MODULUS_64.
            gb_total = (gb_total + eval_A.value as u128) % (MODULUS_64 as u128);
            ev_total = (ev_total + eval_B.value as u128) % (MODULUS_64 as u128);
        }
        println!("Total sum for Server A evaluations = {}", gb_total);
        println!("Total sum for Server B evaluations = {}", ev_total);
        
        // When q exactly matches, each distance should be 0,
        // so the reconstructed (combined) sum (gb_total + ev_total) mod MODULUS_64 is 0.
        // Then MPC circuit will compute:
        //   adjusted_input = (0 - CONSTANT) mod 2^128,
        // and the expected MSB (the negative flag) is computed from that.
        let clear_sum = sum_in_clear(gb_total, ev_total);
        println!("Clear computed sum (adjusted) = {}", clear_sum);
        println!(
            "Clear computed sum (interpreted as signed i128) = {}",
            clear_sum as i128
        );
        let expected_msb = (clear_sum >> 127) & 1;
        println!("Expected MSB (negative flag): {}", expected_msb);
        
        // Now simulate MPC by feeding aggregated sums to garbled circuit.
        // Set up pair of connected UnixStream channels.
        let (sender, receiver) = UnixStream::pair().unwrap();
        
        // Spawn a thread to simulate the garbler.
        std::thread::spawn(move || {
            let mut rng_gb = AesRng::new();
            let reader = BufReader::new(sender.try_clone().unwrap());
            let writer = BufWriter::new(sender);
            let mut channel = Channel::new(reader, writer);
            let result_gb = gb_sum(&mut rng_gb, &mut channel, gb_total);
            println!("Garbled Circuit (garbler) computed MSB: {}", result_gb);
            // Typically the garbler does not output a value to the leader in protocol.
            // Rely on the evaluator’s output for comparison.
        });
        
        // In the main thread, simulate the evaluator.
        let mut rng_ev = AesRng::new();
        let reader = BufReader::new(receiver.try_clone().unwrap());
        let writer = BufWriter::new(receiver);
        let mut channel = Channel::new(reader, writer);
        let result_ev = ev_sum(&mut rng_ev, &mut channel, ev_total);
        println!("Garbled Circuit (evaluator) computed MSB: {}", result_ev);
        
        // For protocol, the final output is taken from the evaluator.
        // The expected MSB is computed from the clear sum.
        assert_eq!(
            result_ev, expected_msb,
            "Expected MSB = {}, but got {}",
            expected_msb, result_ev
        );
    }
}

