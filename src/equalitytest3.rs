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
use ocelot::{ot::AlszReceiver as OtReceiver, ot::AlszSender as OtSender};
use scuttlebutt::{AbstractChannel, AesRng, Channel, SyncChannel};
use std::fmt::Debug;
use std::{
    io::{BufReader, BufWriter},
    os::unix::net::UnixStream,
};
use std::io::{Read, Write};
use std::time::Instant;
use fancy_garbling::util::RngExt;
use ocelot::ot::Sender;
use rayon::prelude::*;

/// Helper function: converts a u128 value to its 128-bit binary representation (MSB first)
/// where each bit is a u16 (0 or 1).
fn u128_to_bits_u16(x: u128) -> Vec<u16> {
    (0..128)
        .rev()
        .map(|i| (((x >> i) & 1) as u16))
        .collect()
}

/// Structure holding both the garbler’s and evaluator’s wire bundles.
pub struct EQInputs<F> {
    pub garbler_wires: BinaryBundle<F>,
    pub evaluator_wires: BinaryBundle<F>,
}

/// The garbler’s equality test.
/// Each test’s input is now a u128. We convert each to a 128‑bit binary string,
/// and then we append one extra “mask” bit (generated randomly) per test.
/// (After flattening, each test contributes 129 bits.)
pub fn multiple_gb_equality_test3<C>(
    rng: &mut AesRng,
    channel: &mut C,
    inputs: &[u128],
) -> Vec<bool>
where
    C: AbstractChannel + Clone,
{
    let num_tests = inputs.len();
    let mut mask_results = Vec::with_capacity(num_tests);
    let mut gb =
        Garbler::<C, AesRng, OtSender, AllWire>::new(channel.clone(), rng.clone())
            .expect("Garbler creation failed");

    // For each test, convert the u128 value to 128 bits and append a mask bit.
    // (Each test yields 129 bits.)
    let mut wire_inputs: Vec<u16> = Vec::with_capacity(num_tests * 129);
    for &val in inputs.iter() {
        let bits = u128_to_bits_u16(val); // 128 bits (each bit as a u16)
        let mask = rng.gen_bool(); // gen_bool takes no arguments
        mask_results.push(mask);
        let mask_bit: u16 = if mask { 1 } else { 0 };
        wire_inputs.extend(bits);
        wire_inputs.push(mask_bit);
    }

    // Set up the garbler’s inputs. Note that now the flat vector is &[u16].
    let wires = gb_set_fancy_inputs(&mut gb, &wire_inputs, num_tests);
    let eq = fancy_equality(&mut gb, wires, num_tests).expect("fancy_equality failed");
    gb.outputs(eq.wires()).expect("output failed");
    channel.flush().expect("channel flush failed");
    let mut ack = [0u8; 1];
    channel.read_bytes(&mut ack).expect("failed to read ack");
    mask_results
}

/// The garbler’s wire exchange method.
///
/// The input here is assumed to be a flat vector of bits (each represented by a u16)
/// whose length is num_tests * 129 (128 bits plus one mask bit per test).
fn gb_set_fancy_inputs<F, E>(
    gb: &mut F,
    input: &[u16],
    num_tests: usize,
) -> EQInputs<<F as FancyInput>::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    // The module for encoding expects inputs (and moduli) of type u16.
    let mod_vector: Vec<u16> = vec![2u16; input.len()];
    let garbler_wires = gb
        .encode_bundle(input, &mod_vector)
        .map(BinaryBundle::from)
        .expect("encode_bundle failed");
    // Since each test produced 129 wires (128 bits + 1 mask),
    // the evaluator’s bundle (received via OT) will have (num_tests * 129) - num_tests wires.
    let evaluator_wires = gb
        .bin_receive(input.len() - num_tests)
        .expect("bin_receive failed");
    EQInputs {
        garbler_wires,
        evaluator_wires,
    }
}

/// The evaluator’s equality test.
/// Now the evaluator’s inputs are provided as one u128 per test.
/// We convert each to its 128‑bit binary representation (as a Vec<u16>).
pub fn multiple_ev_equality_test3<C>(
    rng: &mut AesRng,
    channel: &mut C,
    inputs: &[u128],
) -> Vec<bool>
where
    C: AbstractChannel + Clone,
{
    let num_tests = inputs.len();
    let mut ev = Evaluator::<C, AesRng, OtReceiver, AllWire>::new(channel.clone(), rng.clone())
        .expect("Evaluator creation failed");
    // Convert evaluator’s inputs (one u128 per test) into bits.
    let mut eval_input_bits: Vec<u16> = Vec::with_capacity(num_tests * 128);
    for &val in inputs.iter() {
        eval_input_bits.extend(u128_to_bits_u16(val));
    }
    // Expect the garbler’s wires (same as in gb_set_fancy_inputs) to be of length num_tests * 129.
    let total_wires = num_tests * 129;
    let garbler_wires = ev
        .bin_receive(total_wires)
        .expect("bin_receive failed");
    let mod_vector: Vec<u16> = vec![2u16; eval_input_bits.len()];
    let evaluator_wires = ev
        .encode_bundle(&eval_input_bits, &mod_vector)
        .map(BinaryBundle::from)
        .expect("encode_bundle failed");

    let wire_inputs = EQInputs {
        garbler_wires,
        evaluator_wires,
    };

    let eq = fancy_equality(&mut ev, wire_inputs, num_tests).expect("fancy_equality failed");
    let output = ev
        .outputs(eq.wires())
        .expect("outputs failed")
        .expect("failed to convert outputs to u16");
    let results: Vec<bool> = output.iter().map(|&r| r == 1u16).collect();

    channel.write_bytes(&[1u8]).expect("failed to write ack");
    channel.flush().expect("flush failed");

    results
}

/// The evaluator’s wire exchange method.
///
/// Here the evaluator’s input is assumed to be the flattened binary representation
/// of the numbers (each number encoded in 128 bits, as u16’s).
fn ev_set_fancy_inputs<F, E>(
    ev: &mut F,
    input: &[u16],
    num_tests: usize,
) -> EQInputs<<F as FancyInput>::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    let total_wires = num_tests * 129; // Each test: 128 bits + 1 mask bit from the garbler.
    let garbler_wires = ev
        .bin_receive(total_wires)
        .expect("bin_receive failed");
    let mod_vector: Vec<u16> = vec![2u16; input.len()];
    let evaluator_wires = ev
        .encode_bundle(input, &mod_vector)
        .map(BinaryBundle::from)
        .expect("encode_bundle failed");
    EQInputs {
        garbler_wires,
        evaluator_wires,
    }
}

/// Extension trait for `FancyBinary` providing gadgets that operate over binary bundles.
/// (We name our new equality gadget `bin_eq_bundles2` so it won’t conflict with the inherited version.)
pub trait BinaryGadgetsExt: FancyBinary + BundleGadgets + fancy_garbling::BinaryGadgets {
    fn bin_eq_bundles2(
        &mut self,
        x: &BinaryBundle<Self::Item>,
        y: &BinaryBundle<Self::Item>,
    ) -> Result<Self::Item, Self::Error> {
        // Force a fixed bit-width for arithmetic (each test is 128 bits).
        let nbits = 128;
        assert_eq!(x.wires().len(), nbits, "Expected x bundle to contain 128 wires");
        assert_eq!(y.wires().len(), nbits, "Expected y bundle to contain 128 wires");
    
        // 1. Add the two bundles without generating an extra final carry.
        let sum_bundle = self.bin_addition_no_carry(x, y)?;
    
        // 2. Encode the modulus as a constant bundle (modulus 9223372036854775783 in 128 bits).
        let modulus_val: u128 = 9223372036854775783;
        let modulus_bundle = self.bin_constant_bundle(modulus_val, nbits)?;
    
        // 3. Compute the quotient of the sum divided by the modulus.
        let quotient_bundle = self.bin_div(&sum_bundle, &modulus_bundle)?;
    
        // 4. Multiply the quotient by the modulus.
        let prod_bundle = self.bin_mul(&quotient_bundle, &modulus_bundle)?;
    
        // 5. Subtract the product from the sum to obtain the remainder.
        let (remainder_bundle, _underflow) = self.bin_subtraction(&sum_bundle, &prod_bundle)?;
    
        // 6. Encode delta_p (12) as a constant bundle.
        let delta_p: u128 = 12;
        let delta_bundle = self.bin_constant_bundle(delta_p, nbits)?;
    
        // 7. Check whether delta_p is greater than or equal to the computed remainder.
        let result_bit = self.bin_geq(&delta_bundle, &remainder_bundle)?;
    
        Ok(result_bit)
    }
    

    fn bin_eq_bundles_shared(
        &mut self,
        x: &BinaryBundle<Self::Item>,
        y: &BinaryBundle<Self::Item>,
    ) -> Result<Self::Item, Self::Error> {
        // We expect x to have one extra wire (a mask) than y.
        assert_eq!(
            x.wires().len(),
            y.wires().len() + 1,
            "x must have one more wire than y"
        );

        let (x_wires, mask) = x.wires().split_at(x.wires().len() - 1);
        let mask = &mask[0]; // the last wire is the mask

        let eq_result = self.bin_eq_bundles2(&BinaryBundle::new(x_wires.to_vec()), y)?;

        // Obscure the result with the mask.
        self.xor(&eq_result, mask)
    }

    fn multi_bin_eq_bundles_shared(
        &mut self,
        x: &BinaryBundle<Self::Item>,
        y: &BinaryBundle<Self::Item>,
        num_tests: usize,
    ) -> Result<BinaryBundle<Self::Item>, Self::Error> {
        // x should consist, for each test, of (bits of y + one mask bit).
        assert_eq!(
            x.wires().len(),
            y.wires().len() + num_tests,
            "each string in x must have one extra mask bit"
        );
        assert_eq!(y.wires().len() % num_tests, 0);

        let string_len = y.wires().len() / num_tests;
        let mut results = Vec::with_capacity(num_tests);

        for i in 0..num_tests {
            let x_start = i * (string_len + 1);
            let y_start = i * string_len;
            let eq_result = self.bin_eq_bundles2(
                &BinaryBundle::new(x.wires()[x_start..(x_start + string_len)].to_vec()),
                &BinaryBundle::new(y.wires()[y_start..(y_start + string_len)].to_vec()),
            )?;
            let masked_result = self.xor(&eq_result, &x.wires()[x_start + string_len])?;
            results.push(masked_result);
        }
        Ok(BinaryBundle::new(results))
    }
}

/// Implement BinaryGadgetsExt for Garbler.
impl<C, R, S, W> BinaryGadgetsExt for fancy_garbling::twopac::semihonest::Garbler<C, R, S, W>
where
    Self: FancyBinary + BundleGadgets,
{
}

/// Implement BinaryGadgetsExt for Evaluator.
impl<C, R, S, W> BinaryGadgetsExt for fancy_garbling::twopac::semihonest::Evaluator<C, R, S, W>
where
    Self: FancyBinary + BundleGadgets,
{
}

/// Fancy equality test gadget using garbled circuits.
/// It takes in the garbler’s and evaluator’s wire bundles and returns a bundle of output bits.
fn fancy_equality<F>(
    f: &mut F,
    wire_inputs: EQInputs<F::Item>,
    num_tests: usize,
) -> Result<BinaryBundle<F::Item>, F::Error>
where
    F: FancyReveal + Fancy + BinaryGadgetsExt + FancyBinary + FancyArithmetic,
{
    let equality_bits = f.multi_bin_eq_bundles_shared(
        &wire_inputs.garbler_wires,
        &wire_inputs.evaluator_wires,
        num_tests,
    )?;
    Ok(equality_bits)
}

#[test]
fn eq_gc() {
    let gb_value = vec![vec![0,1,1,0], vec![0,0,0,0], vec![1,1,1,0]];
    let ev_value = vec![vec![0,1,1,0], vec![0,0,0,0], vec![1,1,1,0]];
    let expected = gb_value.iter().enumerate().map(|(i, x)| *x == ev_value[i]).collect::<Vec<bool>>();

    let (sender, receiver) = UnixStream::pair().unwrap();

    let (result_sender, result_receiver) = std::sync::mpsc::channel();

    let x = std::thread::spawn(move || {
        let rng_gb = AesRng::new();
        let reader = BufReader::new(sender.try_clone().unwrap());
        let writer = BufWriter::new(sender);
        let mut channel = Channel::new(reader, writer);
        let masks = multiple_gb_equality_test3(&mut rng_gb.clone(), &mut channel, gb_value.as_slice());
        result_sender.send(masks).unwrap();
    });

    let rng_ev = AesRng::new();
    let reader = BufReader::new(receiver.try_clone().unwrap());
    let writer = BufWriter::new(receiver);
    let mut channel = Channel::new(reader, writer);

    let results = multiple_ev_equality_test3(&mut rng_ev.clone(), &mut channel, ev_value.as_slice());

    let masks = result_receiver.recv().unwrap();
    x.join().unwrap();

    assert_eq!(
        masks.len(),
        results.len(),
        "Masks and results should have the same length"
    );

    for i in 0..results.len() {
        assert_eq!(
            (masks[i] ^ results[i]) as u16,
            expected[i] as u16,
            "The garbled circuit result is incorrect for index {} and should be {}",
            i,
            expected[i]
        );
    }
}

