// use fancy_garbling::{twopac::semihonest::{Evaluator, Garbler}, util, AllWire, BinaryBundle, BundleGadgets, CrtBundle, Fancy, FancyArithmetic, ArithmeticBundleGadgets, FancyBinary, FancyInput, FancyReveal};

// use ocelot::{ot::AlszReceiver as OtReceiver, ot::AlszSender as OtSender};
// use scuttlebutt::{AbstractChannel, AesRng, Channel, SyncChannel};

// use std::fmt::Debug;

// use std::{
//     io::{BufReader, BufWriter},
//     os::unix::net::UnixStream,
// };
// use std::io::{Read, Write};
// use std::time::Instant;
// use fancy_garbling::util::RngExt;
// use ocelot::ot::Sender;
// use rayon::prelude::*;

// /// A structure that contains both the garbler and the evaluators
// /// wires. This structure simplifies the API of the garbled circuit.

// // changed so we can use CRT Bundles instead of Binary Bundles
// struct EQInputs<F> {
//     pub garbler_wires: CrtBundle<F>,
//     pub evaluator_wires: CrtBundle<F>,
// }

// //changed to use CRT bundles and input vector of u128 values
// pub fn multiple_gb_leq_test<C>(
//     rng: &mut AesRng,
//     channel: &mut C,
//     inputs: &[Vec<u128>]
// ) -> Vec<bool>
// where
//     C: AbstractChannel + Clone,
// {
//     let num_tests = inputs.len();
//     let mut results = Vec::with_capacity(num_tests);
//     let mut gb = Garbler::<C, AesRng, OtSender, AllWire>::new(channel.clone(), rng.clone()).unwrap();

//     let masked_inputs =
//         inputs.iter().map(|input| {
//             let mask = rng.clone().gen_bool();
//             results.push(mask);
//             [input.as_slice(), &[mask as u128]].concat()
//         }).collect::<Vec<Vec<u128>>>();

//     let wire_inputs = masked_inputs.into_iter().flatten().collect::<Vec<u128>>();
//     let wires = gb_set_fancy_inputs(&mut gb, wire_inputs.as_slice(), inputs.len());

//     let eq = fancy_equality(&mut gb, wires, num_tests).unwrap();
//     gb.outputs(eq.wires()).unwrap();

//     channel.flush().unwrap();
//     let mut ack = [0u8; 1];
//     channel.read_bytes(&mut ack).unwrap();
//     results
// }

// /// The garbler's wire exchange method, now handling u128 values
// fn gb_set_fancy_inputs<F, E>(gb: &mut F, input: &[u128], num_tests: usize) -> EQInputs<F::Item>
// where
//     F: FancyInput<Item = AllWire, Error = E>,
//     E: Debug,
// {
//     // Encoding inputs to binary wires for u128 values
//     let garbler_wires: CrtBundle<F::Item> = gb.encode_bundle(&input, &vec![128; input.len()])
//         .map(CrtBundle::from)
//         .unwrap();

//     // Evaluator receives labels using Oblivious Transfer (OT)
//     let evaluator_wires: CrtBundle<F::Item> = gb.bin_receive(input.len() - num_tests).unwrap();

//     EQInputs {
//         garbler_wires,
//         evaluator_wires,
//     }
// }

// pub fn multiple_ev_leq_test<C>(
//     rng: &mut AesRng,
//     channel: &mut C,
//     inputs: &[Vec<u128>]
// ) -> Vec<bool>
// where
//     C: AbstractChannel + Clone,
// {
//     let num_tests = inputs.len();
//     let mut ev = Evaluator::<C, AesRng, OtReceiver, AllWire>::new(channel.clone(), rng.clone()).unwrap();
//     let input_vec = inputs.to_vec().into_iter().flatten().collect::<Vec<u128>>();
//     let ev_in = input_vec.as_slice();
//     let wires = ev_set_fancy_inputs(&mut ev, &ev_in, num_tests);
//     let eq = fancy_equality(&mut ev, wires, num_tests).unwrap();
//     let output = ev.outputs(eq.wires()).unwrap().unwrap();
//     let results = output.iter().map(|r| *r == 1).collect();

//     channel.write_bytes(&[1u8]).unwrap();
//     channel.flush().unwrap();

//     results
// }

// /// The evaluator's wire exchange method, now using `u128` and `CrtBundle`
// fn ev_set_fancy_inputs<F, E>(ev: &mut F, input: &[u128], num_tests: usize) -> EQInputs<F::Item>
// where
//     F: FancyInput<Item = AllWire, Error = E>,
//     E: Debug,
// {
//     let nwires = input.len();
//     // The evaluator receives the garbler's input labels.
//     let garbler_wires: CrtBundle<F::Item> = ev.bin_receive(nwires + num_tests).unwrap();
//     // The evaluator receives their own input labels using Oblivious Transfer (OT).
//     let evaluator_wires: CrtBundle<F::Item> = ev.encode_bundle(input, &vec![128; nwires]).unwrap();

//     EQInputs {
//         garbler_wires,
//         evaluator_wires,
//     }
// }


// // Extension trait for 'CrtGadgets'
// pub trait CrtGadgets: FancyArithmetic + FancyBinary + ArithmeticBundleGadgets + BundleGadgets {
//     /// Returns 1 if delta^p is greater or equal than (x ⊕ y) in a CRT sense.
//     fn bin_leq_bundles(
//         &mut self,
//         x: &CrtBundle<Self::Item>,
//         y: &CrtBundle<Self::Item>,
//     ) -> Result<Self::Item, Self::Error> {
//         // Here we perform a CRT addition.
//         let sum = self.crt_add(x, y)?;
//         // Create a constant CRT bundle for the constant value.
//         let delta_p = self.crt_constant_bundle(12, 9223372036854775783u128)?;
//         // Compare: this returns 1 (true) if delta_p is greater or equal to sum under the modulus.
//         // Make sure to supply any extra parameters (for instance "accuracy") if required.
//         self.crt_geq(&delta_p, &sum, "default_accuracy")
//     }

//     /// Shared equality test for two bundles.
//     /// NOTE: In your original code this worked with BinaryBundles.
//     /// When using CRT bundles, one common method is to subtract and then test for zero.
//     fn bin_eq_bundles_shared(
//         &mut self,
//         x: &CrtBundle<Self::Item>,
//         y: &CrtBundle<Self::Item>,
//     ) -> Result<Self::Item, Self::Error> {
//         assert_eq!(x.wires().len(), y.wires().len() + 1, "x must have one more wire than y");

//         let (x_wires, mask) = x.wires().split_at(x.wires().len() - 1);
//         let mask = &mask[0]; // Last wire is the mask

//         let eq_result = self.bin_leq_bundles(&CrtBundle::new(x_wires.to_vec()), y)?;

//         self.xor(&eq_result, mask) // Obscure the output with the mask
//     }

//     /// Shared equality test for multiple parts.
//     /// In this version we assume that both x and y are provided as CRT bundles and the wires
//     /// contained inside represent digits (or bits) of numbers.
//     fn multi_bin_eq_bundles_shared(
//         &mut self,
//         // We now require CRT bundles for both x and y.
//         x: &CrtBundle<Self::Item>,
//         y: &CrtBundle<Self::Item>,
//         num_tests: usize,
//     ) -> Result<CrtBundle<Self::Item>, Self::Error> {
//         // Here we expect that the CRT bundle `x` contains an extra “mask” wire per test.
//         assert_eq!(
//             x.wires().len(),
//             y.wires().len() + num_tests,
//             "each string in x must have one extra mask bit"
//         );
//         // Assume that y wires can be segmented equally.
//         assert_eq!(y.wires().len() % num_tests, 0);

//         let string_len = y.wires().len() / num_tests;
//         let mut results = Vec::with_capacity(num_tests);

//         for i in 0..num_tests {
//             let x_start = i * (string_len + 1);
//             let y_start = i * string_len;
//             // Construct CRT bundles from slices. (Ensure that your wire representation supports this)
//             let x_slice = CrtBundle::new(x.wires()[x_start..x_start+string_len].to_vec());
//             let y_slice = CrtBundle::new(y.wires()[y_start..y_start+string_len].to_vec());
//             let eq_result = self.bin_eq_bundles_shared(&x_slice, &y_slice)?;
//             // The extra wire in x at position x_start + string_len acts as a mask.
//             let masked_result = self.xor(&eq_result, &x.wires()[x_start+string_len])?;
//             results.push(masked_result);
//         }
//         // Return a CRT bundle built from these results.
//         Ok(CrtBundle::new(results))
//     }
// }

// /// Implement CrtGadgets for `Garbler`
// impl<C, R, S, W> CrtGadgets for fancy_garbling::twopac::semihonest::Garbler<C, R, S, W>
// where
//     Self: FancyArithmetic + FancyBinary + ArithmeticBundleGadgets + BundleGadgets,
// {
//     // Because `CrtGadgets` provides default implementations for its methods,
//     // you do not need to implement anything here unless you want to override something.
// }

// /// Implement CrtGadgets for `Evaluator`
// impl<C, R, S, W> CrtGadgets for fancy_garbling::twopac::semihonest::Evaluator<C, R, S, W>
// where
//     Self: FancyArithmetic + FancyBinary + ArithmeticBundleGadgets + BundleGadgets,
// {
//     // Similarly, any CRT-gadget methods will be available here via defaults.
// }


// /// Fancy equality test using garbled circuits
// fn fancy_equality<F>(
//     f: &mut F,
//     wire_inputs: EQInputs<F::Item>,
//     num_tests: usize,
// ) -> Result<CrtBundle<F::Item>, F::Error>
// where
//     F: FancyArithmetic + FancyBinary + ArithmeticBundleGadgets + BundleGadgets
// {
//     let result_bits = f.multi_bin_eq_bundles_shared(&wire_inputs.garbler_wires, &wire_inputs.evaluator_wires,num_tests)?;
//     Ok(result_bits)
// }

use fancy_garbling::{
    twopac::semihonest::{Evaluator, Garbler},
    util,
    AllWire,
    ArithmeticBundleGadgets,
    // We keep both bundle types in the import, though we’re now using CRT bundles.
    BinaryBundle,
    BundleGadgets,
    CrtBundle,
    Fancy,
    FancyArithmetic,
    FancyBinary,
    FancyInput,
    FancyReveal,
};

use ocelot::{ot::AlszReceiver as OtReceiver, ot::AlszSender as OtSender};
use scuttlebutt::{AbstractChannel, AesRng, Channel, SyncChannel};

use std::fmt::Debug;
use std::io::{Read, Write};
use std::time::Instant;
use fancy_garbling::util::RngExt;
use ocelot::ot::Sender;
use rayon::prelude::*;
const Q: u128 = 9223372036854775807u128; 
/// A structure that contains both the garbler’s and evaluator’s wires.
/// (We now use CRT bundles instead of Binary bundles.)
struct EQInputs<F> {
    pub garbler_wires: CrtBundle<F>,
    pub evaluator_wires: CrtBundle<F>,
}

/// Changed to use CRT bundles and input vectors of u128 values.
/// NOTE: The caller must supply vectors of u128 values. If you currently have Vec<Vec<u16>>, convert them.
pub fn multiple_gb_leq_test<C>(
    rng: &mut AesRng,
    channel: &mut C,
    // The caller’s input type must be &[Vec<u128>].
    inputs: &[Vec<u128>]
) -> Vec<bool>
where
    C: AbstractChannel + Clone,
{
    let num_tests = inputs.len();
    let mut results = Vec::with_capacity(num_tests);
    let mut gb = Garbler::<C, AesRng, OtSender, AllWire>::new(channel.clone(), rng.clone()).unwrap();

    // Create masked inputs by appending a boolean mask (converted to u128).
    let masked_inputs: Vec<Vec<u128>> = inputs.iter().map(|input| {
        let mask = rng.clone().gen_bool();
        results.push(mask);
        // Note: we assume each element of input is already u128.
        [input.as_slice(), &[mask as u128]].concat()
    }).collect();

    // Flatten the 2D vector into one vector.
    let wire_inputs: Vec<u128> = masked_inputs.into_iter().flatten().collect();
    let wires = gb_set_fancy_inputs(&mut gb, wire_inputs.as_slice(), inputs.len());

    let eq = fancy_equality(&mut gb, wires, num_tests).unwrap();
    gb.outputs(eq.wires()).unwrap();

    channel.flush().unwrap();
    let mut ack = [0u8; 1];
    channel.read_bytes(&mut ack).unwrap();
    results
}

/// The garbler's wire exchange method, now working with u128 values and CRT bundles.
/// Note: Since the fancy method encode_bundle expects &[u16], we convert the u128 values.
fn gb_set_fancy_inputs<F, E>(
    gb: &mut F,
    input: &[u128],
    num_tests: usize,
) -> EQInputs<F::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    // We still need to encode some values into wires.
    // In your previous code you converted the u128 values to u16; 
    // this may be acceptable if your values actually fit into u16.
    let input_u16: Vec<u16> = input.iter().map(|&x| x as u16).collect();

    // Here we use the existing encode_bundle method to produce a bundle.
    // The provided radix vector here is vec![128u16; input.len()] which means
    // each digit is in base 128. (Make sure that fits your intended CRT encoding.)
    // let garbler_wires: CrtBundle<F::Item> = gb
    //     .encode_bundle(&input_u16, &vec![128u16; input.len()])
    //     .map(CrtBundle::from)
    //     .unwrap();

    let garbler_wires: CrtBundle<F::Item> = gb
    .encode_bundle(&input_u16, &vec![127u16; input.len()])  // use a square-free base
    .map(CrtBundle::from)
    .unwrap();


    // Instead of using bin_receive or converting from a BinaryBundle,
    // we now use the CRT-specific receive method.
    // The documentation tells us that:
    //      fn crt_receive(&mut self, modulus: u128) -> Result<CrtBundle<Self::Item>, Self::Error>
    // so we pass our modulus Q.
    let evaluator_wires: CrtBundle<F::Item> = gb.crt_receive(Q).unwrap();

    EQInputs {
        garbler_wires,
        evaluator_wires,
    }
}


/// For the evaluator.
pub fn multiple_ev_leq_test<C>(
    rng: &mut AesRng,
    channel: &mut C,
    inputs: &[Vec<u128>]
) -> Vec<bool>
where
    C: AbstractChannel + Clone,
{
    let num_tests = inputs.len();
    let mut ev = Evaluator::<C, AesRng, OtReceiver, AllWire>::new(channel.clone(), rng.clone()).unwrap();
    let input_vec: Vec<u128> = inputs.iter().flatten().cloned().collect();
    let wires = ev_set_fancy_inputs(&mut ev, input_vec.as_slice(), num_tests);
    let eq = fancy_equality(&mut ev, wires, num_tests).unwrap();
    let output = ev.outputs(eq.wires()).unwrap().unwrap();
    let results = output.iter().map(|r| *r == 1).collect::<Vec<bool>>();

    channel.write_bytes(&[1u8]).unwrap();
    channel.flush().unwrap();

    results
}

/// The evaluator's wire exchange method, now using u128 and CRT bundles.
fn ev_set_fancy_inputs<F, E>(
    ev: &mut F,
    input: &[u128],
    num_tests: usize,
) -> EQInputs<F::Item>
where
    F: FancyInput<Item = AllWire, Error = E>,
    E: Debug,
{
    let nwires = input.len();
    // Define our modulus Q; note that every CRT operation in the protocol must use the same modulus.
    
    // The evaluator now receives the garbler's input labels using the CRT-aware receive function.
    // We pass Q (the modulus) rather than the number of wires.
    let garbler_wires: CrtBundle<F::Item> = ev.crt_receive(Q).unwrap();
    
    // Convert the evaluator's input from u128 to u16 for the encode_bundle method.
    // (This only works if your u128 values actually fit in u16; if not you would need a different method such as crt_encode_many.)
    let input_u16: Vec<u16> = input.iter().map(|&x| x as u16).collect();
    
    // Encode the evaluator's input into wires using CRT encoding.
    // let evaluator_wires: CrtBundle<F::Item> = ev
    //     .encode_bundle(&input_u16, &vec![128u16; nwires])
    //     .unwrap()
    //     .into();
    let evaluator_wires: CrtBundle<F::Item> = ev
    .encode_bundle(&input_u16, &vec![127u16; nwires])
    .unwrap()
    .into();

    
    EQInputs {
        garbler_wires,
        evaluator_wires,
    }
}


/// ---
///
/// Extension trait for CRT gadgets. We add a supertrait bound on `fancy_garbling::CrtGadgets`
/// so that methods like `crt_add`, `crt_constant_bundle`, and `crt_geq` are available.
pub trait CrtGadgets: FancyArithmetic + FancyBinary + ArithmeticBundleGadgets + BundleGadgets + fancy_garbling::CrtGadgets {
    /// Returns 1 if delta^p is greater or equal than (x + y) in a CRT sense.
    fn bin_leq_bundles(
        &mut self,
        x: &CrtBundle<Self::Item>,
        y: &CrtBundle<Self::Item>,
    ) -> Result<Self::Item, Self::Error> {
        // Perform a CRT addition.
        let sum = self.crt_add(x, y)?;
        // Create a constant CRT bundle for the value 12 under modulus Q.
        // (Here Q is hard-coded; you may want to make it a parameter.)
        let delta_p = self.crt_constant_bundle(12, Q)?;
        // Compare delta_p and sum. (Assumes crt_geq returns 1 when delta_p >= sum.)
        self.crt_geq(&delta_p, &sum, "default_accuracy")
    }

    /// Shared equality test for two bundles.
    /// Here we subtract and compare the result with zero.
    fn bin_eq_bundles_shared(
        &mut self,
        x: &CrtBundle<Self::Item>,
        y: &CrtBundle<Self::Item>,
    ) -> Result<Self::Item, Self::Error> {
        let diff = self.crt_sub(x, y)?;
        let zero = self.crt_constant_bundle(0, Q)?;
        // For illustration, we use crt_geq as a proxy for equality testing.
        // (A proper equality test would combine crt_geq and crt_leq or use a dedicated method.)
        self.crt_geq(&diff, &zero, "default_accuracy")
    }

    /// Shared equality test for multiple parts.
    fn multi_bin_eq_bundles_shared(
        &mut self,
        x: &CrtBundle<Self::Item>,
        y: &CrtBundle<Self::Item>,
        num_tests: usize,
    ) -> Result<CrtBundle<Self::Item>, Self::Error> {
        // Expect that x has one extra mask wire per test.
        assert_eq!(
            x.wires().len(),
            y.wires().len() + num_tests,
            "each string in x must have one extra mask bit"
        );
        // y.wires() must be divisible by num_tests.
        assert_eq!(y.wires().len() % num_tests, 0);

        let string_len = y.wires().len() / num_tests;
        let mut results = Vec::with_capacity(num_tests);

        for i in 0..num_tests {
            let x_start = i * (string_len + 1);
            let y_start = i * string_len;
            // Re-bundle slices as CRT bundles.
            let x_slice = CrtBundle::new(x.wires()[x_start..x_start+string_len].to_vec());
            let y_slice = CrtBundle::new(y.wires()[y_start..y_start+string_len].to_vec());
            let eq_result = self.bin_eq_bundles_shared(&x_slice, &y_slice)?;
            // Use the extra wire as a mask.
            let masked_result = self.xor(&eq_result, &x.wires()[x_start+string_len])?;
            results.push(masked_result);
        }
        Ok(CrtBundle::new(results))
    }
}

/// Implement CrtGadgets for Garbler.
impl<C, R, S, W> CrtGadgets for Garbler<C, R, S, W>
where
    Self: FancyArithmetic + FancyBinary + ArithmeticBundleGadgets + BundleGadgets + fancy_garbling::CrtGadgets,
{
    // Defaults are used.
}

/// Implement CrtGadgets for Evaluator.
impl<C, R, S, W> CrtGadgets for Evaluator<C, R, S, W>
where
    Self: FancyArithmetic + FancyBinary + ArithmeticBundleGadgets + BundleGadgets + fancy_garbling::CrtGadgets,
{
    // Defaults are used.
}

/// Fancy equality test using garbled circuits.
fn fancy_equality<F>(
    f: &mut F,
    wire_inputs: EQInputs<F::Item>,
    num_tests: usize,
) -> Result<CrtBundle<F::Item>, F::Error>
where
    F: FancyArithmetic + FancyBinary + ArithmeticBundleGadgets + BundleGadgets + CrtGadgets,
{
    let result_bits = f.multi_bin_eq_bundles_shared(
        &wire_inputs.garbler_wires,
        &wire_inputs.evaluator_wires,
        num_tests,
    )?;
    Ok(result_bits)
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
        let masks = multiple_leq_equality_test(&mut rng_gb.clone(), &mut channel, gb_value.as_slice());
        result_sender.send(masks).unwrap();
    });

    let rng_ev = AesRng::new();
    let reader = BufReader::new(receiver.try_clone().unwrap());
    let writer = BufWriter::new(receiver);
    let mut channel = Channel::new(reader, writer);

    let results = multiple_leq_equality_test(&mut rng_ev.clone(), &mut channel, ev_value.as_slice());

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

