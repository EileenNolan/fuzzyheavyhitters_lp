use std::convert::{TryFrom, TryInto};
use std::io::{BufReader, BufWriter};
use std::os::unix::net::UnixStream;
use crate::{all_bit_vectors, prg, Group, Share};

use rayon::prelude::*;
use scuttlebutt::{AesRng, Block, SyncChannel};
use serde::{Deserialize, Serialize};
use crate::ibDCF::{ibDCFKey, EvalState, eval_str};
use ocelot::{ot::AlszReceiver as OtReceiver, ot::AlszSender as OtSender};
use ocelot::ot::{Receiver, Sender};
use crate::equalitytest::{multiple_gb_equality_test, multiple_ev_equality_test}; 
use crate::equalitytest2::{multiple_gb_leq_test, multiple_ev_leq_test};
use crate::sum_leqtest::{multiple_gb_less_test, multiple_ev_less_test};
//use crate::sum_leq_binary::{multiple_gb_sum_batch, multiple_ev_sum_batch};
use crate::field::BlockPair;
use std::marker::PhantomData;
use std::net::TcpStream;
use std::time::Instant;
use crate::FieldElm;
use crate::lagrange::{evaluate_polynomial,evaluate_client_polynomial};
use num_traits::cast::ToPrimitive; // Make sure this is imported
use scuttlebutt::Channel;
use std::sync::{Arc, Mutex};

pub const MODULUS_64: u64 = 9223372036854775783u64;

#[derive(Clone)]
struct TreeNode {
    path: Vec<Vec<bool>>,
    key_states: Vec<Vec<(EvalState, EvalState)>>,
}

unsafe impl Send for TreeNode {}
unsafe impl Sync for TreeNode {}


pub type Poly = Vec<Vec<FieldElm>>;
pub type PolyPair = (Poly, Poly);



#[derive(Clone)]
pub struct KeyCollection<T,U>
{
    depth: usize,
    pub keys: Vec<(bool, Vec<(ibDCFKey, ibDCFKey)>)>,
    pub poly: Vec<Poly>,
    pub polynomials: Vec<Vec<Vec<FieldElm>>>,
    frontier: Vec<TreeNode>,
    frontier_last: Vec<Result<U>>,
    rand_stream: prg::PrgStream,
    _phantom: PhantomData<(T, U)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Result<T> {
    pub path: Vec<Vec<bool>>,
    pub value: T,
}

fn field_elm_to_bits(fe: &FieldElm) -> Vec<bool> {
    let val = fe.value.to_u128().expect("Value does not fit in u128");
    (0..128).map(|i| ((val >> i) & 1) == 1).collect()
}

impl<T, U> KeyCollection<T, U>
where
    T: Share + Clone + std::fmt::Debug + PartialOrd + From<u32> + Send + Sync + TryFrom<Block> + Into<Block>,
    U: Share + Clone + std::fmt::Debug + PartialOrd + From<u32> + Send + Sync + TryFrom<BlockPair> + Into<BlockPair> 
       + Into<u128> + From<FieldElm> + PartialEq,
    <U as TryFrom<BlockPair>>::Error: std::fmt::Debug,
{
    pub fn new(seed: &prg::PrgSeed, depth: usize) -> KeyCollection<T,U> {
        KeyCollection::<T,U> {
            depth,
            keys: vec![],
            polynomials: vec![],      // initialize the new field
            poly: vec![],
            frontier: vec![],
            frontier_last: vec![],
            rand_stream: seed.to_rng(),
            _phantom: PhantomData,
        }
    }

    pub fn add_key(&mut self, key: Vec<(ibDCFKey, ibDCFKey)>) {
        self.keys.push((true, key)); //TODO: come back and remove this bool

    }
    
    // New function: add_polynomial.
    // This function takes a polynomial (which we represent as a Vec<FieldElm>)
    // and stores it. You can later process these polynomials as needed.
    pub fn add_polynomial(&mut self, poly: Poly) {
        self.poly.push(poly);
    }


    pub fn tree_init(&mut self) {
        let mut root = TreeNode {
            path: vec![],
            // value: T::zero(),
            key_states: vec![],
            // key_values: vec![],
        };

        for k in &self.keys {
            let mut root_states = vec![];
            for interval_key in k.1.clone(){
                root_states.push((interval_key.0.eval_init(), interval_key.1.eval_init()));
            }
            root.key_states.push(root_states);
        }

        assert!(self.keys.len() > 0);
        for i in 0..self.keys[0].1.len(){
            root.path.push(vec![]);
        }

        self.frontier.clear();
        self.frontier_last.clear();
        self.frontier.push(root);

    }

    fn make_tree_node(&self, parent: &TreeNode, search_string: &Vec<bool>) -> TreeNode {
        let key_states = self
            .keys
            .par_iter()
            .enumerate()
            .map(|(i, key)| {
                let ev = eval_str(&key.1, &parent.key_states[i], search_string);
                ev
            })
            .collect();

        let mut new_path = vec![];
        for (i, dim_path) in parent.path.iter().enumerate(){
            let mut new_dim_path = dim_path.clone();
            new_dim_path.push(search_string[i]);
            new_path.push(new_dim_path)
        }

        let child = TreeNode {
            path: new_path.clone(),
            // value: child_val,
            key_states,
            // key_values : vec![],
        };
        child
    }

    pub fn tree_crawl(
        &mut self,
        gc_sender: bool,
        channels: &mut [&mut SyncChannel<BufReader<TcpStream>, BufWriter<TcpStream>>]
    ) -> Vec<T> {
        println!("Crawl");
        let start = Instant::now();

        // 1. Prepare next frontier (parallel tree expansion)
        let next_frontier = self
            .frontier
            .par_iter()
            .map(|node| {
                let mut children = vec![];
                let search_strings = all_bit_vectors(node.key_states[0].len());
                for s in search_strings {
                    children.push(self.make_tree_node(node, &s));
                }
                children
            })
            .flatten()
            .collect::<Vec<TreeNode>>();

        let node_client_string: Vec<Vec<Vec<bool>>> = next_frontier
            .par_iter()
            .map(|node| {
                node.key_states
                    .par_iter()
                    .map(|state| {
                        let mut left_bits: Vec<bool> = state.iter()
                            .map(|(left, _)| left.y_bit ^ left.bit)
                            .collect();
                        let mut right_bits: Vec<bool> = state.iter()
                            .map(|(_, right)| right.y_bit ^ right.bit)
                            .collect();
                        left_bits.append(&mut right_bits);
                        left_bits
                    })
                    .collect()
            })
            .collect();

        let non_mpc = start.elapsed();
        println!("Tree searching and FSS - {:?}", non_mpc); 
        //PARALLELIZATION

        let all_client_strings: Vec<Vec<u16>> = node_client_string
            .iter()
            .flat_map(|node| node.iter().map(|client| client.iter().map(|&b| b as u16).collect::<Vec<u16>>()))
            .collect();
        let all_node_vals = crossbeam::scope(|s| {
            let mut results = vec![];
            let mut handles = vec![];

            let chunk_size = (all_client_strings.len() + channels.len() - 1) / channels.len();

            for (i, channel) in channels.iter().enumerate() {
                let start_idx = i * chunk_size;
                let end_idx = std::cmp::min(start_idx + chunk_size, all_client_strings.len());
                let chunk = all_client_strings[start_idx..end_idx].to_vec();

                handles.push(s.spawn(move |_| {
                    let mut rng = AesRng::new();
                    let mut channel = (*channel).clone();
                    let bin_shares = if gc_sender {
                        multiple_gb_equality_test(&mut rng, &mut channel, &chunk)
                    } else {
                        multiple_ev_equality_test(&mut rng, &mut channel, &chunk)
                    };
                    let mut node_vals = vec![];
                    if gc_sender{
                        let mut all_shares = Vec::with_capacity(bin_shares.len());
                        for i in 0..bin_shares.len() {
                            let r0 = T::random();
                            let mut r1 = r0.clone();
                            r1.add(&T::one());
                            node_vals.push(r1.clone());
                            let r0_block: Block = r0.try_into().expect("Conversion failed");
                            let r1_block: Block = r1.try_into().expect("Conversion failed");
                            if bin_shares[i] {
                                all_shares.push((r0_block, r1_block));
                            } else {
                                all_shares.push((r1_block, r0_block));
                            }
                        }
                        let mut ot = OtSender::init(&mut channel, &mut rng).unwrap();
                        ot.send(&mut channel, all_shares.as_slice(), &mut rng).map_err(|e| {
                            println!("Error in tree_crawl ot send")
                        }).unwrap();
                    }
                    else{
                        let mut ot = OtReceiver::init(&mut channel, &mut rng).unwrap();
                        let out_blocks = ot.receive(&mut channel, bin_shares.as_slice(), &mut rng).unwrap();
                        node_vals = out_blocks.into_iter()
                            .map(|b| {
                                T::try_from(b)
                                    .map_err(|e| {
                                        // eprintln!("Conversion error: {:?}", e);  // Changed to {:?}
                                        // e
                                    })
                                    .unwrap()
                            })
                            .collect();
                    }
                    node_vals
                }));
            }

            for handle in handles {
                results.extend(handle.join().unwrap());
            }

            results
        }).unwrap();


        let ot = start.elapsed() - non_mpc;
        println!("Garbled Circuit and OT - {:?}", ot);
        let mut results_by_node = Vec::new();
        let mut current_idx = 0;
        for node in &node_client_string {
            let num_clients = node.len();
            let node_results : Vec<T> = all_node_vals[current_idx..current_idx + num_clients].to_vec();
            let mut node_sum = T::zero();
            for (i, v) in node_results.iter().enumerate() {
                // Add in only live values
                if self.keys[i].0 {
                    node_sum.add_lazy(v);
                }
            }
            results_by_node.push(node_sum);
            current_idx += num_clients;
        }

        println!("Field actions - {:?}", start.elapsed() - (ot + non_mpc));
        println!("...done");
        self.frontier = next_frontier;
        results_by_node
    }

    pub fn tree_crawl_last(
        &mut self,
        gc_sender: bool,
        channels: &mut [&mut SyncChannel<BufReader<TcpStream>, BufWriter<TcpStream>>]
    ) -> Vec<U> {
        println!("Crawl");
        let start = Instant::now();

        // 1. Prepare next frontier (parallel tree expansion)
        let next_frontier = self
            .frontier
            .par_iter()
            .map(|node| {
                let mut children = vec![];
                let search_strings = all_bit_vectors(node.key_states[0].len());
                for s in search_strings {
                    children.push(self.make_tree_node(node, &s));
                }
                children
            })
            .flatten()
            .collect::<Vec<TreeNode>>();

        let node_client_string: Vec<Vec<Vec<bool>>> = next_frontier
            .par_iter()
            .map(|node| {
                node.key_states
                    .par_iter()
                    .map(|state| {
                        let mut left_bits: Vec<bool> = state.iter()
                            .map(|(left, _)| left.y_bit ^ left.bit)
                            .collect();
                        let mut right_bits: Vec<bool> = state.iter()
                            .map(|(_, right)| right.y_bit ^ right.bit)
                            .collect();
                        left_bits.append(&mut right_bits);
                        left_bits
                    })
                    .collect()
            })
            .collect();

        let non_mpc = start.elapsed();
        println!("Tree searching and FSS - {:?}", non_mpc);

        let all_client_strings: Vec<Vec<u16>> = node_client_string
            .iter()
            .flat_map(|node| node.iter().map(|client| client.iter().map(|&b| b as u16).collect::<Vec<u16>>()))
            .collect();
        let all_node_vals = crossbeam::scope(|s| {
            let mut results = vec![];
            let mut handles = vec![];

            let chunk_size = (all_client_strings.len() + channels.len() - 1) / channels.len();

            for (i, channel) in channels.iter().enumerate() {
                let start_idx = i * chunk_size;
                let end_idx = std::cmp::min(start_idx + chunk_size, all_client_strings.len());
                let chunk = all_client_strings[start_idx..end_idx].to_vec();

                handles.push(s.spawn(move |_| {
                    let mut rng = AesRng::new();
                    let mut channel = (*channel).clone();
                    let bin_shares = if gc_sender {
                        multiple_gb_equality_test(&mut rng, &mut channel, &chunk)
                    } else {
                        multiple_ev_equality_test(&mut rng, &mut channel, &chunk)
                    };
                    let mut node_vals = vec![];
                    if gc_sender{
                        let mut all_shares = Vec::with_capacity(bin_shares.len());
                        for i in 0..bin_shares.len() {
                            let r0 = U::random();
                            let mut r1 = r0.clone();
                            r1.add(&U::one());
                            node_vals.push(r1.clone());
                            let r0_block: BlockPair = r0.try_into().expect("Conversion failed");
                            let r1_block: BlockPair = r1.try_into().expect("Conversion failed");
                            if bin_shares[i] {
                                all_shares.push((r0_block.0[0], r1_block.0[0]));
                                all_shares.push((r0_block.0[1], r1_block.0[1]));
                            } else {
                                all_shares.push((r1_block.0[0], r0_block.0[0]));
                                all_shares.push((r1_block.0[1], r0_block.0[1]));
                            }
                        }
                        let mut ot = OtSender::init(&mut channel, &mut rng).unwrap();
                        ot.send(&mut channel, all_shares.as_slice(), &mut rng).map_err(|e| {
                            println!("Error in tree_crawl ot send")
                        }).unwrap();
                    }
                    else{
                        let mut ot = OtReceiver::init(&mut channel, &mut rng).unwrap();
                        let doubled_binary_shares = bin_shares.iter().flat_map(|&b| [b, b]).collect::<Vec<bool>>();
                        let out_blocks = ot.receive(&mut channel, doubled_binary_shares.as_slice(), &mut rng).unwrap();
                        let mut i = 0;
                        while i < out_blocks.len() - 1 {
                            let val = U::try_from(BlockPair([out_blocks[i], out_blocks[i+1]])).map_err(|e| {}).unwrap();
                            node_vals.push(val);
                            i += 2;
                        }
                    }
                    node_vals
                }));
            }

            for handle in handles {
                results.extend(handle.join().unwrap());
            }

            results
        }).unwrap();


        let ot = start.elapsed() - non_mpc;
        println!("Garbled Circuit and OT - {:?}", ot);
        let mut results_by_node = Vec::new();
        let mut current_idx = 0;
        for node in &node_client_string {
            let num_clients = node.len();
            let node_results : Vec<U> = all_node_vals[current_idx..current_idx + num_clients].to_vec();
            let mut node_sum = U::zero();
            for (i, v) in node_results.iter().enumerate() {
                // Add in only live values
                if self.keys[i].0 {
                    node_sum.add_lazy(v);
                }
            }
            results_by_node.push(node_sum);
            current_idx += num_clients;
        }

        println!("Field actions - {:?}", start.elapsed() - (ot + non_mpc));
        println!("...done");
        self.frontier_last = next_frontier.par_iter().enumerate().map(|(i,node)| {
                Result::<U> {
                    path: node.path.clone(),
                    value: results_by_node[i].clone(),
                }
            }).collect::<Vec<Result<U>>>();
        results_by_node
    }

    // pub fn evaluate_client_polynomial(client_poly: &[Vec<FieldElm>], w: &[u64]) -> FieldElm {
    //     assert_eq!(client_poly.len(), w.len(), "Dimension mismatch.");
    //     //let mut sum = FieldElm::from(0);
    //     let mut sum = FieldElm::from(0u64);

    //     for i in 0..w.len() {
    //         // If no hashing is involved, directly convert:
    //         let key_i = FieldElm::from(w[i]);
    //         let x_i = evaluate_polynomial(&client_poly[i], &key_i);
    //         sum = FieldElm::from((sum.value + x_i.value) % MODULUS_64);
    //     }
    //     sum
    // }

    pub fn tree_crawl_last_known_poly(
        &mut self,
        gc_sender: bool,
        channels: &mut [&mut SyncChannel<BufReader<TcpStream>, BufWriter<TcpStream>>]
    ) -> u128 {
        println!("pub fn tree_crawl_last_known_poly");
        let start = Instant::now();

        // 1. Prepare next frontier (parallel polynomial evaluations)
        let server_w: Vec<u64> = vec![5u64, 10]; // One value per dimension
        println!("self.polynomials = {:?}", self.polynomials);

        let node_client_string: Vec<FieldElm> = self.poly
            .iter()
            .map(|client_poly| {
                // Evaluate the polynomial over all dimensions using server_w.
                evaluate_client_polynomial(client_poly, &server_w)
            })
            .collect();
        println!("node_client_string length = {}", node_client_string.len());

        // for (i, elm) in node_client_string.iter().enumerate() {
        //     println!("Element {}: {:?}", i, elm);
        // }
            
        // Convert FieldElm values into u128 for the next step
        let next_frontier: Vec<u128> = node_client_string
        .iter()
        .map(|fe| fe.value.to_u128().expect("BigUint to u128 conversion failed"))
        .collect();
        println!("next_frontier length = {}", next_frontier.len());
        //let next_frontier: Vec<u128> = node_client_string.iter().map(|fe| fe.value).collect();

        let non_mpc = start.elapsed();
        println!("Poly evaluations - {:?}", non_mpc);

        // Convert `FieldElm` values into bit-representations for the garbled circuit
        let all_client_strings: Vec<Vec<u16>> = node_client_string
            .iter()
            .map(|fe| {
                // Assuming FieldElm has a method to extract bits
                let bits: Vec<bool> = field_elm_to_bits(fe);
                bits.into_iter().map(|b| if b { 1u16 } else { 0u16 }).collect::<Vec<u16>>()
            })
            .collect();

        println!("PROGRESS STOPPED HERE! Next step is the garbled circuit.");
        println!("number of channels = {}", channels.len());
        let all_node_vals = crossbeam::scope(|s| {
            let mut results = vec![];
            let mut handles = vec![];

            let chunk_size = (all_client_strings.len() + channels.len() - 1) / channels.len();

            for (i, channel) in channels.iter().enumerate() {
                let start_idx = i * chunk_size;
                let end_idx = std::cmp::min(start_idx + chunk_size, all_client_strings.len());
                let chunk = all_client_strings[start_idx..end_idx].to_vec();

                let chunk_u128: Vec<Vec<u128>> = chunk.iter()
                .map(|vec_u16| vec_u16.iter().map(|&x| x as u128).collect())
                .collect();

                handles.push(s.spawn(move |_| {
                    let mut rng = AesRng::new();
                    let mut channel = (*channel).clone();

                    let bin_shares = if gc_sender {
                        multiple_gb_leq_test(&mut rng, &mut channel, &chunk_u128)
                    } else {
                        multiple_ev_leq_test(&mut rng, &mut channel, &chunk_u128)
                    };
                    println!("did gc");
                    let mut node_vals = vec![];
                    if gc_sender{
                        let mut all_shares = Vec::with_capacity(bin_shares.len());
                        for i in 0..bin_shares.len() {
                            let r0 = U::random();
                            let mut r1 = r0.clone();
                            r1.add(&U::one());
                            node_vals.push(r1.clone());
                            let r0_block: BlockPair = r0.try_into().expect("Conversion failed");
                            let r1_block: BlockPair = r1.try_into().expect("Conversion failed");
                            if bin_shares[i] {
                                all_shares.push((r0_block.0[0], r1_block.0[0]));
                                all_shares.push((r0_block.0[1], r1_block.0[1]));
                            } else {
                                all_shares.push((r1_block.0[0], r0_block.0[0]));
                                all_shares.push((r1_block.0[1], r0_block.0[1]));
                            }
                        }
                        let mut ot = OtSender::init(&mut channel, &mut rng).unwrap();
                        ot.send(&mut channel, all_shares.as_slice(), &mut rng).map_err(|e| {
                            println!("Error in tree_crawl ot send")
                        }).unwrap();
                    }
                    else{
                        let mut ot = OtReceiver::init(&mut channel, &mut rng).unwrap();
                        let doubled_binary_shares = bin_shares.iter().flat_map(|&b| [b, b]).collect::<Vec<bool>>();
                        let out_blocks = ot.receive(&mut channel, doubled_binary_shares.as_slice(), &mut rng).unwrap();
                        let mut i = 0;
                        while i < out_blocks.len() - 1 {
                            let val = U::try_from(BlockPair([out_blocks[i], out_blocks[i+1]])).map_err(|e| {}).unwrap();
                            node_vals.push(val);
                            i += 2;
                        }
                    }
                    node_vals
                }));
            }

            for handle in handles {
                results.extend(handle.join().unwrap());
            }

            results
        }).unwrap();


        let ot = start.elapsed() - non_mpc;
        // println!("Garbled Circuit and OT - {:?}", ot);
        // let mut results_by_node = Vec::new();
        // let mut current_idx = 0;
        // for node in &node_client_string {
        //     let num_clients = node.len();
        //     let node_results : Vec<U> = all_node_vals[current_idx..current_idx + num_clients].to_vec();
        //     let mut node_sum = U::zero();
        //     for (i, v) in node_results.iter().enumerate() {
        //         // Add in only live values
        //         if self.keys[i].0 {
        //             node_sum.add_lazy(v);
        //         }
        //     }
        //     results_by_node.push(node_sum);
        //     current_idx += num_clients;
        // }

        // Assuming node_client_string is now a Vec<FieldElm> from your polynomial evaluations.
        // println!("Garbled Circuit and OT - {:?}", ot);

        // // If you want to aggregate them (e.g., sum relevant results) into one final result:
        // let mut final_sum = U::zero();
        // for (i, fe) in node_client_string.iter().enumerate() {
        //     // Optionally, you might check if the corresponding key is marked "live"
        //     if self.keys[i].0 {
        //         // Depending on the type U you might have a conversion from FieldElm or extract u128, etc.
        //         // Here, we assume you have a conversion U::from_field_elm(fe) or similar.
        //         final_sum.add_lazy(&U::from_field_elm(fe));
        //     }
        // }

        // println!("Field actions - {:?}", start.elapsed() - (ot + non_mpc));
        // println!("...done");
        // self.frontier_last = next_frontier.par_iter().enumerate().map(|(i,node)| {
        //         Result::<U> {
        //             path: node.path.clone(),
        //             value: results_by_node[i].clone(),
        //         }
        //     }).collect::<Vec<Result<U>>>();
        // results_by_node

        //println!("Garbled Circuit and OT - {:?}", ot);

        // Sum up all binary values (bits) in all_node_vals to get a final integer result.
        // Sum up all binary values (bits) in all_node_vals to get a final integer result.
        println!("about to calcualte sum");
        println!("number of channelsvalues in gc results = {}", all_node_vals.len());
        // Assuming your output type is FieldElm and that FieldElm::one() gives you the representation of 1.
        let final_sum: u128 = all_node_vals
        .iter()
        .map(|bit| if *bit == U::from(FieldElm::one()) { 1u128 } else { 0u128 })
        .sum();
    


        println!("Final result after garbled circuit: {}", final_sum);
        println!("Field actions - {:?}", start.elapsed() - (ot + non_mpc));
        println!("...done");

        // Return the sum as an integer
        final_sum

    }

    pub fn tree_prune(&mut self, alive_vals: &[bool]) {
        assert_eq!(alive_vals.len(), self.frontier.len());

        // Remove from back to front to preserve indices
        for i in (0..alive_vals.len()).rev() {
            if !alive_vals[i] {
                self.frontier.remove(i);
            }
        }

        //println!("Size of frontier: {:?}", self.frontier.len());
    }

    pub fn tree_prune_last(&mut self, alive_vals: &[bool]) {
        assert_eq!(alive_vals.len(), self.frontier_last.len());

        // Remove from back to front to preserve indices
        for i in (0..alive_vals.len()).rev() {
            if !alive_vals[i] {
                self.frontier_last.remove(i);
            }
        }

        //println!("Size of frontier: {:?}", self.frontier.len());
    }


    pub fn keep_values(nclients: usize, threshold: &T, vals0: &[T], vals1: &[T]) -> Vec<bool> {
        assert_eq!(vals0.len(), vals1.len());

        let nclients = T::from(nclients as u32);
        let mut keep = vec![];
        for i in 0..vals0.len() {
            let mut v = T::zero();
            v.add(&vals0[i]);
            v.sub(&vals1[i]);
            // println!("-> {:?} {:?} {:?}", v, *threshold, nclients);

            debug_assert!(v <= nclients);

            // Keep nodes that are above threshold
            // println!("{:?}",v);
            keep.push(v >= *threshold);
        }

        keep
    }

    pub fn keep_values_last(nclients: usize, threshold: &U, vals0: &[U], vals1: &[U]) -> Vec<bool> {
        assert_eq!(vals0.len(), vals1.len());

        let nclients = U::from(nclients as u32);
        let mut keep = vec![];
        for i in 0..vals0.len() {
            let mut v = U::zero();
            let mut v0 = vals0[i].clone();
            let mut v1 = vals1[i].clone();
            v0.reduce();
            v1.reduce();
            v.add(&v0);
            v.sub(&v1);
            // println!("-> {:?} {:?} {:?}", v, *threshold, nclients);

            debug_assert!(v <= nclients);

            // Keep nodes that are above threshold
            // println!("{:?}",v);
            keep.push(v >= *threshold);
        }

        keep
    }



    pub fn final_shares(&self) -> Vec<Result<U>> {
        let mut alive = vec![];
        for n in &self.frontier_last {
            alive.push(Result::<U> {
                path: n.path.clone(),
                value: n.value.clone()
            });

            println!("Final {:?}", n.path);
        }

        alive
    }

    pub fn final_values(res0: &[Result<U>], res1: &[Result<U>]) -> Vec<Result<U>> {
        assert_eq!(res0.len(), res1.len());

        let mut out = vec![];
        for i in 0..res0.len() {
            assert_eq!(res0[i].path, res1[i].path);

            let mut v = U::zero();
            let mut v0 = res0[i].value.clone();
            let mut v1 = res1[i].value.clone();
            v0.reduce();
            v1.reduce();
            v.add(&v0);
            v.sub(&v1);

            out.push(Result {
                path: res0[i].path.clone(),
                value: v,
            });
        }

        out
    }
}

