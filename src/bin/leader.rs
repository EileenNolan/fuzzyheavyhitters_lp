use counttree::{add_bitstrings, collect, config, fastfield, rpc::{
    AddKeysRequest, PolyRequest, PolyRequestU2, FinalSharesRequest, ResetRequest,
    TreeInitRequest,
    TreeCrawlRequest,
}, string_to_bits, FieldElm, MSB_u32_to_bits};

use std::time::Instant;

use futures::try_join;
use std::io;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use tarpc::{
    client,
    context,
    serde_transport::tcp,
    tokio_serde::formats::Bincode,
    //server::{self, Channel},
};

use rand::distributions::Alphanumeric;

use std::time::{Duration, SystemTime};
use counttree::ibDCF::{eval_str, ibDCFKey};
use counttree::rpc::{TreeCrawlLastRequest, TreePruneLastRequest, TreePruneRequest};
use counttree::sample_covid_data::sample_covid_locations;
use counttree::lagrange::{compute_polynomials, compute_polynomials_prefix};
use std::convert::TryInto;

type IntervalKey = (ibDCFKey, ibDCFKey);
fn long_context() -> context::Context {
    let mut ctx = context::current();

    // Increase timeout to one hour
    ctx.deadline = SystemTime::now() + Duration::from_secs(1000000);
    ctx
}

fn sample_string(len: usize) -> String {
    let mut rng = rand::thread_rng();
    std::iter::repeat(())
        .map(|()| rng.sample(Alphanumeric))
        .take(len / 8)
        .collect()
}
fn generate_random_bit_vectors(len: usize, d: usize) -> Vec<Vec<bool>> {
    let mut rng = rand::thread_rng();
    (0..d)
        .map(|_| {
            let s: String = std::iter::repeat(())
                .map(|()| rng.sample(Alphanumeric))
                .take((len + 7) / 8) // Round up to ensure enough bits
                .collect();
            let mut bits = string_to_bits(&s);
            bits.truncate(len);
            bits
        })
        .collect()
}

fn generate_strings(cfg: &config::Config, aug_len : usize) -> Vec<Vec<Vec<bool>>> {
    (0..cfg.num_sites)
        .map(|_| {
            generate_random_bit_vectors(cfg.data_len - aug_len, cfg.n_dims) //leaving space for later per-client augmentation
        })
        .collect::<Vec<Vec<Vec<bool>>>>()
}
fn generate_covid_samples(nreq : usize, aug_len : usize) -> Vec<Vec<Vec<bool>>> {
    let covid_path = "data/COVID-19_Case_Surveillance_Public_Use_Data_with_Geography_20250430.csv";
    let centroids_path = "data/county_centroids.csv";
    if aug_len > 0 {
        sample_covid_locations(&covid_path, &centroids_path, nreq, Some(aug_len as f64)).unwrap()
    }
    else{
        sample_covid_locations(&covid_path, &centroids_path, nreq, None).unwrap()
    }
}

fn augment_string(string: Vec<Vec<bool>>, aug_len : usize) -> Vec<Vec<bool>> {
    let mut out = vec![];
    for mut d_str in string{
        let aug : String = sample_string(aug_len);
        let mut bits = string_to_bits(&aug);
        d_str.append(&mut bits);
        out.push(d_str);
    }
    out
}

/// Convert a slice of bools to a u64 value.
/// Assumes that the first bit is the most significant.
fn bits_to_u64(bits: &[bool]) -> u64 {
    bits.iter().fold(0u64, |acc, &bit| (acc << 1) | if bit { 1 } else { 0 })
}


fn generate_keys(cfg: &config::Config) -> (Vec<Vec<IntervalKey>>, Vec<Vec<IntervalKey>>) {
    let (keys0, keys1): (Vec<Vec<IntervalKey>>, Vec<Vec<IntervalKey>>) = rayon::iter::repeat(0)
        .take(cfg.num_sites)
        .map(|_| {
            let data = generate_random_bit_vectors(cfg.data_len, cfg.n_dims);
            let keys = ibDCFKey::gen_l_inf_ball(data, 1);
            (keys.0.clone(), keys.1.clone())
        })
        .collect::<Vec<_>>()
        .into_iter()
        .unzip();
    let encoded: Vec<u8> = bincode::serialize(&keys0[0]).unwrap();
    println!("Key size: {:?} bytes", encoded.len());
    (keys0, keys1)
}

/// Generate polynomial shares for each "site" using random bit vectors.
///
/// This function mimics the structure of the original `generate_keys` function.
/// It first produces for each site a set of random bit vectors (using `generate_random_bit_vectors`),
/// then converts each bit vector to a u64 query value. Finally, it computes polynomial shares
/// (via `compute_polynomials`) on that query.
fn generate_poly_shares(cfg: &config::Config) -> (Vec<Vec<Vec<FieldElm>>>, Vec<Vec<Vec<FieldElm>>>) {
    // Use Rayon to generate one set of polynomial shares per site in parallel.
    // The result is a vector of tuples: one tuple per site.
    // Each tuple contains (poly_shares_A_site, poly_shares_B_site),
    // where each poly_shares_* is a Vec<Vec<FieldElm>>.
    let (poly_shares_A, poly_shares_B): (Vec<Vec<Vec<FieldElm>>>, Vec<Vec<Vec<FieldElm>>>) = 
        rayon::iter::repeat(0)
            .take(cfg.num_sites)
            .map(|_| {
                // Generate `cfg.n_dims` random bit vectors, each of length `cfg.data_len`
                let data: Vec<Vec<bool>> = generate_random_bit_vectors(cfg.data_len, cfg.n_dims);
                // Convert each vector of bools to a u64 value.
                let q: Vec<u64> = data.iter().map(|bits| bits_to_u64(bits)).collect();
                // Compute the polynomial shares using your unchanged compute_polynomials function.
                compute_polynomials(&q, cfg.n_dims, cfg.ball_size as i64)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .unzip();

    // Serialize the polynomial shares for the first site of the A share
    // to check the size (for benchmarking or debugging purposes).
    let encoded: Vec<u8> = bincode::serialize(&poly_shares_A[0]).unwrap();
    println!("Polynomial shares size: {:?} bytes", encoded.len());
    
    (poly_shares_A, poly_shares_B)
}

async fn reset_servers(
    client0: &mut counttree::CollectorClient,
    client1: &mut counttree::CollectorClient,
) -> io::Result<()> {
    let req = ResetRequest {};
    let response0 = client0.reset(long_context(), req.clone());
    let response1 = client1.reset(long_context(), req);
    try_join!(response0, response1).unwrap();

    Ok(())
}

async fn tree_init(
    client0: &mut counttree::CollectorClient,
    client1: &mut counttree::CollectorClient,
) -> io::Result<()> {
    let req = TreeInitRequest {};
    let response0 = client0.tree_init(long_context(), req.clone());
    let response1 = client1.tree_init(long_context(), req);
    try_join!(response0, response1).unwrap();

    Ok(())
}

async fn add_fuzzy_keys(
    cfg: &config::Config,
    client0: counttree::CollectorClient,
    client1: counttree::CollectorClient,
    strings: &Vec<Vec<Vec<bool>>>,
    nreqs: usize,
    aug_len: usize,
) -> io::Result<()> {
    use rand::distributions::Distribution;
    let mut rng = rand::thread_rng();
    let zipf = zipf::ZipfDistribution::new(cfg.num_sites, cfg.zipf_exponent).unwrap(); //TODO: replace with real dist

    let mut addkey0 = Vec::with_capacity(nreqs);
    let mut addkey1 = Vec::with_capacity(nreqs);

    for i in 0..nreqs {
        let sample = zipf.sample(&mut rng) - 1;
        let key_str = augment_string(strings[i].clone(), aug_len);
        let (key0, key1) = ibDCFKey::gen_l_inf_ball(key_str, cfg.ball_size as u32);
        addkey0.push(key0);
        addkey1.push(key1);
    }


    let req0 = AddKeysRequest { keys: addkey0 };
    let req1 = AddKeysRequest { keys: addkey1 };

    let response0 = client0.add_keys(long_context(), req0.clone());
    let response1 = client1.add_keys(long_context(), req1.clone());

    try_join!(response0, response1).unwrap();

    Ok(())
}

// async fn add_polynomials(
//     cfg: &config::Config,
//     client0: counttree::CollectorClient,
//     client1: counttree::CollectorClient,
//     strings: &Vec<Vec<Vec<bool>>>,
//     nreqs: usize,
//     aug_len: usize,
// ) -> io::Result<()> {
//     println!("pub fn add_polynomials");
//     use rand::distributions::Distribution;
//     let mut rng = rand::thread_rng();
//     let zipf = zipf::ZipfDistribution::new(cfg.num_sites, cfg.zipf_exponent)
//         .unwrap(); // TODO: replace with real dist

//     // Vectors to hold your batch of polynomial shares.
//     let mut addpoly0 = Vec::with_capacity(nreqs);
//     let mut addpoly1 = Vec::with_capacity(nreqs);

//     for i in 0..nreqs {
//         let sample = zipf.sample(&mut rng) - 1;
//         let key_str = augment_string(strings[i].clone(), aug_len);

//         // Convert each Vec<bool> into a u64 inline.
//         let q: Vec<u64> = key_str.iter().map(|bits| {
//             // Initialize accumulator.
//             let mut value: u64 = 0;
//             // For each bit in the vector, shift and add the bit.
//             for &b in bits {
//                 value = (value << 1) | if b { 1 } else { 0 };
//             }
//             value
//         }).collect();

//         // Here we assume compute_polynomials returns a pair: (poly0, poly1)
//         let (poly0, poly1) =
//             compute_polynomials(&q, cfg.n_dims, cfg.ball_size as i64);

//         addpoly0.push(poly0);
//         addpoly1.push(poly1);
//     }

//     let req0 = PolyRequest { poly: addpoly0 };
//     let req1 = PolyRequest { poly: addpoly1 };

//     let response0 = client0.add_polynomials(long_context(), req0.clone());
//     let response1 = client1.add_polynomials(long_context(), req1.clone());

//     try_join!(response0, response1).map_err(|e| {
//         eprintln!("Error in try_join!: {:?}", e);
//         io::Error::new(io::ErrorKind::Other, format!("Disconnected: {:?}", e))
//     })?;
    

//     Ok(())
// }

async fn add_polynomials(
    cfg: &config::Config,
    client0: counttree::CollectorClient,
    client1: counttree::CollectorClient,
    strings: &Vec<Vec<Vec<bool>>>,
    nreqs: usize,
    aug_len: usize,
) -> io::Result<()> {
    println!("pub fn add_polynomials: starting to generate polynomials");
    use rand::distributions::Distribution;
    let mut rng = rand::thread_rng();
    let zipf = zipf::ZipfDistribution::new(cfg.num_sites, cfg.zipf_exponent)
        .unwrap(); // TODO: replace with real distribution

    // Vectors to hold your batch of polynomial shares.
    let mut addpoly0 = Vec::with_capacity(nreqs);
    let mut addpoly1 = Vec::with_capacity(nreqs);

    for i in 0..nreqs {
        // Debug: report progress periodically.
        if i % 1000 == 0 {
            println!("Processing polynomial request {}/{}", i, nreqs);
        }
        let sample = zipf.sample(&mut rng) - 1;
        let key_str = augment_string(strings[i].clone(), aug_len);

        // Convert each Vec<bool> into a u64 inline.
        let q: Vec<u64> = key_str
            .iter()
            .map(|bits| {
                let mut value: u64 = 0;
                for &b in bits {
                    value = (value << 1) | if b { 1 } else { 0 };
                }
                value
            })
            .collect();

        // Here we assume compute_polynomials returns a pair: (poly0, poly1)
        let (poly0, poly1) = compute_polynomials(&q, cfg.n_dims, cfg.ball_size as i64);
        addpoly0.push(poly0);
        addpoly1.push(poly1);
    }

    println!(
        "Finished generating polynomials. Total requests: {}",
        nreqs
    );

    let req0 = PolyRequest { poly: addpoly0 };
    let req1 = PolyRequest { poly: addpoly1 };

    println!("Sending polynomial requests to clients.");
    let response0 = client0.add_polynomials(long_context(), req0.clone());
    let response1 = client1.add_polynomials(long_context(), req1.clone());
    println!("Requests sent. Awaiting responses...");

    match try_join!(response0, response1) {
        Ok(_) => {
            println!("Received successful responses from both clients.");
            Ok(())
        }
        Err(e) => {
            eprintln!("Error in try_join!: {:?}", e);
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Disconnected: {:?}", e),
            ))
        }
    }
}

async fn add_polynomials_unknown(
    cfg: &config::Config,
    client0: counttree::CollectorClient,
    client1: counttree::CollectorClient,
    strings: &Vec<Vec<Vec<bool>>>,
    nreqs: usize,
    aug_len: usize,
) -> io::Result<()> {
    println!("pub fn add_polynomials: starting to generate polynomials");
    use rand::distributions::Distribution;
    let mut rng = rand::thread_rng();
    let zipf = zipf::ZipfDistribution::new(cfg.num_sites, cfg.zipf_exponent)
        .unwrap(); // TODO: replace with real distribution

    // Vectors to hold your batch of polynomial shares.
    let mut addpoly0 = Vec::with_capacity(nreqs);
    let mut addpoly1 = Vec::with_capacity(nreqs);

    for i in 0..nreqs {
        let mut client0_all = Vec::new();
        let mut client1_all = Vec::new();


        // Debug: report progress periodically.
        if i % 1000 == 0 {
            println!("Processing polynomial request {}/{}", i, nreqs);
        }
        let sample = zipf.sample(&mut rng) - 1;
        let key_str = augment_string(strings[i].clone(), aug_len);

        // Convert each Vec<bool> into a u64 inline.
        let q: Vec<u64> = key_str
            .iter()
            .map(|bits| {
                let mut value: u64 = 0;
                for &b in bits {
                    value = (value << 1) | if b { 1 } else { 0 };
                }
                value
            })
            .collect();

        // Here we assume compute_polynomials returns a pair: (poly0, poly1)
        let mut client0_all_dimension = Vec::new();
        let mut client1_all_dimension = Vec::new();
        for element in q{
            let (poly0, poly1) = compute_polynomials_prefix(element, cfg.ball_size.try_into().unwrap(), cfg.data_len);
            client0_all_dimension.push(poly0);
            client1_all_dimension.push(poly1);

        }
        addpoly0.push(client0_all_dimension);
        addpoly1.push(client1_all_dimension);
    }

    println!(
        "Finished generating polynomials. Total requests: {}",
        nreqs
    );

    let req0 = PolyRequestU2 { poly: addpoly0 };
    let req1 = PolyRequestU2 { poly: addpoly1 };

    println!("Sending polynomial requests to clients.");
    let response0 = client0.add_polynomials_unknown(long_context(), req0.clone());
    let response1 = client1.add_polynomials_unknown(long_context(), req1.clone());
    println!("Requests sent. Awaiting responses...");

    match try_join!(response0, response1) {
        Ok(_) => {
            println!("Received successful responses from both clients.");
            Ok(())
        }
        Err(e) => {
            eprintln!("Error in try_join!: {:?}", e);
            Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Disconnected: {:?}", e),
            ))
        }
    }
}



async fn add_keys(
    cfg: &config::Config,
    client0: counttree::CollectorClient,
    client1: counttree::CollectorClient,
    keys0: Vec<Vec<IntervalKey>>,
    keys1: Vec<Vec<IntervalKey>>,
    nreqs: usize,
) -> io::Result<()> {

    let req0 = AddKeysRequest { keys: keys0 };
    let req1 = AddKeysRequest { keys: keys1 };

    let response0 = client0.add_keys(long_context(), req0.clone());
    let response1 = client1.add_keys(long_context(), req1.clone());

    try_join!(response0, response1).unwrap();

    Ok(())
}

async fn run_level(
    cfg: &config::Config,
    client0: &mut counttree::CollectorClient,
    client1: &mut counttree::CollectorClient,
    level: usize,
    nreqs: usize,
    start_time: Instant,
) -> io::Result<usize> {
    let threshold64 = core::cmp::max(1, (cfg.threshold * (nreqs as f64)) as u64);
    let threshold = fastfield::FE::new(threshold64);

    // Tree crawl
    println!(
        "TreeCrawlStart {:?} {:?} {:?}",
        level,
        "-",
        start_time.elapsed().as_secs_f64()
    );

    let req0 = TreeCrawlRequest { gc_sender: true };
    let req1 = TreeCrawlRequest { gc_sender: false };

    let response0 = client0.tree_crawl(long_context(), req0);
    let response1 = client1.tree_crawl(long_context(), req1);

    let (vals0, vals1) = try_join!(response0, response1).unwrap();

    println!(
        "TreeCrawlDone {:?} {:?} {:?}",
        level,
        "-",
        start_time.elapsed().as_secs_f64()
    );

    assert_eq!(vals0.len(), vals1.len());
    let keep = collect::KeyCollection::<fastfield::FE,FieldElm>::keep_values(nreqs, &threshold, &vals0, &vals1);

    println!("Keep: {:?}", &keep);
    let mut ap = 0;
    for i in keep.clone() {
        if i {ap+= 1;};
    }
    println!("Active paths: {:?}", ap);

    // Tree prune
    let req = TreePruneRequest { keep };
    let response0 = client0.tree_prune(long_context(), req.clone());
    let response1 = client1.tree_prune(long_context(), req);
    try_join!(response0, response1).unwrap();

    Ok(vals0.len())
}

async fn run_level_last(
    cfg: &config::Config,
    client0: &mut counttree::CollectorClient,
    client1: &mut counttree::CollectorClient,
    nreqs: usize,
    start_time: Instant,
) -> io::Result<usize> {
    let threshold64 = core::cmp::max(1, (cfg.threshold * (nreqs as f64)) as u32);
    let threshold = FieldElm::from(threshold64);

    // Tree crawl
    println!(
        "TreeCrawlStart LAST {:?} {:?}",
        "-",
        start_time.elapsed().as_secs_f64()
    );

    let req0 = TreeCrawlLastRequest { gc_sender: true };
    let req1 = TreeCrawlLastRequest { gc_sender: false };

    let response0 = client0.tree_crawl_last(long_context(), req0);
    let response1 = client1.tree_crawl_last(long_context(), req1);

    let (vals0, vals1) = try_join!(response0, response1).unwrap();

    println!(
        "TreeCrawlDone LAST {:?} {:?}",
        "-",
        start_time.elapsed().as_secs_f64()
    );

    assert_eq!(vals0.len(), vals1.len());
    let keep = collect::KeyCollection::<fastfield::FE,FieldElm>::keep_values_last(nreqs, &threshold, &vals0, &vals1);

    println!("Keep: {:?}", keep);

    let req = TreePruneLastRequest { keep };
    let response0 = client0.tree_prune_last(long_context(), req.clone());
    let response1 = client1.tree_prune_last(long_context(), req);
    try_join!(response0, response1).unwrap();

    Ok(vals0.len())
}

async fn run_level_last_known_poly(
    cfg: &config::Config,
    client0: &mut counttree::CollectorClient,
    client1: &mut counttree::CollectorClient,
    nreqs: usize,
    start_time: Instant,
) -> io::Result<usize> {
    let threshold64 = core::cmp::max(1, (cfg.threshold * (nreqs as f64)) as u32);
    let threshold = FieldElm::from(threshold64);

    // Tree crawl
    println!(
        "run_level_known_poly {:?} {:?}",
        "-",
        start_time.elapsed().as_secs_f64()
    );

    let req0 = TreeCrawlLastRequest { gc_sender: true };
    let req1 = TreeCrawlLastRequest { gc_sender: false };

    let response0 = client0.run_level_last_known_poly(long_context(), req0);
    let response1 = client1.run_level_last_known_poly(long_context(), req1);

    let (vals0, vals1) = try_join!(response0, response1).unwrap();

    println!(
        "TreeCrawlDone LAST {:?} {:?}",
        "-",
        start_time.elapsed().as_secs_f64()
    );

    assert_eq!(vals0.len(), vals1.len());
    let keep = collect::KeyCollection::<fastfield::FE,FieldElm>::keep_values_last(nreqs, &threshold, &vals0, &vals1);

    println!("Keep: {:?}", keep);

    // let req = TreePruneLastRequest { keep };
    // let response0 = client0.tree_prune_last(long_context(), req.clone());
    // let response1 = client1.tree_prune_last(long_context(), req);
    // try_join!(response0, response1).unwrap();
    let num_trues = keep.iter().filter(|&&b| b).count();
    println!("Number of trues: {}", num_trues);
    Ok(num_trues)
}

async fn final_shares(
    client0: &mut counttree::CollectorClient,
    client1: &mut counttree::CollectorClient,
) -> io::Result<()> {
    // Final shares
    let req = FinalSharesRequest {};
    let response0 = client0.final_shares(long_context(), req.clone());
    let response1 = client1.final_shares(long_context(), req);
    let (vals0, vals1) = try_join!(response0, response1).unwrap();

    for res in &collect::KeyCollection::<fastfield::FE,FieldElm>::final_values(&vals0, &vals1) {
        println!("Path = {:?}", res.path);
    }

    Ok(())
}

#[tokio::main]
async fn main() -> io::Result<()> {
    println!("Using only one thread!");
    rayon::ThreadPoolBuilder::new().num_threads(1).build_global().unwrap();

    env_logger::init();
    let (cfg, _, nreqs) = config::get_args("Leader", false, true);
    debug_assert_eq!(cfg.data_len % 8, 0);

    // XXX WARNING: THERE IS NO TLS HERE!!!
    let mut client0 = counttree::CollectorClient::new(
        client::Config::default(),
        tcp::connect(cfg.server0, Bincode::default).await?
    ).spawn();

    println!("Successfully connected to server0 at {}", cfg.server0);

    let mut client1 = counttree::CollectorClient::new(
        client::Config::default(),
        tcp::connect(cfg.server1, Bincode::default).await?
    ).spawn();

    println!("Successfully connected to server1 at {}", cfg.server1);

    let start = Instant::now();
    println!("Generating keys DCF...");
    let (bench_keys0, bench_keys1) = generate_keys(&cfg);
    println!("Done.");
    let delta = start.elapsed().as_secs_f64();
    println!(
        "Generated DCF {:?} keys in {:?} seconds ({:?} sec/key)",
        bench_keys0.len(),
        delta,
        delta / (bench_keys0.len() as f64)
    );
    
    println!("Generating polynomials...");
    let start2 = Instant::now();
    let (bench_poly0, bench_poly1) = generate_poly_shares(&cfg);
    let delta2 = start2.elapsed().as_secs_f64();  // corrected to use start2
    println!(
        "Generated {:?} polynomials in {:?} seconds ({:?} sec/polynomial)",
        bench_poly0.len(),
        delta2,
        delta2 / (bench_poly0.len() as f64)
    );
        

    if cfg.problem.as_str() == "known"{

        let aug_len = 4; //make larger
        if cfg.distribution.as_str() == "zipf"{
            println!("Zipf distribution sampling...");
            let strings = generate_strings(&cfg, aug_len);
            println!("Generated {:?} samples", strings.len());


            reset_servers(&mut client0, &mut client1).await?;

            let mut left_to_go = nreqs;
            let reqs_in_flight = 1000;
            while left_to_go > 0 {
                let mut resps = vec![];

                for _j in 0..reqs_in_flight {
                    let this_batch = std::cmp::min(left_to_go, cfg.addkey_batch_size);
                    left_to_go -= this_batch;

                    if this_batch > 0 {
                        resps.push(add_polynomials(
                            &cfg,
                            client0.clone(),
                            client1.clone(),
                            &strings,
                            this_batch,
                            aug_len
                        ));
                    }
                }

                for r in resps {
                    r.await?;
                }
            }
        }
        //tree_init(&mut client0, &mut client1).await?;


        let start = Instant::now();
        let mut active_paths = 0;

        let active_paths = run_level_last_known_poly(&cfg, &mut client0, &mut client1, nreqs, start).await?;
        println!(
            "Level {:?} active_paths={:?} {:?}",
            cfg.data_len,
            active_paths,
            start.elapsed().as_secs_f64()
        );

        println!("IN LEADER STOPPED PROGRESS HERE");

        final_shares(&mut client0, &mut client1).await?;

        Ok(())
    }
    else{
        let aug_len = 2;
        if cfg.distribution.as_str() == "zipf"{
            println!("Zipf distribution sampling...");
            let strings = generate_strings(&cfg, aug_len);
            println!("Generated {:?} samples", strings.len());


            reset_servers(&mut client0, &mut client1).await?;

            let mut left_to_go = nreqs;
            let reqs_in_flight = 1000;
            while left_to_go > 0 {
                let mut resps = vec![];

                for _j in 0..reqs_in_flight {
                    let this_batch = std::cmp::min(left_to_go, cfg.addkey_batch_size);
                    left_to_go -= this_batch;

                    if this_batch > 0 {
                        resps.push(add_polynomials_unknown(
                            &cfg,
                            client0.clone(),
                            client1.clone(),
                            &strings,
                            this_batch,
                            aug_len
                        ));
                    }
                }

                for r in resps {
                    r.await?;
                }
            }
        }
        tree_init(&mut client0, &mut client1).await?;


        let start = Instant::now();
        let mut active_paths = 0;
        for level in 0..cfg.data_len-1 {
            active_paths = run_level(&cfg, &mut client0, &mut client1, level, nreqs, start).await?;

            println!(
                "Level {:?} {:?}",
                level,
                start.elapsed().as_secs_f64()
            );
        }

        let active_paths = run_level_last(&cfg, &mut client0, &mut client1, nreqs, start).await?;
        println!(
            "Level {:?} active_paths={:?} {:?}",
            cfg.data_len,
            active_paths,
            start.elapsed().as_secs_f64()
        );

        final_shares(&mut client0, &mut client1).await?;

        Ok(())
    }
}