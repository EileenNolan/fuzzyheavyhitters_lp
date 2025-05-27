// Starter code from:
//   https://github.com/google/tarpc/blob/master/example-service/src/server.rs

use counttree::{
    collect, config,
    FieldElm,
    fastfield::FE, prg,
    rpc::Collector,
    rpc::{
        AddKeysRequest, PolyRequest, FinalSharesRequest, ResetRequest, TreeCrawlRequest, TreeInitRequest,
        TreePruneRequest,
        TreePruneLastRequest,
    },
};

use futures::{
    future::{self, Ready},
    prelude::*,
};
use std::{
    io,
    sync::{Arc, Mutex},
};
use std::convert::TryFrom;
use std::io::{BufReader, BufWriter};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::thread::available_parallelism;
use std::time::Duration;
use tarpc::{
    context,
    server::{self, Channel},
    tokio_serde::formats::Bincode,
    serde_transport::tcp,
};
use counttree::rpc::TreeCrawlLastRequest;

extern crate num_cpus;
// type MyChannel = scuttlebutt::SyncChannel<BufReader<UnixStream>, BufWriter<UnixStream>>;
type MyChannel = scuttlebutt::SyncChannel<BufReader<TcpStream>, BufWriter<TcpStream>>;

pub type Poly = Vec<Vec<FieldElm>>;
// pub type PolyPair = (Poly, Poly);


#[derive(Clone)]
struct CollectorServer {
    seed: prg::PrgSeed,
    data_len: usize,
    server_idx: u16,
    arc: Arc<Mutex<collect::KeyCollection<FE, FieldElm>>>,
    // gc_channel: Option<Arc<Mutex<MyChannel>>>
    gc_channels: Vec<Arc<Mutex<MyChannel>>>
}

impl Collector for CollectorServer {
    type AddKeysFut = Ready<String>;
    type AddPolynomialsFut = Ready<String>; 
    type TreeInitFut = Ready<String>;
    type TreeCrawlFut = Ready<Vec<FE>>;
    type TreeCrawlLastFut = Ready<Vec<FieldElm>>;
    type RunLevelLastKnownPolyFut = Ready<Vec<FieldElm>>; //added
    type TreePruneFut = Ready<String>;
    type TreePruneLastFut = Ready<String>;
    type FinalSharesFut = Ready<Vec<collect::Result<FieldElm>>>;
    type ResetFut = Ready<String>;

    fn reset(self, _: context::Context, _rst: ResetRequest) -> Self::ResetFut {
        let mut coll = self.arc.lock().unwrap();
        *coll = collect::KeyCollection::new(&self.seed, self.data_len);

        future::ready("Done".to_string())
    }

    fn add_keys(self, _: context::Context, add: AddKeysRequest) -> Self::AddKeysFut {
        let mut coll = self.arc.lock().unwrap();
        for k in add.keys {
            coll.add_key(k);
        }
        future::ready("".to_string())
    }

    // fn add_polynomials(self, _: context::Context, poly_req: PolyRequest) -> Self::AddPolynomialsFut {
    //     let mut coll = self.arc.lock().unwrap();
    //     for p in poly_req.poly {
    //         coll.add_polynomial(p);
    //     }
    //     future::ready("".to_string())
    // }
    fn add_polynomials(self, _: context::Context, poly_req: PolyRequest) -> Self::AddPolynomialsFut {
        println!("Client: Received add_polynomials request with {} polynomials.", poly_req.poly.len());
        // Try to lock the mutex.
        let mut coll = match self.arc.lock() {
            Ok(guard) => guard,
            Err(e) => {
                eprintln!("Client: Mutex lock failed: {:?}", e);
                // Instead of returning Err(...), return an error message as a String.
                return future::ready("Mutex lock failed".to_string());
            }
        };
    
        for p in poly_req.poly {
            coll.add_polynomial(p);
        }
        println!("Client: Updated collector; total polynomials now: {}", coll.poly.len());
    
        // Return a success string (here an empty string).
        future::ready("".to_string())
    }

    fn add_polynomials_unknown(self, _: context::Context, poly_req: PolyRequestU2) -> Self::AddPolynomialsFut {
        println!("Client: Received add_polynomials request with {} polynomials.", poly_req.poly.len());
        // Try to lock the mutex.
        let mut coll = match self.arc.lock() {
            Ok(guard) => guard,
            Err(e) => {
                eprintln!("Client: Mutex lock failed: {:?}", e);
                // Instead of returning Err(...), return an error message as a String.
                return future::ready("Mutex lock failed".to_string());
            }
        };
    
        for p in poly_req.poly {
            coll.add_polynomial(p);
        }
        println!("Client: Updated collector; total polynomials now: {}", coll.poly.len());
    
        // Return a success string (here an empty string).
        future::ready("".to_string())
    }
    
    
    
    fn tree_init(self, _: context::Context, _req: TreeInitRequest) -> Self::TreeInitFut {
        let mut coll = self.arc.lock().unwrap();
        coll.tree_init();
        future::ready("Done".to_string())
    }
    fn tree_crawl(
        self,
        _: context::Context,
        req: TreeCrawlRequest
    ) -> Self::TreeCrawlFut {
        let mut coll = self.arc.lock().unwrap();

        // Lock all channels
        let mut locked_channels: Vec<_> = self.gc_channels
            .iter()
            .map(|c| c.lock().unwrap())
            .collect();

        // Get mutable references to inner channels
        let mut channel_refs: Vec<&mut MyChannel> = locked_channels
            .iter_mut()
            .map(|guard| &mut **guard)
            .collect();

        let results = coll.tree_crawl(req.gc_sender, &mut channel_refs[..]);

        future::ready(results)
    }
    fn tree_crawl_last(
        self,
        _: context::Context,
        req: TreeCrawlLastRequest
    ) -> Self::TreeCrawlLastFut {
        let mut coll = self.arc.lock().unwrap();

        // Lock all channels
        let mut locked_channels: Vec<_> = self.gc_channels
            .iter()
            .map(|c| c.lock().unwrap())
            .collect();

        // Get mutable references to inner channels
        let mut channel_refs: Vec<&mut MyChannel> = locked_channels
            .iter_mut()
            .map(|guard| &mut **guard)
            .collect();

        let results = coll.tree_crawl_last(req.gc_sender, &mut channel_refs[..]);

        future::ready(results)
    }

    fn run_level_last_known_poly(
        self,
        _: context::Context,
        req: TreeCrawlLastRequest
    ) -> Self::TreeCrawlLastFut {
        let mut coll = self.arc.lock().unwrap();
        println!("pub fn run_level_last_known_poly");
        // Lock all channels
        let mut locked_channels: Vec<_> = self.gc_channels
            .iter()
            .map(|c| c.lock().unwrap())
            .collect();

        // Get mutable references to inner channels
        let mut channel_refs: Vec<&mut MyChannel> = locked_channels
            .iter_mut()
            .map(|guard| &mut **guard)
            .collect();

        let results = coll.tree_crawl_last_known_poly(req.gc_sender, &mut channel_refs[..]);
        //let field_result = FieldElm::from_u64(results as u64);

        //future::ready(vec![results])

        future::ready(results)  
    }

    fn tree_prune(self, _: context::Context, req: TreePruneRequest) -> Self::TreePruneFut {
        let mut coll = self.arc.lock().unwrap();
        coll.tree_prune(&req.keep);
        future::ready("Done".to_string())
    }

    fn tree_prune_last(self, _: context::Context, req: TreePruneLastRequest) -> Self::TreePruneLastFut {
        let mut coll = self.arc.lock().unwrap();
        coll.tree_prune_last(&req.keep);
        future::ready("Done".to_string())
    }

    fn final_shares(self, _: context::Context, _req: FinalSharesRequest) -> Self::FinalSharesFut {
        let coll = self.arc.lock().unwrap();
        let out = coll.final_shares();
        future::ready(out)
    }

    // fn evaluate_polynomials(&self, poly: &Vec<Vec<FieldElm>>) -> Vec<FieldElm> {
    //     // Example server dictionary; ideally, you'd store this as part of configuration.
    //     let w = vec![5u64, 10];  
    //     let mut evaluations = Vec::with_capacity(poly.len());
    //     for i in 0..poly.len() {
    //         // Optionally, you might hash w[i] to a field element; here we convert directly.
    //         let key_i = FieldElm::from(w[i]);
    //         // Evaluate the polynomial for coordinate i at key_i.
    //         let x_i = evaluate_polynomial(&poly[i], &key_i);
    //         evaluations.push(x_i);
    //     }
    //     evaluations
    // }
}

fn create_server_tcp_socket(port: u16) -> io::Result<MyChannel> {
    let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], port)));
    let (stream, _) = listener.unwrap().accept()?;
    stream.set_nodelay(true)?;

    Ok(scuttlebutt::SyncChannel::new(
        BufReader::new(stream.try_clone()?),
        BufWriter::new(stream),
    ))
}

fn setup_tcp_sockets(
    server_idx: u16,
    num_cpus: usize,
    server0_addr: SocketAddr,
    server1_addr: SocketAddr,
) -> io::Result<Vec<Arc<Mutex<MyChannel>>>> {
    let mut channels = Vec::with_capacity(num_cpus);
    let base_port = server1_addr.port(); // Use the port from the provided address

    for i in 0..num_cpus {
        let port = base_port + i as u16;
        let channel_result = if server_idx == 0 {
            // Garbler (client) side - connect to server1
            let target_addr = SocketAddr::new(server1_addr.ip(), port);
            connect_with_retries_tcp(target_addr)
        } else {
            // Evaluator (server) side - listen for connections from server0
            create_server_tcp_socket(port)
        };

        let channel = channel_result?;
        channels.push(Arc::new(Mutex::new(channel)));
    }

    Ok(channels)
}

fn connect_with_retries_tcp(addr: SocketAddr) -> io::Result<MyChannel> {
    let mut retries = 0;
    let mut last_error = None;

    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                return Ok(scuttlebutt::SyncChannel::new(
                    BufReader::new(stream.try_clone()?),
                    BufWriter::new(stream),
                ));
            }
            Err(e) => {
                last_error = Some(e);
                if retries >= 10 {
                    return Err(io::Error::new(
                        io::ErrorKind::ConnectionRefused,
                        format!("Failed to connect to {} after {} retries: {:?}",
                                addr, 10, last_error)
                    ));
                }
                retries += 1;
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
}


#[tokio::main]
async fn main() -> io::Result<()> {
    println!("Server main() started.");
    env_logger::init();
    println!("Server starting...");

    let (cfg, sid, _) = config::get_args("Server", true, false);
    println!("Configuration loaded. Server id: {}", sid);
    let server_addr = match sid {
        0 => cfg.server0,
        1 => cfg.server1,
        _ => panic!("Oh no!"),
    };
    println!("Server will bind to: {:?}", server_addr);

    let server_idx = match sid {
        0 => 0,
        1 => 1,
        _ => panic!("Oh no!"),
    };

    // XXX This is bogus
    let seed = prg::PrgSeed { key: [1u8; 16] };

    let coll = collect::KeyCollection::new(&seed, cfg.data_len);
    let arc = Arc::new(Mutex::new(coll));

    let num_cpus = available_parallelism().unwrap().get();

    let gc_channels = setup_tcp_sockets(server_idx, num_cpus, cfg.server0, cfg.server1).unwrap_or_else(|e| {
        eprintln!("Warning: Failed to setup GC channels: {}", e);
        vec![] // Fallback to no channels
    });

    let mut server_addr = server_addr;
    // Listen on any IP
    server_addr.set_ip("0.0.0.0".parse().expect("Could not parse"));

    println!("Starting TCP listener on: {}", server_addr);
    tcp::listen(&server_addr, Bincode::default)
        .await?
        .filter_map(|r| future::ready(r.ok()))
        .map(server::BaseChannel::with_defaults)
        .map(|channel| {
            let coll_server = CollectorServer {
                server_idx,
                seed: seed.clone(),
                data_len: cfg.data_len,
                arc: arc.clone(),
                gc_channels: gc_channels.clone(),
            };

            channel.execute(coll_server.serve())
        })
        .buffer_unordered(100)
        .for_each(|_| async {})
        .await;

    Ok(())
}
