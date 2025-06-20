use clap::{App, Arg};
use serde_json::Value;
use std::{fs, net::SocketAddr};

pub struct Config {
    pub data_len: usize,
    pub n_dims: usize,
    pub ball_size: usize,
    pub addkey_batch_size: usize,
    pub num_sites: usize,
    pub threshold: f64,
    pub zipf_exponent: f64,
    pub server0: SocketAddr,
    pub server1: SocketAddr,
    pub distribution: String,
    pub problem: String
}

fn parse_ip(v: &Value, error_msg: &str) -> SocketAddr {
    v.as_str().expect(error_msg).parse().expect(error_msg)
}

pub fn get_config(filename: &str) -> Config {
    let json_data = &fs::read_to_string(filename).expect("Cannot open JSON file");
    let v: Value = serde_json::from_str(json_data).expect("Cannot parse JSON config");

    let data_len: usize = v["data_len"].as_u64().expect("Can't parse data_len") as usize;
    let n_dims: usize = v["n_dims"].as_u64().expect("Can't parse data_len") as usize;
    let ball_size: usize = v["ball_size"].as_u64().expect("Can't parse data_len") as usize;
    let addkey_batch_size: usize = v["addkey_batch_size"]
        .as_u64()
        .expect("Can't parse addkey_batch_size") as usize;
    let num_sites: usize = v["num_sites"].as_u64().expect("Can't parse num_sites") as usize;
    let threshold = v["threshold"].as_f64().expect("Can't parse threshold");
    let zipf_exponent = v["zipf_exponent"]
        .as_f64()
        .expect("Can't parse zipf_exponent");
    let server0 = parse_ip(&v["server0"], "Can't parse server0 addr");
    let server1 = parse_ip(&v["server1"], "Can't parse server1 addr");
    let distribution: String = v["distribution"].as_str().expect("Can't parse distribution").to_string();
    let problem: String = v["problem"].as_str().expect("Can't parse distribution").to_string();

    Config {
        data_len,
        n_dims,
        ball_size,
        addkey_batch_size,
        num_sites,
        threshold,
        zipf_exponent,
        server0,
        server1,
        distribution,
        problem
    }
}

pub fn get_args(name: &str, get_server_id: bool, get_n_reqs: bool) -> (Config, i8, usize) {
    let mut flags = App::new(name)
        .version("0.1")
        .author("Henry Corrigan-Gibbs <henrycg@csail.mit.edu>")
        .about("Prototype of privacy-preserving heavy hitters scheme.")
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .value_name("FILENAME")
                .help("Location of JSON config file")
                .required(true)
                .takes_value(true),
        );

    if get_server_id {
        flags = flags.arg(
            Arg::with_name("server_id")
                .short("i")
                .long("server_id")
                .value_name("NUMBER")
                .help("Zero-indexed ID of server")
                .required(true)
                .takes_value(true),
        );
    }

    if get_n_reqs {
        flags = flags.arg(
            Arg::with_name("num_requests")
                .short("n")
                .long("num_requests")
                .value_name("NUMBER")
                .help("Number of client requests to generate")
                .required(true)
                .takes_value(true),
        );
    }

    let flags = flags.get_matches();

    let mut server_id = -1;
    if get_server_id {
        server_id = flags.value_of("server_id").unwrap().parse().unwrap();
    }

    let mut n_reqs = 0;
    if get_n_reqs {
        n_reqs = flags.value_of("num_requests").unwrap().parse().unwrap();
    }

    (
        get_config(flags.value_of("config").unwrap()),
        server_id,
        n_reqs,
    )
}
