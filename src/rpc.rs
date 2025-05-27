use crate::collect;
use crate::FieldElm;
use crate::fastfield::FE;

use serde::Deserialize;
use serde::Serialize;
use crate::ibDCF::ibDCFKey;

pub type Poly = Vec<Vec<FieldElm>>;

pub type PolyU2 = Vec<Vec<Vec<FieldElm>>>;

pub type PolyPair = (Poly, Poly);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResetRequest {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AddKeysRequest {
    pub keys: Vec<Vec<(ibDCFKey, ibDCFKey)>>,
}

// *** lp addition
// #[derive(Clone, Debug, Serialize, Deserialize)]
// pub struct PolyRequest {
//     pub poly: Vec<Vec<FieldElm>>,
// }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyRequest {
    pub poly: Vec<Poly>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyRequestU2 {
    pub poly: Vec<PolyU2>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyResponse {
    pub share: FieldElm,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeInitRequest {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeCrawlRequest {
    pub gc_sender: bool
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeCrawlLastRequest {
    pub gc_sender: bool
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreePruneRequest {
    pub keep: Vec<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreePruneLastRequest {
    pub keep: Vec<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeSketchFrontierRequest {
    pub level: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TreeSketchFrontierLastRequest {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalSharesRequest {}


#[tarpc::service]
pub trait Collector {
    async fn reset(rst: ResetRequest) -> String;
    async fn add_keys(add: AddKeysRequest) -> String;
    async fn add_polynomials(poly_req: PolyRequest) -> String; // Added
    async fn add_polynomials_unknown(poly_req: PolyRequestU2) -> String; // Added
    async fn tree_init(req: TreeInitRequest) -> String;
    async fn tree_crawl(req: TreeCrawlRequest) -> Vec<FE>;
    async fn tree_crawl_last(req: TreeCrawlLastRequest) -> Vec<FieldElm>;
    async fn run_level_last_known_poly(req: TreeCrawlLastRequest) -> Vec<FieldElm>; //added
    async fn tree_prune(req: TreePruneRequest) -> String;
    async fn tree_prune_last(req: TreePruneLastRequest) -> String;
    async fn final_shares(req: FinalSharesRequest) -> Vec<collect::Result<FieldElm>>;

    // // *** lp add in
    // type ProcessPolyFut = Ready<PolyResponse>;
    // fn process_poly(self, _: context::Context, req: PolyRequest) -> Self::ProcessPolyFut;
}
