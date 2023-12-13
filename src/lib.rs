use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("model path did not exist")]
    PathNotFound,
    #[error("model failed to load")]
    ModelFail,
    #[error("failed to tokenize text")]
    Tokenize,
    #[error("failed to preprocess image")]
    Preprocess,
}

mod image;
mod model;
mod params;

pub use self::image::{Image, RGBImage};
pub use model::Model;
pub use params::{TextParams, VisionParams};
