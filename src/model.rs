use std::ffi::CString;
use std::path::{Path, PathBuf};

use ndarray::{ArrayView, ArrayViewMut, Zip};

use super::{Error, Image, TextParams, VisionParams};

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum Verbosity {
    Minimum = 0,
    Default = 1,
    Maximum = 2,
}

impl Default for Verbosity {
    fn default() -> Self {
        Self::Default
    }
}

pub struct ModelBuilder {
    verbosity: Verbosity,
    path: PathBuf,
    threads: i32,
}

impl ModelBuilder {
    pub fn verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = verbosity;
        self
    }

    pub fn threads(mut self, threads: usize) -> Self {
        self.threads = threads as _;
        self
    }

    pub fn build(self) -> Result<Model, Error> {
        if !self.path.exists() {
            return Err(Error::PathNotFound);
        }
        let path = {
            use std::os::unix::ffi::OsStrExt;
            CString::new(self.path.as_os_str().as_bytes()).unwrap()
        };
        let ctx = unsafe { clip_cpp_sys::clip_model_load(path.as_ptr(), self.verbosity as i32) };
        let ctx = match std::ptr::NonNull::new(ctx) {
            Some(ctx) => ctx,
            None => return Err(Error::ModelFail),
        };

        let text_params = unsafe { *clip_cpp_sys::clip_get_text_hparams(ctx.as_ptr()) };
        let vision_params = unsafe { *clip_cpp_sys::clip_get_vision_hparams(ctx.as_ptr()) };
        let mean = unsafe {
            let mean = clip_cpp_sys::clip_get_image_mean(ctx.as_ptr());
            std::slice::from_raw_parts(mean, 3).to_vec()
        };
        let std = unsafe {
            let std = clip_cpp_sys::clip_get_image_std(ctx.as_ptr());
            std::slice::from_raw_parts(std, 3).to_vec()
        };

        Ok(Model {
            ctx,
            text_params: text_params.into(),
            vision_params: vision_params.into(),
            threads: self.threads,
            mean,
            std,
        })
    }
}

#[derive(Debug)]
pub struct Tokens {
    tokens: clip_cpp_sys::clip_tokens,
}

impl AsRef<clip_cpp_sys::clip_tokens> for Tokens {
    fn as_ref(&self) -> &clip_cpp_sys::clip_tokens {
        &self.tokens
    }
}

#[derive(Debug)]
pub struct Blob {
    image: clip_cpp_sys::clip_image_f32,
    _data: Vec<f32>,
}

impl AsRef<clip_cpp_sys::clip_image_f32> for Blob {
    fn as_ref(&self) -> &clip_cpp_sys::clip_image_f32 {
        &self.image
    }
}

pub struct Model {
    ctx: std::ptr::NonNull<clip_cpp_sys::clip_ctx>,
    threads: i32,
    text_params: TextParams,
    vision_params: VisionParams,
    mean: Vec<f32>,
    std: Vec<f32>,
}

unsafe impl Send for Model {}

impl Model {
    pub fn builder<P: AsRef<Path>>(model_path: P) -> ModelBuilder {
        ModelBuilder {
            verbosity: Verbosity::default(),
            threads: 1,
            path: PathBuf::from(model_path.as_ref()),
        }
    }

    pub fn text_params(&self) -> &TextParams {
        &self.text_params
    }

    pub fn vision_params(&self) -> &VisionParams {
        &self.vision_params
    }

    pub fn tokenize<T: AsRef<str>>(&self, text: T) -> Result<Tokens, Error> {
        let mut tokens: clip_cpp_sys::clip_tokens = unsafe { std::mem::zeroed() };

        let text = CString::new(text.as_ref())
            .expect("Failed to convert text into cstring: contained null byte?");
        unsafe {
            if !clip_cpp_sys::clip_tokenize(self.ctx.as_ptr(), text.as_ptr(), &mut tokens) {
                return Err(Error::Tokenize);
            }
        }

        Ok(Tokens { tokens })
    }

    pub fn encode_tokens(&self, tokens: &Tokens, normalize: bool) -> Vec<f32> {
        let mut encode = vec![0f32; self.vision_params.projection_dim() as usize];
        unsafe {
            clip_cpp_sys::clip_text_encode(
                self.ctx.as_ptr(),
                self.threads,
                &tokens.tokens,
                encode.as_mut_ptr(),
                normalize,
            );
        }

        encode
    }

    pub fn encode_text<T: AsRef<str>>(&self, text: T, normalize: bool) -> Result<Vec<f32>, Error> {
        let tokens = self.tokenize(text)?;
        Ok(self.encode_tokens(&tokens, normalize))
    }

    pub fn preprocess_image<I: Image>(&self, image: I) -> Result<Blob, Error> {
        let mat = ArrayView::from_shape(
            (image.height() as usize, image.width() as usize, 3),
            image.data(),
        )
        .unwrap();

        let mut dest = unsafe {
            let mut v = Vec::<f32>::with_capacity(image.data().len());
            v.set_len(image.data().len());
            v
        };

        let dest_av = ArrayViewMut::from_shape(
            (image.height() as usize, image.width() as usize, 3),
            &mut dest,
        )
        .unwrap();

        Zip::indexed(dest_av)
            .and(&mat)
            .for_each(|(_, _, c), im, mat| {
                *im = ((*mat as f32) / 255.0 - self.mean[c]) / self.std[c]
            });

        let cimage = clip_cpp_sys::clip_image_f32 {
            nx: image.width() as _,
            ny: image.height() as _,
            size: image.size(),
            data: dest.as_mut_ptr(),
        };

        Ok(Blob {
            image: cimage,
            _data: dest,
        })
    }

    pub fn encode_image(&self, blob: &Blob, normalize: bool) -> Vec<f32> {
        let image_size = self.vision_params.image_size();
        if blob.image.nx != image_size || blob.image.ny != image_size {
            panic!(
                "invalid image size, expected ({image_size}x{image_size}) found ({}x{})",
                blob.image.nx, blob.image.ny
            );
        }
        let mut encode = vec![0f32; self.vision_params.projection_dim() as usize];
        unsafe {
            clip_cpp_sys::clip_image_encode(
                self.ctx.as_ptr(),
                self.threads,
                &blob.image as *const _ as *mut _,
                encode.as_mut_ptr(),
                normalize,
            );
        }

        encode
    }

    pub fn preprocess_images<T>(&self, images: T) -> Result<Vec<Blob>, Error>
    where
        T: IntoIterator,
        T::Item: Image,
    {
        let blobs = images
            .into_iter()
            .map(|i| self.preprocess_image(i))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(blobs)
    }

    pub fn encode_images<'a, T: IntoIterator<Item = &'a Blob>>(
        &self,
        images: T,
        normalize: bool,
    ) -> Vec<Vec<f32>> {
        let image_size = self.vision_params.image_size();
        let mut images = images
            .into_iter()
            .map(|blob| {
                if blob.image.nx != image_size || blob.image.ny != image_size {
                    panic!(
                        "invalid image size, expected ({image_size}x{image_size}) found ({}x{})",
                        blob.image.nx, blob.image.ny
                    );
                }
                blob.image
            })
            .collect::<Vec<_>>();
        let mut encode = vec![0f32; images.len() * (self.vision_params.projection_dim() as usize)];

        let input_img_batch = clip_cpp_sys::clip_image_f32_batch {
            data: images.as_mut_ptr(),
            size: images.len(),
        };

        unsafe {
            clip_cpp_sys::clip_image_batch_encode(
                self.ctx.as_ptr(),
                self.threads,
                &input_img_batch,
                encode.as_mut_ptr(),
                normalize,
            );
        }

        encode
            .chunks(self.vision_params.projection_dim() as usize)
            .into_iter()
            .map(|v| v.to_owned())
            .collect()
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            clip_cpp_sys::clip_free(self.ctx.as_ptr());
        }
    }
}
