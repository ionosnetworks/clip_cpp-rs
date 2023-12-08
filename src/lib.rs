use std::{error::Error, ffi::CString, vec};

use clip_cpp_sys::{
    clip_ctx, clip_free, clip_get_vision_hparams, clip_image_batch_encode,
    clip_image_batch_preprocess, clip_image_f32, clip_image_f32_batch, clip_image_load_from_file,
    clip_image_u8, clip_image_u8_batch, clip_model_load, clip_text_encode, clip_tokenize,
    clip_tokens,
};

pub struct Model {
    ctx: std::ptr::NonNull<clip_ctx>,
    n_threads: i32,
}

unsafe impl Send for Model {}

impl Model {
    pub fn new<T: AsRef<str>>(path: T, n_threads: Option<i32>) -> Result<Self, Box<dyn Error>> {
        // @Todo: check to make sure path exists before proceeding so there's no seg fault
        let fname = CString::new(path.as_ref())?; // @Todo: better error
        let verbosity = 1; // @Todo: set based on log level
        let ctx = unsafe { clip_model_load(fname.as_ptr(), verbosity) };
        let ctx = match std::ptr::NonNull::new(ctx) {
            Some(ctx) => ctx,
            None => return Err("failed to load model".into()),
        };

        let n_threads = n_threads.unwrap_or(1);

        Ok(Self { ctx, n_threads })
    }

    pub fn set_n_threads(&mut self, n_threads: i32) {
        self.n_threads = n_threads;
    }

    pub fn projection_dim(&self) -> Option<i32> {
        let v = unsafe { clip_get_vision_hparams(self.ctx.as_ptr()).as_ref() };
        v.as_ref().map(|v| v.projection_dim)
    }

    pub fn batch_image_preprocess(
        &self,
        imgs: &mut [&mut Image<u8>],
    ) -> Result<Vec<Image<f32>>, Box<dyn Error>> {
        let mut preproc_imgs = (0..imgs.len())
            .map(|_| clip_image_f32 {
                nx: 0,
                ny: 0,
                data: std::ptr::null_mut(),
                size: 0,
            })
            .collect::<Vec<_>>();
        let mut preproc_img_batch = clip_image_f32_batch {
            data: preproc_imgs.as_mut_ptr(),
            size: 0,
        };
        let mut clip_imgs = imgs
            .iter_mut()
            .map(|i| clip_image_u8 {
                nx: i.nx,
                ny: i.ny,
                data: i.data.as_mut_ptr(),
                size: i.size,
            })
            .collect::<Vec<_>>();
        let input_img_batch = clip_image_u8_batch {
            data: clip_imgs.as_mut_ptr(),
            size: imgs.len(),
        };
        unsafe {
            clip_image_batch_preprocess(
                self.ctx.as_ptr(),
                self.n_threads,
                &input_img_batch,
                &mut preproc_img_batch,
            );
        }

        let preproc_imgs = unsafe {
            std::slice::from_raw_parts(preproc_img_batch.data, preproc_img_batch.size)
                .to_vec()
                .into_iter()
                .map(|img| Image {
                    nx: img.nx,
                    ny: img.ny,
                    data: Box::from_raw(std::slice::from_raw_parts_mut(img.data, img.size)),
                    size: img.size,
                })
                .collect::<Vec<_>>()
        };

        Ok(preproc_imgs)
    }

    pub fn batch_image_encode(
        &self,
        imgs: &mut [&mut Image<f32>],
    ) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
        let proj_dim = self.projection_dim().ok_or("no projection dim")? as usize;
        let mut embeddings = vec![0.; proj_dim * imgs.len()];

        let mut clip_imgs = imgs
            .iter_mut()
            .map(|i| clip_image_f32 {
                nx: i.nx,
                ny: i.ny,
                data: i.data.as_mut_ptr(),
                size: i.size,
            })
            .collect::<Vec<_>>();
        let input_img_batch = clip_image_f32_batch {
            data: clip_imgs.as_mut_ptr(),
            size: imgs.len(),
        };

        unsafe {
            clip_image_batch_encode(
                self.ctx.as_ptr(),
                self.n_threads,
                &input_img_batch,
                embeddings.as_mut_ptr(),
                true,
            );
        }

        Ok(embeddings
            .chunks(proj_dim)
            .into_iter()
            .map(|v| v.to_owned())
            .collect())
    }

    pub fn tokenize<T: AsRef<str>>(&self, text: T) -> Result<Tokens, Box<dyn Error>> {
        let mut tokens = clip_tokens {
            data: std::ptr::null_mut(),
            size: 0,
        };
        let text = CString::new(text.as_ref())?;
        unsafe {
            if !clip_tokenize(self.ctx.as_ptr(), text.as_ptr(), &mut tokens) {
                return Err("failed to tokenize text".into());
            }
        }

        Ok(Tokens { tokens })
    }

    pub fn text_encode(&self, tokens: &Tokens) -> Result<Vec<f32>, Box<dyn Error>> {
        let proj_dim = self.projection_dim().ok_or("no projection dim")? as usize;
        let normalize = true;
        let mut encode = vec![0f32; proj_dim];
        unsafe {
            clip_text_encode(
                self.ctx.as_ptr(),
                self.n_threads,
                tokens.clip_tokens(),
                encode.as_mut_ptr(),
                normalize,
            );
        }

        Ok(encode)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe {
            clip_free(self.ctx.as_ptr());
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image<T: num_traits::Num> {
    nx: ::std::os::raw::c_int,
    ny: ::std::os::raw::c_int,
    data: Box<[T]>,
    size: usize,
}

impl<T> std::default::Default for Image<T>
where
    T: num_traits::Num,
{
    fn default() -> Self {
        Self {
            nx: Default::default(),
            ny: Default::default(),
            data: unsafe { Box::from_raw(std::slice::from_raw_parts_mut(std::ptr::null_mut(), 0)) },
            size: Default::default(),
        }
    }
}

// impl From<&mut Image<u8>> for clip_image_u8 {
//     fn from(value: &mut Image<u8>) -> Self {
//         Self {
//             nx: value.nx,
//             ny: value.ny,
//             data: value.data.as_mut_ptr(),
//             size: value.size,
//         }
//     }
// }

// impl From<&mut Image<f32>> for clip_image_f32 {
//     fn from(value: &mut Image<f32>) -> Self {
//         Self {
//             nx: value.nx,
//             ny: value.ny,
//             data: value.data.as_mut_ptr(),
//             size: value.size,
//         }
//     }
// }

impl Image<u8> {
    pub fn new(img: &[u8], nx: i32, ny: i32) -> Result<Image<u8>, Box<dyn Error>> {
        Ok(Image {
            nx,
            ny,
            data: Box::from(&img[..]),
            size: (nx * ny) as usize,
        })
    }

    pub fn load_image_from_file<T: AsRef<str>>(path: T) -> Result<Image<u8>, Box<dyn Error>> {
        let path = CString::new(path.as_ref())?;
        let mut img = clip_image_u8 {
            nx: 0,
            ny: 0,
            data: std::ptr::null_mut(),
            size: 0,
        };
        unsafe {
            if !clip_image_load_from_file(path.as_ptr(), &mut img) {
                return Err("failed to load image".into());
            }
        }

        Ok(Image {
            nx: img.nx,
            ny: img.ny,
            data: unsafe { Box::from_raw(std::slice::from_raw_parts_mut(img.data, img.size)) },
            size: img.size,
        })
    }
}

#[derive(Debug)]
pub struct Tokens {
    tokens: clip_tokens,
}

impl Tokens {
    pub fn tokens(&self) -> &[i32] {
        unsafe { std::slice::from_raw_parts(self.tokens.data, self.tokens.size) }
    }

    pub fn clip_tokens(&self) -> &clip_tokens {
        return &self.tokens;
    }
}
