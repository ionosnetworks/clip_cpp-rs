pub struct TextParams {
    params: clip_cpp_sys::clip_text_hparams,
}

impl TextParams {
    pub fn vocab(&self) -> i32 {
        self.params.n_vocab
    }

    pub fn positions(&self) -> i32 {
        self.params.num_positions
    }

    pub fn hidden_size(&self) -> i32 {
        self.params.hidden_size
    }

    pub fn intermediate(&self) -> i32 {
        self.params.n_intermediate
    }

    pub fn projection_dim(&self) -> i32 {
        self.params.projection_dim
    }

    pub fn head(&self) -> i32 {
        self.params.n_head
    }

    pub fn layer(&self) -> i32 {
        self.params.n_layer
    }

    pub fn eps(&self) -> f32 {
        self.params.eps
    }
}

impl From<clip_cpp_sys::clip_text_hparams> for TextParams {
    fn from(params: clip_cpp_sys::clip_text_hparams) -> Self {
        Self { params }
    }
}

pub struct VisionParams {
    params: clip_cpp_sys::clip_vision_hparams,
}

impl VisionParams {
    pub fn image_size(&self) -> i32 {
        self.params.image_size
    }

    pub fn patch_size(&self) -> i32 {
        self.params.patch_size
    }

    pub fn hidden_size(&self) -> i32 {
        self.params.hidden_size
    }

    pub fn intermediate(&self) -> i32 {
        self.params.n_intermediate
    }

    pub fn projection_dim(&self) -> i32 {
        self.params.projection_dim
    }

    pub fn head(&self) -> i32 {
        self.params.n_head
    }

    pub fn layer(&self) -> i32 {
        self.params.n_layer
    }

    pub fn eps(&self) -> f32 {
        self.params.eps
    }
}

impl From<clip_cpp_sys::clip_vision_hparams> for VisionParams {
    fn from(params: clip_cpp_sys::clip_vision_hparams) -> Self {
        Self { params }
    }
}
