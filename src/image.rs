pub trait Image {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn size(&self) -> usize;
    fn data(&self) -> &[u8];
}

pub struct RGBImage {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl RGBImage {
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Self {
        Self {
            width,
            height,
            data,
        }
    }
}

impl Image for &RGBImage {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn data(&self) -> &[u8] {
        self.data.as_ref()
    }
}
