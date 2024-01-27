use clip_cpp_rs as clip;

fn score(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}

fn main() {
    let model_path = "./models/clip-vit-base-patch32_ggml-model-q4_1.gguf";
    let img_path = "./red_apple.jpg";
    let text = "an apple";

    let model = clip::Model::builder(model_path)
        .build()
        .expect("Failed to build model");

    let img = image::open(img_path).expect("Failed to open image");
    let image_size = model.vision_params().image_size();
    let img = img.resize_exact(image_size as _, image_size as _, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();

    let img = clip::RGBImage::new(img.width(), img.height(), img.into_vec());

    let blob = model.preprocess_image(&img).expect("Failed to preprocess");
    let tokens = model.tokenize(text).expect("Failed to tokenize");
    let v = model.encode_tokens(&tokens, false);
    // let v = model.encode_text(text, false);
    let z = model.encode_image(&blob, false);

    println!("score: {:?}", score(&v, &z));
}
