use anyhow::anyhow;
use clap::Parser;
use half::f16;
use num_traits::Float;
use std::str;

use yllama::llama::{Llama, LlamaParams};
use yllama::llm::{Instantiable, LLM};
use yloader::*;
use yloader::{load_build, load_fast, ModelFile};
use ymath::tensor::*;
use ymath::Matmul;

pub struct VIRTUALMEM; // Basically MacOS, Linux, Windows

impl<'a, T: Float + Matmul, const D0: usize, const D1: usize>
    Instantiable<VIRTUALMEM, (usize, String)> for Tensor<'a, true, T, M<D0, D1>, VecStore<T>>
{
    fn instantiate(_: (usize, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(TensorMut::new_matrix())
    }
}

impl<'a, T: Float + Matmul, const D0: usize> Instantiable<VIRTUALMEM, (usize, String)>
    for Tensor<'a, true, T, V<D0>, VecStore<T>>
{
    fn instantiate(_: (usize, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(TensorMut::new_vector())
    }
}

impl<'a, const D0: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, V<D0>, VecStore<f32>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, M<D0, D1>, VecStore<f32>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, M<D0, D1>, VecStore<f16>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize, const D1: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, M<D0, D1>, MmapStore<f32, f32>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, const D0: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, false, f32, V<D0>, MmapStore<f32, f32>>
{
    fn instantiate((model, name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let t = model.tensors.get(&name).expect("Tensor not found");
        Ok(t.to_tensor(model)
            .map_err(|_| anyhow!("GGUF tensor import error"))?)
    }
}

impl<'a, T: Float + Matmul, const D0: usize> Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, true, T, V<D0>, VecStore<T>>
{
    fn instantiate((_model, _name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(Tensor::new_vector())
    }
}

impl<'a, T: Float + Matmul, const D0: usize, const D1: usize>
    Instantiable<VIRTUALMEM, (&'a ModelFile, String)>
    for Tensor<'a, true, T, M<D0, D1>, VecStore<T>>
{
    fn instantiate((_model, _name): (&'a ModelFile, String)) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        Ok(Tensor::new_matrix())
    }
}

impl Instantiable<VIRTUALMEM, &ModelFile> for LlamaParams<f32> {
    fn instantiate(model: &ModelFile) -> Result<Self, anyhow::Error>
    where
        Self: Sized,
    {
        let header = &model.header;
        let embedding_length = header_find_usize(header, "llama.embedding_length")?;
        let attention_head_count_kv = header_find_usize(header, "llama.attention.head_count_kv")?;
        let attention_head_count = header_find_usize(header, "llama.attention.head_count")?;
        let params: LlamaParams<f32> = LlamaParams {
            block_count: header_find_usize(header, "llama.block_count")?,
            _context_length: header_find_usize(header, "llama.context_length")?,
            embedding_length,
            feed_forward_length: header_find_usize(header, "llama.feed_forward_length")?,
            attention_head_count,
            attention_head_count_kv,
            attention_layer_norm_rms_epsilon: header_find_f32(
                header,
                "llama.attention.layer_norm_rms_epsilon",
            )?,
            rope_freq_base: header_find_f32(header, "llama.rope.freq_base")?,
            _rope_dimension_count: header_find_usize(header, "llama.rope.dimension_count")?,
            vocab_size: header_find_usize(header, "llama.vocab_size")?,
            _max_seq_len: header_find_usize(header, "llama.context_length")?,
            _attention_kv_length: embedding_length * attention_head_count_kv / attention_head_count,
        };
        Ok(params)
    }
}

fn process(
    path: &str,
    tokenizer_path: &str,
    prompt: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let (arch, name, gguf) = load_fast(path)?;

    println!("Architecture == {}", arch);
    println!("Name == '{}'", name);

    match arch.as_str() {
        "llama" => {
            let model = load_build(path, gguf)?;
            type A = MmapStore<f32, f32>;
            //type B = MmapStore<f32, f16>;
            type C = VecStore<f32>;
            type D = VecStore<f16>;
            let typ = yllama::llama::llama_find_type(&model)?;
            const EMBED: usize = 2048;
            const VOCAB: usize = 128256;
            const FF: usize = 8192;
            const KV: usize = 512;
            const CONTEXT: usize = 2048;
            const FREQ: usize = 32;
            match typ {
                "F16" => {
                    type LlamaType<'a> = Llama<
                        'a,
                        VIRTUALMEM, //TA
                        ModelFile,  //D
                        f32,        //T
                        D,          //TokenEmbd
                        C,          //RoPE
                        C,          //OutputNorm
                        D,          //AttnK
                        D,          //AttnQ
                        D,          //AttnV
                        C,          //AttnNorm
                        D,          //FfnDown
                        D,          //FfnGate
                        C,          //FfnNorm
                        D,          //FfnUp
                        D,          //AttnOutput
                        EMBED,
                        VOCAB,
                        FF,
                        KV,
                        CONTEXT,
                        FREQ,
                    >;
                    let mut runnable: LlamaType = Llama::instantiate((&model, tokenizer_path))?;
                    unsafe { runnable.run(prompt) }
                }

                _ => Err(anyhow!("Unknown configuration").into()),
            }
        }
        // "gpt" => {
        //     let model = load_build(path, gguf)?;
        //     let runnable: Gpt = LLM::build(&model, tokenizer_path)?;
        //     run(runnable, prompt)
        // }
        _ => anyhow::Result::Err(anyhow!("Unsupported architecture"))?,
    }?;
    Ok(())
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    file: String,

    #[arg(short, long)]
    tokenizer: String,

    #[arg(short, long)]
    prompt: String,

    #[arg(short, long, default_value_t = false)]
    clone: bool,

    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    #[arg(long, default_value_t = 0.7)]
    temp: f32,

    #[arg(short, long)]
    seed: Option<u64>,
}

fn main() {
    let args = Args::parse();

    // let result = process(
    //     "../../../model/Llama-3.2-1B-Instruct-f16.gguf",
    //     "../../../model/tokenizer.json",
    //     "What is Llama Eat? ",
    // );
    let result = process(&args.file, &args.tokenizer, &args.prompt);

    match result {
        Err(e) => {
            println!("tinyllama: error: {}", e)
        }
        _ => println!("tinyllama: done"),
    }
}
