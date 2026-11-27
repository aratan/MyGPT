use mini_gpt::cli;
use mini_gpt::data::{read_utf8_file, split_corpus_utf8, utf8_codepoint_count, CorpusSplit};
use mini_gpt::BpeTokenizer;
use std::path::Path;

fn print_help() {
    println!(
        "tokenizer_train - Entrena Byte-Level BPE\n\n\
         Uso: tokenizer_train --corpus <archivo> [opciones]\n\n\
           --corpus <ruta>          Corpus UTF-8 (obligatorio)\n\
           --vocab-size <N>         Vocabulario máximo (defecto: 2048)\n\
           --output <ruta>          Vocabulario de salida (defecto: tokenizer.bpe)\n\
           --validation-ratio <x>   Fracción reservada, no usada para BPE (defecto: 0.1; 0 desactiva)\n\
           --progress-every <N>     Mostrar progreso cada N merges (defecto: 25)\n\
           --overwrite             Autoriza sobrescribir la salida\n\
           --help                  Muestra esta ayuda"
    );
}

fn nonempty_lines(text: &str) -> usize {
    text.lines().filter(|line| !line.is_empty()).count()
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run() -> mini_gpt::Result<()> {
    let args = cli::collect_arguments();
    if cli::has(&args, "--help") {
        print_help();
        return Ok(());
    }
    cli::validate(
        &args,
        &["--help", "--overwrite"],
        &[
            "--corpus",
            "--vocab-size",
            "--output",
            "--validation-ratio",
            "--progress-every",
        ],
    )?;

    let corpus_path = cli::required(&args, "--corpus")?;
    let output_path = cli::value(&args, "--output", "tokenizer.bpe")?;
    let vocab_size = cli::size_value(&args, "--vocab-size", 2048, 260)?;
    let validation_ratio = cli::float_value(&args, "--validation-ratio", 0.1)?;
    let progress_every = cli::size_value(&args, "--progress-every", 25, 1)?;
    let overwrite = cli::has(&args, "--overwrite");
    if validation_ratio < 0.0 || validation_ratio >= 1.0 {
        return Err(mini_gpt::Error::new(
            "--validation-ratio debe estar entre 0 y 1, sin incluir 1",
        ));
    }
    cli::require_input_file(Path::new(&corpus_path))?;
    cli::require_output_file(Path::new(&output_path), overwrite)?;

    let corpus = read_utf8_file(Path::new(&corpus_path))?;
    let split = if validation_ratio == 0.0 {
        CorpusSplit {
            training: corpus.clone(),
            validation: String::new(),
        }
    } else {
        split_corpus_utf8(&corpus, f64::from(validation_ratio))?
    };
    println!("Corpus: {corpus_path}");
    println!("Bytes: {}", corpus.len());
    println!("Puntos de código UTF-8: {}", utf8_codepoint_count(&corpus));
    println!("Caracteres entrenamiento: {}", split.training.len());
    println!("Caracteres validación: {}", split.validation.len());
    println!("Iniciando BPE con máximo {vocab_size} tokens...");

    let mut tokenizer = BpeTokenizer::new(vocab_size)?;
    tokenizer.train(split.training.as_bytes(), |learned, target| {
        if learned % progress_every == 0 || learned == target {
            let percent = if target == 0 {
                100.0
            } else {
                100.0 * learned as f64 / target as f64
            };
            println!("  merges={learned}/{target} ({percent:.2}%)");
        }
    });

    if tokenizer.decode(&tokenizer.encode(split.training.as_bytes()))? != split.training.as_bytes()
        || tokenizer.decode(&tokenizer.encode(split.validation.as_bytes()))?
            != split.validation.as_bytes()
    {
        return Err(mini_gpt::Error::new(
            "La comprobación encode/decode ha fallado",
        ));
    }
    tokenizer.save(Path::new(&output_path))?;

    let train_tokens = tokenizer.encode(split.training.as_bytes()).len();
    let validation_tokens = tokenizer.encode(split.validation.as_bytes()).len();
    let total_tokens = train_tokens + validation_tokens;
    let lines = nonempty_lines(&corpus).max(1);
    println!("Entrenamiento BPE completado");
    println!("Vocabulario aprendido: {}", tokenizer.vocab_size());
    println!("Merges aprendidos: {}", tokenizer.merges().len());
    println!("Tokens entrenamiento: {train_tokens}");
    println!("Tokens validación: {validation_tokens}");
    println!("Tokens totales: {total_tokens}");
    println!(
        "Media aproximada por línea: {:.4}",
        total_tokens as f64 / lines as f64
    );
    println!(
        "Bytes por token: {:.4}",
        corpus.len() as f64 / total_tokens as f64
    );
    println!("encode/decode: correcto");
    println!("Vocabulario guardado en: {output_path}");
    Ok(())
}
