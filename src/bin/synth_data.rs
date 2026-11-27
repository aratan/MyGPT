//! Genera un corpus sintético de conversaciones en español.

use mini_gpt::cli;
use mini_gpt::{Error, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

fn print_help() {
    println!(
        "synth_data - Genera conversaciones sintéticas para entrenar el modelo\n\n\
         Uso: synth_data [opciones]\n\n\
           --output <ruta>   Archivo de salida (defecto: data/synthetic.txt)\n\
           --count <N>       Número de conversaciones (defecto: 2000)\n\
           --seed <N>        Semilla determinista (defecto: 42)\n\
           --overwrite       Autoriza sobrescribir el archivo de salida\n\
           --help            Muestra esta ayuda\n\n\
         El formato es el mismo que usa gpt_chat en modo chat:\n\
           <|user|> pregunta <|eot|>\n\
           <|assistant|> respuesta <|eot|>"
    );
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = cli::collect_arguments();
    if cli::has(&args, "--help") {
        print_help();
        return Ok(());
    }
    cli::validate(
        &args,
        &["--help", "--overwrite"],
        &["--output", "--count", "--seed"],
    )?;

    let output = cli::value(&args, "--output", "data/synthetic.txt")?;
    let count = cli::size_value(&args, "--count", 2000, 1)?;
    let seed = cli::size_value(&args, "--seed", 42, 0)? as u32;
    let overwrite = cli::has(&args, "--overwrite");
    let path = Path::new(&output);
    if path.exists() && !overwrite {
        return Err(Error::new(format!(
            "El archivo ya existe: {output} (usa --overwrite)"
        )));
    }

    let samples = mini_gpt::synthetic::generate_tagged(count, seed);
    let mut categories: BTreeMap<&str, usize> = BTreeMap::new();
    let mut unique: BTreeSet<&str> = BTreeSet::new();
    let mut text = String::new();
    for (category, conversation) in &samples {
        *categories.entry(category).or_insert(0) += 1;
        unique.insert(conversation.as_str());
        text.push_str(conversation);
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, &text)?;

    println!("Conversaciones generadas: {count}");
    println!("Conversaciones únicas: {}", unique.len());
    println!("Bytes: {}", text.len());
    println!("Categorías:");
    for (category, total) in &categories {
        println!("  {category}: {total}");
    }
    println!("Salida: {output}");
    Ok(())
}
