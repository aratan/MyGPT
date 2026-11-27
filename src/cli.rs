//! Utilidades de línea de comandos equivalentes a las del original.

use crate::error::{Error, Result};
use std::path::Path;

pub type Arguments = Vec<String>;

pub fn collect_arguments() -> Arguments {
    std::env::args().collect()
}

pub fn has(args: &[String], flag: &str) -> bool {
    args.iter().skip(1).any(|argument| argument == flag)
}

pub fn value(args: &[String], option: &str, default_value: &str) -> Result<String> {
    let mut result = default_value.to_string();
    let mut found = false;
    let mut index = 1;
    while index < args.len() {
        if args[index] != option {
            index += 1;
            continue;
        }
        if found {
            return Err(Error::new(format!("Opción duplicada: {option}")));
        }
        if index + 1 >= args.len() || args[index + 1].starts_with("--") {
            return Err(Error::new(format!("Falta valor para {option}")));
        }
        result = args[index + 1].clone();
        found = true;
        index += 2;
    }
    Ok(result)
}

pub fn required(args: &[String], option: &str) -> Result<String> {
    let result = value(args, option, "")?;
    if result.is_empty() {
        return Err(Error::new(format!(
            "Falta la opción obligatoria {option}"
        )));
    }
    Ok(result)
}

pub fn size_value(
    args: &[String],
    option: &str,
    default_value: usize,
    minimum: usize,
) -> Result<usize> {
    let text = value(args, option, &default_value.to_string())?;
    let result = text
        .parse::<usize>()
        .map_err(|_| Error::new(format!("Entero inválido en {option}: {text}")))?;
    if result < minimum {
        return Err(Error::new(format!(
            "Valor fuera de rango en {option}: {text}"
        )));
    }
    Ok(result)
}

pub fn float_value(args: &[String], option: &str, default_value: f32) -> Result<f32> {
    let text = value(args, option, &format!("{default_value}"))?;
    let result = text
        .parse::<f32>()
        .map_err(|_| Error::new(format!("Número inválido en {option}: {text}")))?;
    if !result.is_finite() {
        return Err(Error::new(format!("Número inválido en {option}: {text}")));
    }
    Ok(result)
}

pub fn validate(args: &[String], flags: &[&str], options: &[&str]) -> Result<()> {
    let mut index = 1;
    while index < args.len() {
        let argument = args[index].as_str();
        if flags.contains(&argument) {
            index += 1;
            continue;
        }
        if options.contains(&argument) {
            if index + 1 >= args.len() || args[index + 1].starts_with("--") {
                return Err(Error::new(format!("Falta valor para {argument}")));
            }
            index += 2;
            continue;
        }
        return Err(Error::new(format!("Argumento desconocido: {argument}")));
    }
    Ok(())
}

pub fn require_input_file(path: &Path) -> Result<()> {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(()),
        _ => Err(Error::new(format!(
            "No existe el archivo de entrada: {}",
            path.display()
        ))),
    }
}

pub fn require_output_file(path: &Path, overwrite: bool) -> Result<()> {
    if path.as_os_str().is_empty() {
        return Err(Error::new("La ruta de salida está vacía"));
    }
    match std::fs::metadata(path) {
        Ok(metadata) => {
            if !overwrite || metadata.is_dir() {
                return Err(Error::new(format!(
                    "La salida ya existe; usa --overwrite para autorizarla: {}",
                    path.display()
                )));
            }
        }
        Err(_) => {}
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(Error::new(format!(
                "No existe el directorio de salida: {}",
                parent.display()
            )));
        }
    }
    Ok(())
}
