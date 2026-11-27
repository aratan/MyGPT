use mini_gpt::adamw::{clip_gradients, scale_gradients, AdamW, AdamWConfig};
use mini_gpt::cli;
use mini_gpt::data::{read_utf8_file, split_corpus_utf8, TextDataLoader};
use mini_gpt::loss::{cross_entropy, perplexity};
use mini_gpt::training_state::{load_training_state, save_training_state, TrainingState};
use mini_gpt::{Error, Result, Tensor, GPT};
use std::path::{Path, PathBuf};
use std::time::Instant;

struct CheckpointPaths {
    model: PathBuf,
    optimizer: PathBuf,
    state: PathBuf,
}

fn derived_paths(model: &Path) -> CheckpointPaths {
    let parent = model.parent().unwrap_or(Path::new(""));
    let stem = model
        .file_stem()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_default();
    let extension = model
        .extension()
        .map(|value| format!(".{}", value.to_string_lossy()))
        .unwrap_or_default();
    CheckpointPaths {
        model: model.to_path_buf(),
        optimizer: parent.join(format!("{stem}_adamw{extension}")),
        state: PathBuf::from(format!("{}.data", model.display())),
    }
}

fn next_paths(resume: &Path) -> CheckpointPaths {
    let parent = resume.parent().unwrap_or(Path::new(""));
    let stem = resume
        .file_stem()
        .map(|value| value.to_string_lossy().to_string())
        .unwrap_or_default();
    let extension = resume
        .extension()
        .map(|value| format!(".{}", value.to_string_lossy()))
        .unwrap_or_default();
    derived_paths(&parent.join(format!("{stem}_next{extension}")))
}

fn normalized_path(path: &str) -> String {
    std::fs::canonicalize(path)
        .unwrap_or_else(|_| PathBuf::from(path))
        .to_string_lossy()
        .to_string()
}

fn print_help() {
    println!(
        "gpt_train - Entrena o reanuda un GPT\n\n\
         Desde cero:\n  gpt_train --corpus <ruta> --tokenizer <bpe> [opciones]\n\
         Reanudar:\n  gpt_train --resume <checkpoint> --steps <N> [opciones]\n\n\
         Datos:\n\
           --corpus <ruta>          Corpus UTF-8\n\
           --tokenizer <ruta>       Vocabulario BPE\n\
           --resume <ruta>          Checkpoint a continuar\n\
           --steps <N>              Pasos adicionales (defecto: 20)\n\
           --batch-size <N>         Tamaño de batch (defecto: 2)\n\
           --grad-accum <N>         Micro-batches acumulados por paso (defecto: 1)\n\
           --context-length <N>     Tokens por secuencia (defecto: 128)\n\
           --seed <N>               Semilla de datos (defecto: 42)\n\n\
         Arquitectura, solo desde cero:\n\
           --embedding-dim <N>      Defecto: 256\n\
           --num-heads <N>          Defecto: 4\n\
           --num-layers <N>         Defecto: 4\n\
           --mlp-hidden-dim <N>     Defecto: 1024\n\
           --max-seq-len <N>        Defecto: 128; máximo: 256\n\n\
         Optimización:\n\
           --lr <x>                 Defecto: 0.0003\n\
           --beta1 <x>              Defecto: 0.9\n\
           --beta2 <x>              Defecto: 0.999\n\
           --epsilon <x>            Defecto: 1e-8\n\
           --weight-decay <x>       Defecto: 0.01\n\
           --clip <x>               Norma global máxima (defecto: 1.0)\n\n\
         Registro y checkpoints:\n\
           --save-every <N>         Guardar cada N pasos; 0 desactiva (defecto: 50)\n\
           --eval-every <N>         Evaluar cada N pasos; 0 desactiva (defecto: 10)\n\
           --validation-samples <N> Secuencias por evaluación (defecto: 16)\n\
           --log-every <N>          Registrar cada N pasos (defecto: 1)\n\
           --output-model <ruta>    Modelo de salida\n\
           --output-optimizer <ruta> Estado AdamW de salida\n\
           --output-state <ruta>    Estado de datos de salida\n\
           --optimizer <ruta>       AdamW de entrada al reanudar\n\
           --overwrite             Autoriza sobrescribir salidas\n\
           --help                  Muestra esta ayuda\n\n\
         Por defecto no se sobrescribe el checkpoint de entrada: se crea *_next.bin."
    );
}

fn evaluate_subset(loader: &mut TextDataLoader, model: &GPT, max_samples: usize) -> Result<f64> {
    loader.reset_validation();
    let mut loss_sum = 0.0f64;
    let mut tokens = 0usize;
    let mut samples = 0usize;
    while samples < max_samples {
        let Some(batch) = loader.next_validation_batch() else {
            break;
        };
        let logits: Tensor = model.forward(&batch.inputs, batch.batch_size, batch.seq_len)?;
        let loss = cross_entropy(&logits, &batch.targets, &batch.mask)?;
        loss_sum += f64::from(loss) * batch.real_tokens as f64;
        tokens += batch.real_tokens;
        samples += batch.batch_size;
    }
    if tokens == 0 {
        return Err(Error::new("No hay muestras de validación disponibles"));
    }
    Ok(loss_sum / tokens as f64)
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
        &[
            "--corpus",
            "--tokenizer",
            "--resume",
            "--steps",
            "--batch-size",
            "--grad-accum",
            "--context-length",
            "--seed",
            "--embedding-dim",
            "--num-heads",
            "--num-layers",
            "--mlp-hidden-dim",
            "--max-seq-len",
            "--lr",
            "--beta1",
            "--beta2",
            "--epsilon",
            "--weight-decay",
            "--clip",
            "--save-every",
            "--eval-every",
            "--validation-samples",
            "--log-every",
            "--output-model",
            "--output-optimizer",
            "--output-state",
            "--optimizer",
        ],
    )?;

    let resume = cli::value(&args, "--resume", "")?;
    let is_resume = !resume.is_empty();
    let overwrite = cli::has(&args, "--overwrite");
    let additional_steps = cli::size_value(&args, "--steps", 20, 1)?;
    let batch_size = cli::size_value(&args, "--batch-size", 2, 1)?;
    let grad_accum = cli::size_value(&args, "--grad-accum", 1, 1)?;
    let context_length = cli::size_value(&args, "--context-length", 128, 1)?;
    let save_every = cli::size_value(&args, "--save-every", 50, 0)?;
    let eval_every = cli::size_value(&args, "--eval-every", 10, 0)?;
    let validation_samples = cli::size_value(&args, "--validation-samples", 16, 1)?;
    let log_every = cli::size_value(&args, "--log-every", 1, 1)?;
    let seed = cli::size_value(&args, "--seed", 42, 0)? as u32;
    let clipping = f64::from(cli::float_value(&args, "--clip", 1.0)?);
    if !clipping.is_finite() || clipping <= 0.0 {
        return Err(Error::new("--clip debe ser positivo"));
    }

    let mut output = if is_resume {
        next_paths(Path::new(&resume))
    } else {
        derived_paths(Path::new("gpt_checkpoint.bin"))
    };
    output.model = PathBuf::from(cli::value(
        &args,
        "--output-model",
        &output.model.to_string_lossy(),
    )?);
    output.optimizer = PathBuf::from(cli::value(
        &args,
        "--output-optimizer",
        &output.optimizer.to_string_lossy(),
    )?);
    output.state = PathBuf::from(cli::value(
        &args,
        "--output-state",
        &output.state.to_string_lossy(),
    )?);
    cli::require_output_file(&output.model, overwrite)?;
    cli::require_output_file(&output.optimizer, overwrite)?;
    cli::require_output_file(&output.state, overwrite)?;

    let previous = if is_resume && PathBuf::from(format!("{resume}.data")).exists() {
        Some(load_training_state(Path::new(&format!("{resume}.data")))?)
    } else {
        None
    };
    let corpus_argument = cli::value(&args, "--corpus", "")?;
    let tokenizer_argument = cli::value(&args, "--tokenizer", "")?;
    let corpus_path = if !corpus_argument.is_empty() {
        corpus_argument
    } else if let Some(previous) = &previous {
        previous.corpus_path.clone()
    } else {
        String::new()
    };
    let tokenizer_path = if !tokenizer_argument.is_empty() {
        tokenizer_argument
    } else if let Some(previous) = &previous {
        previous.tokenizer_path.clone()
    } else {
        String::new()
    };
    if corpus_path.is_empty() || tokenizer_path.is_empty() {
        return Err(Error::new(
            "Debes indicar --corpus y --tokenizer, o usar un checkpoint con metadatos",
        ));
    }
    cli::require_input_file(Path::new(&corpus_path))?;
    cli::require_input_file(Path::new(&tokenizer_path))?;

    let optimizer_config = AdamWConfig {
        learning_rate: cli::float_value(&args, "--lr", 0.0003)?,
        beta1: cli::float_value(&args, "--beta1", 0.9)?,
        beta2: cli::float_value(&args, "--beta2", 0.999)?,
        epsilon: cli::float_value(&args, "--epsilon", 1.0e-8)?,
        weight_decay: cli::float_value(&args, "--weight-decay", 0.01)?,
    };

    let tokenizer = mini_gpt::BpeTokenizer::load(Path::new(&tokenizer_path))?;
    let mut model = if is_resume {
        GPT::load(Path::new(&resume))?
    } else {
        GPT::new(
            tokenizer.vocab_size(),
            cli::size_value(&args, "--embedding-dim", 256, 1)?,
            cli::size_value(&args, "--num-heads", 4, 1)?,
            cli::size_value(&args, "--num-layers", 4, 1)?,
            cli::size_value(&args, "--mlp-hidden-dim", 1024, 1)?,
            cli::size_value(&args, "--max-seq-len", 128, 1)?,
            seed,
            1.0e-5,
        )?
    };
    if tokenizer.vocab_size() != model.vocab_size() {
        return Err(Error::new("El vocabulario no es compatible con el modelo"));
    }
    if context_length > model.max_seq_len() || model.max_seq_len() > 256 {
        return Err(Error::new(
            "El contexto excede max_seq_len del modelo o el máximo 256",
        ));
    }

    let mut optimizer = if is_resume {
        let optimizer_path = cli::value(
            &args,
            "--optimizer",
            &derived_paths(Path::new(&resume))
                .optimizer
                .to_string_lossy(),
        )?;
        cli::require_input_file(Path::new(&optimizer_path))?;
        AdamW::load(Path::new(&optimizer_path), &model)?
    } else {
        AdamW::new(&model, optimizer_config)?
    };

    let corpus = read_utf8_file(Path::new(&corpus_path))?;
    let split = split_corpus_utf8(&corpus, 0.1)?;
    let mut loader = TextDataLoader::new(
        &tokenizer,
        &split.training,
        &split.validation,
        context_length,
        batch_size,
        seed,
        true,
    )?;
    if loader.training_sample_count() == 0 {
        return Err(Error::new(
            "El corpus no contiene suficientes tokens para entrenar",
        ));
    }

    let same_dataset = previous.as_ref().is_some_and(|previous| {
        normalized_path(&previous.corpus_path) == normalized_path(&corpus_path)
            && normalized_path(&previous.tokenizer_path) == normalized_path(&tokenizer_path)
    });
    if same_dataset {
        if let Some(previous) = &previous {
            if loader.restore_state(&previous.data).is_err() {
                println!("El estado de datos no es compatible; se inicia un nuevo muestreo.");
            }
        }
    } else if previous.is_some() {
        println!(
            "Corpus o vocabulario cambiado: se conservan pesos y AdamW, pero se reinicia el muestreo."
        );
    }

    let mut epoch = if same_dataset {
        previous.as_ref().map(|value| value.epoch).unwrap_or(1)
    } else {
        1
    };
    let mut processed_tokens = if same_dataset {
        previous
            .as_ref()
            .map(|value| value.processed_tokens)
            .unwrap_or(0)
    } else {
        0
    };
    let mut session_tokens = 0usize;
    let mut completed = 0usize;
    let mut last_loss = 0.0f32;
    let initial_parameter = model.output_weights().at(&[0, 0])?;
    let start_time = Instant::now();

    println!(
        "Modo: {}\nVocabulario: {}\nParámetros: {}\nPasos adicionales: {}\nBatch: {}; acumulación: {}; contexto: {}\nLearning rate: {}\nPaso inicial de AdamW: {}",
        if is_resume { "reanudación" } else { "desde cero" },
        tokenizer.vocab_size(),
        model.parameter_count().total(),
        additional_steps,
        batch_size,
        grad_accum,
        context_length,
        optimizer.config().learning_rate,
        optimizer.step_count()
    );
    if eval_every != 0 {
        let initial_validation = evaluate_subset(&mut loader, &model, validation_samples)?;
        println!("Validation Loss inicial: {initial_validation:.6}");
        println!(
            "Perplexity inicial: {:.6}",
            perplexity(initial_validation as f32)?
        );
    }

    while completed < additional_steps {
        let step_start = Instant::now();
        model.zero_grad();
        let mut accumulated_loss = 0.0f64;
        let mut accumulated_tokens = 0usize;
        for _ in 0..grad_accum {
            let batch = match loader.next_batch() {
                Some(batch) => batch,
                None => {
                    epoch += 1;
                    loader.reset_epoch();
                    loader
                        .next_batch()
                        .ok_or_else(|| Error::new("No hay más batches de entrenamiento"))?
                }
            };
            let loss = model.backward(
                &batch.inputs,
                batch.batch_size,
                batch.seq_len,
                &batch.targets,
                &batch.mask,
            )?;
            if !loss.is_finite() {
                return Err(Error::new("Entrenamiento detenido por valores no finitos"));
            }
            accumulated_loss += f64::from(loss);
            accumulated_tokens += batch.real_tokens;
        }
        scale_gradients(&mut model, 1.0 / grad_accum as f32)?;
        let loss = (accumulated_loss / grad_accum as f64) as f32;
        let norm = clip_gradients(&mut model, clipping)?;
        if !norm.is_finite() {
            return Err(Error::new("Entrenamiento detenido por valores no finitos"));
        }
        optimizer.step(&mut model)?;
        completed += 1;
        processed_tokens += accumulated_tokens;
        session_tokens += accumulated_tokens;
        last_loss = loss;
        let step_seconds = step_start.elapsed().as_secs_f64();
        let elapsed = start_time.elapsed().as_secs_f64();

        if completed % log_every == 0 || completed == additional_steps {
            println!(
                "Paso {} (+{}): loss={:.6} perplexity={:.6} tokens/s={:.2} tiempo_paso={:.4}s total={:.2}s",
                optimizer.step_count(),
                completed,
                loss,
                perplexity(loss)?,
                accumulated_tokens as f64 / step_seconds.max(1.0e-9),
                step_seconds,
                elapsed
            );
        }
        if eval_every != 0 && completed % eval_every == 0 {
            let validation = evaluate_subset(&mut loader, &model, validation_samples)?;
            println!(
                "  Validation Loss: {validation:.6} perplexity={:.6}",
                perplexity(validation as f32)?
            );
        }
        if save_every != 0 && completed % save_every == 0 {
            model.save(&output.model)?;
            optimizer.save(&output.optimizer)?;
            save_training_state(
                &output.state,
                &TrainingState {
                    corpus_path: normalized_path(&corpus_path),
                    tokenizer_path: normalized_path(&tokenizer_path),
                    epoch,
                    processed_tokens,
                    data: loader.state(),
                },
            )?;
            println!("  Checkpoint guardado en {}", output.model.display());
        }
    }

    model.save(&output.model)?;
    optimizer.save(&output.optimizer)?;
    save_training_state(
        &output.state,
        &TrainingState {
            corpus_path: normalized_path(&corpus_path),
            tokenizer_path: normalized_path(&tokenizer_path),
            epoch,
            processed_tokens,
            data: loader.state(),
        },
    )?;
    if initial_parameter == model.output_weights().at(&[0, 0])? && completed != 0 {
        return Err(Error::new(
            "Los parámetros no cambiaron durante el entrenamiento",
        ));
    }
    if eval_every != 0 {
        let validation = evaluate_subset(&mut loader, &model, validation_samples)?;
        println!("Validation Loss final: {validation:.6}");
        println!("Perplexity final: {:.6}", perplexity(validation as f32)?);
    }
    let elapsed = start_time.elapsed().as_secs_f64();
    println!("Última training loss: {last_loss:.6}");
    println!("Pasos adicionales completados: {completed}");
    println!("Paso total de AdamW: {}", optimizer.step_count());
    println!("Tiempo total: {elapsed:.2}s");
    println!(
        "Tokens/s medio: {:.2}",
        session_tokens as f64 / elapsed.max(1.0e-9)
    );
    println!("Checkpoint final: {}", output.model.display());
    println!("AdamW: {}", output.optimizer.display());
    println!("Datos: {}", output.state.display());
    Ok(())
}
