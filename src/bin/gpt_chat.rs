use mini_gpt::chat_mode::{build_chat_prompt, parse_chat_mode, ChatMode};
use mini_gpt::cli;
use mini_gpt::inference::{evaluate_continuation, top_token_probabilities, TokenProbability};
use mini_gpt::{
    BpeTokenizer, Error, GenerationConfig, Result, SamplingStrategy, StopCondition, GPT,
};
use std::io::{BufRead, Write};
use std::path::Path;

fn is_valid_utf8(bytes: &[u8]) -> bool {
    std::str::from_utf8(bytes).is_ok()
}

fn escape_text(bytes: &[u8]) -> String {
    const HEXADECIMAL: &[u8; 16] = b"0123456789ABCDEF";
    let valid_utf8 = is_valid_utf8(bytes);
    let mut escaped = String::new();
    for &byte in bytes {
        match byte {
            b'\n' => escaped.push_str("\\n"),
            b'\r' => escaped.push_str("\\r"),
            b'\t' => escaped.push_str("\\t"),
            0 => escaped.push_str("\\0"),
            b'"' => escaped.push_str("\\\""),
            b'\\' => escaped.push_str("\\\\"),
            _ => {
                if (32..=126).contains(&byte) || (byte >= 0x80 && valid_utf8) {
                    escaped.push(byte as char);
                } else {
                    escaped.push_str("\\x");
                    escaped.push(HEXADECIMAL[(byte >> 4) as usize] as char);
                    escaped.push(HEXADECIMAL[(byte & 0x0F) as usize] as char);
                }
            }
        }
    }
    escaped
}

fn print_ranked_tokens(tokenizer: &BpeTokenizer, ranked: &[TokenProbability]) {
    for item in ranked {
        let text = tokenizer
            .token(item.id)
            .map(escape_text)
            .unwrap_or_else(|_| String::from("<id fuera de rango>"));
        println!(
            "{}. \"{}\" (id={}) {:.4} %",
            item.rank,
            text,
            item.id,
            item.probability * 100.0
        );
    }
}

fn show_top_predictions(
    tokenizer: &BpeTokenizer,
    model: &GPT,
    context: &[u32],
    count: usize,
    adjusted: bool,
    config: &GenerationConfig,
) -> Result<()> {
    if !adjusted {
        println!(
            "Predicciones del siguiente token (logits originales del modelo, sin temperatura, Top-k ni repetition penalty):"
        );
        print_ranked_tokens(tokenizer, &top_token_probabilities(model, context, count)?);
        return Ok(());
    }

    let logits = model.forward(context, 1, context.len())?;
    let vocabulary = model.vocab_size();
    let start = (context.len() - 1) * vocabulary;
    let mut adjusted_logits = logits.data()[start..start + vocabulary].to_vec();
    GPT::apply_repetition_penalty(
        context,
        &mut adjusted_logits,
        config.repetition_penalty,
        config.repetition_window,
    )?;
    let probabilities = GPT::softmax(&adjusted_logits, config.temperature)?;
    let mut ranked: Vec<TokenProbability> = probabilities
        .iter()
        .enumerate()
        .map(|(index, &probability)| TokenProbability {
            id: index as u32,
            probability,
            rank: 0,
        })
        .collect();
    ranked.sort_by(|lhs, rhs| {
        rhs.probability
            .partial_cmp(&lhs.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(lhs.id.cmp(&rhs.id))
    });
    for (index, item) in ranked.iter_mut().enumerate() {
        item.rank = index + 1;
    }
    ranked.truncate(count);
    println!(
        "Predicciones del siguiente token (logits ajustados por repetition penalty y temperatura, antes de Top-k):"
    );
    print_ranked_tokens(tokenizer, &ranked);
    Ok(())
}

fn print_continuation_evaluation(
    tokenizer: &BpeTokenizer,
    evaluation: &mini_gpt::inference::ContinuationEvaluation,
) {
    println!("Evaluación determinista de la continuación esperada:");
    for (index, item) in evaluation.tokens.iter().enumerate() {
        let text = tokenizer
            .token(item.id)
            .map(escape_text)
            .unwrap_or_else(|_| String::from("<id fuera de rango>"));
        println!(
            "  posición {}: \"{}\" (id={}) probabilidad={:.6} ranking={}",
            index, text, item.id, item.probability, item.rank
        );
    }
    println!(
        "Cross-Entropy media de la continuación: {:.6}",
        evaluation.loss
    );
}

fn print_help() {
    println!(
        "gpt_chat - Generación interactiva con un GPT\n\n\
         Uso: gpt_chat --model <checkpoint> --tokenizer <bpe> [opciones]\n\n\
           --model <ruta>          Modelo o checkpoint (obligatorio)\n\
           --tokenizer <ruta>      Vocabulario BPE (obligatorio)\n\
           --strategy <estrategia> greedy o sampling (defecto: greedy)\n\
           --temperature <x>       Temperatura para sampling (defecto: 1.0)\n\
           --max-tokens <N>        Máximo de tokens nuevos por turno (defecto: 16)\n\
           --seed <N>              Semilla base de sampling (defecto: 42)\n\
           --mode <modo>           chat o completion (defecto: chat; continue es alias)\n\
           --top-k <N>             Candidatos para sampling; 0 desactiva (defecto: 0)\n\
           --repetition-penalty <x> Penalización de repetición (defecto: 1.0)\n\
           --repetition-window <N> Tokens recientes penalizados; 0 = todo el contexto\n\
           --show-top <N>          Mostrar N predicciones (defecto: 0)\n\
           --show-adjusted         Usar logits ajustados en vez de originales\n\
           --prompt <texto>        Procesar un prompt una vez y salir\n\
           --expected <texto>      Continuación esperada; evalúa sin generar\n\
           --help                  Muestra esta ayuda\n\n\
         chat usa <|user|>, <|assistant|> y se detiene en <|eot|>.\n\
         completion genera continuaciones sin añadir tokens especiales.\n\
         Comandos interactivos: /reset limpia el contexto; /exit termina."
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
        &["--help", "--show-adjusted"],
        &[
            "--model",
            "--tokenizer",
            "--strategy",
            "--temperature",
            "--max-tokens",
            "--seed",
            "--mode",
            "--top-k",
            "--repetition-penalty",
            "--repetition-window",
            "--show-top",
            "--prompt",
            "--expected",
        ],
    )?;

    let model_path = cli::required(&args, "--model")?;
    let tokenizer_path = cli::required(&args, "--tokenizer")?;
    let strategy_name = cli::value(&args, "--strategy", "greedy")?;
    let mode = parse_chat_mode(&cli::value(&args, "--mode", "chat")?)?;
    if strategy_name != "greedy" && strategy_name != "sampling" {
        return Err(Error::new("--strategy debe ser greedy o sampling"));
    }
    cli::require_input_file(Path::new(&model_path))?;
    cli::require_input_file(Path::new(&tokenizer_path))?;

    let config = GenerationConfig {
        max_new_tokens: cli::size_value(&args, "--max-tokens", 16, 1)?,
        temperature: cli::float_value(&args, "--temperature", 1.0)?,
        seed: cli::size_value(&args, "--seed", 42, 0)? as u32,
        strategy: if strategy_name == "greedy" {
            SamplingStrategy::Greedy
        } else {
            SamplingStrategy::Sampling
        },
        top_k: cli::size_value(&args, "--top-k", 0, 0)?,
        repetition_penalty: cli::float_value(&args, "--repetition-penalty", 1.0)?,
        repetition_window: cli::size_value(&args, "--repetition-window", 0, 0)?,
        stop_condition: if mode == ChatMode::Chat {
            StopCondition::EndOfTurn
        } else {
            StopCondition::EndOfConversation
        },
    };
    if config.temperature <= 0.0 || !config.temperature.is_finite() {
        return Err(Error::new("--temperature debe ser positiva"));
    }
    if !config.repetition_penalty.is_finite() || config.repetition_penalty < 1.0 {
        return Err(Error::new("--repetition-penalty debe ser al menos 1.0"));
    }

    let show_top = cli::size_value(&args, "--show-top", 0, 0)?;
    let show_adjusted = cli::has(&args, "--show-adjusted");
    let prompt_argument = cli::value(&args, "--prompt", "")?;
    let expected = cli::value(&args, "--expected", "")?;
    if !expected.is_empty() && prompt_argument.is_empty() {
        return Err(Error::new("--expected requiere también --prompt"));
    }

    let tokenizer = BpeTokenizer::load(Path::new(&tokenizer_path))?;
    let model = GPT::load(Path::new(&model_path))?;
    if tokenizer.vocab_size() != model.vocab_size() {
        return Err(Error::new("El vocabulario no es compatible con el modelo"));
    }
    if config.top_k > model.vocab_size() {
        return Err(Error::new("--top-k no puede superar el vocabulario"));
    }
    if show_top > model.vocab_size() {
        return Err(Error::new("--show-top no puede superar el vocabulario"));
    }

    let process_prompt =
        |input: &str, context: &mut Vec<u32>, config: &GenerationConfig| -> Result<Vec<u32>> {
            let prompt = build_chat_prompt(mode, input);
            let prompt_ids = tokenizer.encode(prompt.as_bytes());
            context.extend_from_slice(&prompt_ids);
            if context.len() > model.max_seq_len() {
                let excess = context.len() - model.max_seq_len();
                context.drain(..excess);
            }
            if show_top != 0 {
                show_top_predictions(
                    &tokenizer,
                    &model,
                    context,
                    show_top,
                    show_adjusted,
                    &config,
                )?;
            }
            if !expected.is_empty() {
                let expected_ids = tokenizer.encode(expected.as_bytes());
                print_continuation_evaluation(
                    &tokenizer,
                    &evaluate_continuation(&model, context, &expected_ids)?,
                );
                return Ok(Vec::new());
            }
            let generated = model.generate(context, &config)?;
            let text = tokenizer.decode(&generated)?;
            let printable = if is_valid_utf8(&text) {
                String::from_utf8_lossy(&text).to_string()
            } else {
                escape_text(&text)
            };
            println!(
                "{} {}",
                if mode == ChatMode::Chat {
                    "asistente>"
                } else {
                    "continuación>"
                },
                printable
            );
            Ok(generated)
        };

    if !prompt_argument.is_empty() {
        let mut context = Vec::new();
        process_prompt(&prompt_argument, &mut context, &config)?;
        return Ok(());
    }

    let base_seed = cli::size_value(&args, "--seed", 42, 0)? as u32;
    let mut context: Vec<u32> = Vec::new();
    let mut turn = 0usize;
    println!(
        "Modelo: {}\nModo: {}; estrategia: {}\nEscribe /reset para limpiar el contexto o /exit para salir.",
        model_path,
        if mode == ChatMode::Chat {
            "chat"
        } else {
            "completion"
        },
        strategy_name
    );
    let stdin = std::io::stdin();
    loop {
        print!(
            "{}",
            if mode == ChatMode::Chat {
                "\n\nusuario> "
            } else {
                "\n\ntexto> "
            }
        );
        std::io::stdout().flush()?;
        let mut input = String::new();
        if stdin.lock().read_line(&mut input)? == 0 {
            println!("\nEntrada finalizada.");
            break;
        }
        let input = input.trim_end_matches(['\n', '\r']).to_string();
        if input == "/exit" {
            break;
        }
        if input == "/reset" {
            context.clear();
            println!("Contexto reiniciado.");
            continue;
        }
        if input.is_empty() && context.is_empty() {
            println!("Introduce texto para iniciar el contexto.");
            continue;
        }

        let mut turn_config = config.clone();
        turn_config.seed = base_seed.wrapping_add(turn as u32);
        let generated = process_prompt(&input, &mut context, &turn_config)?;
        context.extend_from_slice(&generated);
        if context.len() > model.max_seq_len() {
            let excess = context.len() - model.max_seq_len();
            context.drain(..excess);
        }
        turn += 1;
    }
    Ok(())
}
