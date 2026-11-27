use mini_gpt::adamw::{clip_gradients, AdamW, AdamWConfig};
use mini_gpt::data::{split_corpus_utf8, TextDataLoader};
use mini_gpt::inference::{evaluate_continuation, top_token_probabilities};
use mini_gpt::loss::{cross_entropy, cross_entropy_backward, perplexity, shift_tokens};
use mini_gpt::{BpeTokenizer, GenerationConfig, SamplingStrategy, StopCondition, Tensor, GPT};
use std::path::PathBuf;

fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "mini_gpt_rust_{}_{}_{}",
        std::process::id(),
        name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn tensor_matmul_matches_naive() {
    let lhs = Tensor::random([3, 4], 7, -1.0, 1.0).unwrap();
    let rhs = Tensor::random([4, 2], 8, -1.0, 1.0).unwrap();
    let product = lhs.matmul(&rhs).unwrap();
    assert_eq!(product.shape(), [3, 2]);
    for row in 0..3 {
        for column in 0..2 {
            let mut expected = 0.0f32;
            for inner in 0..4 {
                expected += lhs.at(&[row, inner]).unwrap() * rhs.at(&[inner, column]).unwrap();
            }
            let actual = product.at(&[row, column]).unwrap();
            assert!((actual - expected).abs() < 1e-5, "{actual} vs {expected}");
        }
    }

    let transposed = lhs.transpose().unwrap();
    assert_eq!(transposed.shape(), [4, 3]);
    assert_eq!(transposed.at(&[2, 1]).unwrap(), lhs.at(&[1, 2]).unwrap());

    let mut reshaped = lhs.clone();
    reshaped.reshape([2, 6]).unwrap();
    assert_eq!(reshaped.shape(), [2, 6]);
    assert_eq!(reshaped.size(), 12);
}

#[test]
fn tokenizer_trains_and_roundtrips() {
    let corpus = b"hola mundo, hola mundo, hola mundo\n<|user|> hola <|eot|>";
    let mut tokenizer = BpeTokenizer::new(300).unwrap();
    tokenizer.train(corpus, |_, _| {});
    assert!(tokenizer.vocab_size() > 260);
    assert!(tokenizer.vocab_size() <= 300);
    assert_eq!(tokenizer.vocab_size() - 260, tokenizer.merges().len());

    let encoded = tokenizer.encode(corpus);
    assert_eq!(tokenizer.decode(&encoded).unwrap(), corpus);
    assert!(encoded.contains(&mini_gpt::tokenizer::USER_ID));
    assert!(encoded.contains(&mini_gpt::tokenizer::EOT_ID));

    let path = temp_path("tokenizer.bpe");
    tokenizer.save(&path).unwrap();
    let loaded = BpeTokenizer::load(&path).unwrap();
    assert_eq!(loaded.vocab_size(), tokenizer.vocab_size());
    assert_eq!(loaded.merges(), tokenizer.merges());
    assert_eq!(loaded.encode(corpus), encoded);
    let _ = std::fs::remove_file(path);
}

#[test]
fn cross_entropy_and_backward_are_consistent() {
    let logits = Tensor::new([1, 2, 4], vec![1.0, 2.0, 3.0, 0.5, 0.0, 0.5, 2.0, 1.0]).unwrap();
    let targets = [2u32, 0];
    let mask = [1u8, 1];

    let first = GPT::softmax(&[1.0, 2.0, 3.0, 0.5], 1.0).unwrap();
    let second = GPT::softmax(&[0.0, 0.5, 2.0, 1.0], 1.0).unwrap();
    let expected = (-first[2].ln() + -second[0].ln()) / 2.0;
    let loss = cross_entropy(&logits, &targets, &mask).unwrap();
    assert!((loss - expected).abs() < 1e-5);

    let gradient = cross_entropy_backward(&logits, &targets, &mask).unwrap();
    for row in 0..2 {
        let sum: f32 = (0..4)
            .map(|column| gradient.at(&[0, row, column]).unwrap())
            .sum();
        assert!(sum.abs() < 1e-5);
    }
    assert!((perplexity(loss).unwrap() - loss.exp()).abs() < 1e-4);

    let shifted = shift_tokens(&[1, 2, 3]).unwrap();
    assert_eq!(shifted.inputs, vec![1, 2]);
    assert_eq!(shifted.targets, vec![2, 3]);
}

fn tiny_model() -> GPT {
    GPT::new(260, 16, 2, 1, 32, 8, 42, 1.0e-5).unwrap()
}

#[test]
fn gpt_backward_updates_gradients_and_adamw() {
    let mut model = tiny_model();
    let tokens: Vec<u32> = (0..4).collect();
    let targets: Vec<u32> = (1..5).collect();
    let mask = vec![1u8; 4];

    let logits = model.forward(&tokens, 1, 4).unwrap();
    assert_eq!(logits.shape(), [1, 4, 260]);

    model.zero_grad();
    let loss = model.backward(&tokens, 1, 4, &targets, &mask).unwrap();
    assert!(loss.is_finite() && loss > 0.0);
    let gradient_norm: f64 = model
        .gradients()
        .tensors()
        .iter()
        .flat_map(|gradient| gradient.data().iter())
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>()
        .sqrt();
    assert!(gradient_norm > 0.0);

    let clip_norm = clip_gradients(&mut model, 1.0).unwrap();
    assert!(clip_norm.is_finite());

    let before = model.output_weights().at(&[0, 0]).unwrap();
    let mut optimizer = AdamW::new(&model, AdamWConfig::default()).unwrap();
    optimizer.step(&mut model).unwrap();
    let after = model.output_weights().at(&[0, 0]).unwrap();
    assert_ne!(before, after);
    assert_eq!(optimizer.step_count(), 1);
}

#[test]
fn gpt_checkpoint_roundtrip_preserves_outputs() {
    let model = tiny_model();
    let tokens = [1u32, 2, 3, 4];
    let expected = model.forward(&tokens, 1, 4).unwrap();

    let path = temp_path("model.bin");
    model.save(&path).unwrap();
    let loaded = GPT::load(&path).unwrap();
    assert_eq!(loaded.vocab_size(), model.vocab_size());
    assert_eq!(loaded.embedding_dim(), model.embedding_dim());
    assert_eq!(loaded.parameter_count(), model.parameter_count());
    let actual = loaded.forward(&tokens, 1, 4).unwrap();
    assert_eq!(actual.data(), expected.data());
    let _ = std::fs::remove_file(path);

    let mut generator = mini_gpt::rng::Rng::new(1);
    let config = GenerationConfig {
        max_new_tokens: 5,
        strategy: SamplingStrategy::Greedy,
        stop_condition: StopCondition::EndOfConversation,
        ..Default::default()
    };
    let generated = model.generate(&[1, 2], &config).unwrap();
    assert!(generated.len() <= 5);
    let next = GPT::select_token(
        &[0.0, 1.0, 2.0],
        1.0,
        SamplingStrategy::Greedy,
        &mut generator,
        0,
    )
    .unwrap();
    assert_eq!(next, 2);
}

#[test]
fn data_loader_batches_and_state() {
    let mut tokenizer = BpeTokenizer::new(260).unwrap();
    tokenizer.train(b"", |_, _| {});
    let split = split_corpus_utf8("abcdefghijklmnopqrstuvwxyz0123456789", 0.25).unwrap();
    assert!(!split.training.is_empty());
    assert!(!split.validation.is_empty());

    let mut loader = TextDataLoader::new(
        &tokenizer,
        &split.training,
        &split.validation,
        4,
        2,
        7,
        false,
    )
    .unwrap();
    let batch = loader.next_batch().unwrap();
    assert_eq!(batch.batch_size, 2);
    assert_eq!(batch.inputs.len(), 8);
    assert!(batch.mask.iter().any(|&value| value == 1));
    let state = loader.state();
    let mut restored = TextDataLoader::new(
        &tokenizer,
        &split.training,
        &split.validation,
        4,
        2,
        7,
        false,
    )
    .unwrap();
    restored.restore_state(&state).unwrap();
    assert_eq!(restored.state(), state);
}

#[test]
fn inference_reports_probabilities() {
    let model = tiny_model();
    let context = [1u32, 2, 3];
    let ranked = top_token_probabilities(&model, &context, 5).unwrap();
    assert_eq!(ranked.len(), 5);
    assert_eq!(ranked[0].rank, 1);
    for window in ranked.windows(2) {
        assert!(window[0].probability >= window[1].probability);
    }
    let evaluation = evaluate_continuation(&model, &[1, 2, 3], &[4, 5]).unwrap();
    assert!(evaluation.loss.is_finite());
    assert_eq!(evaluation.tokens.len(), 2);
}

#[test]
fn synthetic_corpus_is_deterministic_and_well_formed() {
    let first = mini_gpt::synthetic::generate(300, 7);
    let second = mini_gpt::synthetic::generate(300, 7);
    assert_eq!(first, second);
    assert_eq!(first.len(), 300);
    for conversation in &first {
        assert!(conversation.starts_with("<|user|> "));
        assert!(conversation.contains(" <|eot|>\n<|assistant|> "));
        assert!(conversation.ends_with(" <|eot|>\n"));
    }
    let unique: std::collections::BTreeSet<&String> = first.iter().collect();
    assert!(unique.len() > 120, "solo {} únicas", unique.len());
}

#[test]
fn activation_trace_reports_attention_and_logit_lens() {
    use mini_gpt::trace;
    use std::collections::HashMap;

    let model = GPT::new(24, 8, 2, 2, 16, 8, 3, 1e-5).unwrap();
    let ids = [1u32, 2, 3, 4, 5];
    let (logits, trace) = model.forward_traced(&ids, 1, ids.len(), 4).unwrap();
    assert_eq!(logits.shape(), [1, 5, 24]);
    assert_eq!(trace.layers.len(), 2);
    assert_eq!(trace.tokens, ids);
    for layer in &trace.layers {
        assert_eq!(layer.block.attention.len(), 2);
        for head in &layer.block.attention {
            assert_eq!(head.len(), 25);
            for query in 0..5 {
                let sum: f32 = head[query * 5..query * 5 + 5].iter().sum();
                assert!((sum - 1.0).abs() < 1e-4);
                for key in query + 1..5 {
                    assert_eq!(head[query * 5 + key], 0.0);
                }
            }
        }
        assert!(layer.residual_input_norm > 0.0);
        assert_eq!(layer.logit_lens.len(), 4);
    }
    assert_eq!(trace.final_predictions.len(), 4);

    let labels: HashMap<u32, String> = ids.iter().map(|&id| (id, format!("t{id}"))).collect();
    let terminal = trace::render_terminal(&trace, &labels, 1, 0).unwrap();
    assert!(terminal.contains("Atención"));
    let html = trace::render_html(&trace, &labels, "test").unwrap();
    assert!(html.starts_with("<!doctype html>"));
    assert!(html.contains("const DATA = {"));
    assert!(html.contains("\"tokens\":[1,2,3,4,5]"));
}


#[test]
fn binary_reader_rejects_truncated_arrays_before_allocating() {
    let mut reader = mini_gpt::serialize::BytesReader::from_bytes(vec![0; 4]);
    assert!(reader.values(2).is_err());
    assert_eq!(reader.remaining(), 4);
}

#[test]
fn binary_writer_replaces_checkpoint_after_complete_write() {
    let path = temp_path("atomic-checkpoint.bin");
    std::fs::write(&path, b"old checkpoint").unwrap();

    let mut writer = mini_gpt::serialize::BytesWriter::new();
    writer.bytes(b"new checkpoint");
    writer.finish(&path).unwrap();

    assert_eq!(std::fs::read(&path).unwrap(), b"new checkpoint");
    let _ = std::fs::remove_file(path);
}


#[test]
fn output_projection_gradient_matches_finite_differences() {
    let mut model = tiny_model();
    let tokens = [1u32, 2, 3, 4];
    let targets = [2u32, 3, 4, 5];
    let mask = [1u8; 4];

    model.zero_grad();
    model.backward(&tokens, 1, tokens.len(), &targets, &mask).unwrap();

    let parameter_index = model.parameter_count_refs() - 1;
    let (_, gradient) = model.parameter(parameter_index).unwrap();
    let (coordinate, &analytic) = gradient
        .data()
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
        .unwrap();
    let original = model.output_weights().data()[coordinate];
    let epsilon = 1.0e-2;

    model.output_weights_mut().data_mut()[coordinate] = original + epsilon;
    let plus = cross_entropy(
        &model.forward(&tokens, 1, tokens.len()).unwrap(),
        &targets,
        &mask,
    )
    .unwrap();

    model.output_weights_mut().data_mut()[coordinate] = original - epsilon;
    let minus = cross_entropy(
        &model.forward(&tokens, 1, tokens.len()).unwrap(),
        &targets,
        &mask,
    )
    .unwrap();

    model.output_weights_mut().data_mut()[coordinate] = original;
    let numerical = (plus - minus) / (2.0 * epsilon);
    let tolerance = 5.0e-3 + analytic.abs() * 2.0e-2;
    assert!(
        (numerical - analytic).abs() <= tolerance,
        "gradient mismatch at {coordinate}: analytic={analytic}, numerical={numerical}"
    );
}


fn assert_parameter_gradient_matches_finite_differences(parameter_index: usize, name: &str) {
    let mut model = tiny_model();
    let tokens = [1u32, 2, 3, 4];
    let targets = [2u32, 3, 4, 5];
    let mask = [1u8; 4];

    model.zero_grad();
    model.backward(&tokens, 1, tokens.len(), &targets, &mask).unwrap();

    let (coordinate, &analytic) = model
        .parameter(parameter_index)
        .unwrap()
        .1
        .data()
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.abs().total_cmp(&right.1.abs()))
        .unwrap();
    let original = model.parameter(parameter_index).unwrap().0.data()[coordinate];
    let epsilon = 2.0e-2;

    model.parameter_mut(parameter_index).unwrap().0.data_mut()[coordinate] =
        original + epsilon;
    let plus = cross_entropy(
        &model.forward(&tokens, 1, tokens.len()).unwrap(),
        &targets,
        &mask,
    )
    .unwrap();

    model.parameter_mut(parameter_index).unwrap().0.data_mut()[coordinate] =
        original - epsilon;
    let minus = cross_entropy(
        &model.forward(&tokens, 1, tokens.len()).unwrap(),
        &targets,
        &mask,
    )
    .unwrap();

    model.parameter_mut(parameter_index).unwrap().0.data_mut()[coordinate] = original;
    let numerical = (plus - minus) / (2.0 * epsilon);
    let tolerance = 3.0e-3 + analytic.abs() * 5.0e-2;
    assert!(
        (numerical - analytic).abs() <= tolerance,
        "{name} gradient mismatch at {coordinate}: analytic={analytic}, numerical={numerical}"
    );
}

#[test]
fn transformer_parameter_gradients_match_finite_differences() {
    // Parameter indices: norm1.gamma=2, Q=6, V=8, attention output=9,
    // MLP fc1=10, MLP fc2=11 and final LayerNorm gamma=12.
    for (index, name) in [
        (2, "attention input LayerNorm gamma"),
        (6, "attention query weights"),
        (8, "attention value weights"),
        (9, "attention output weights"),
        (10, "MLP first projection weights"),
        (11, "MLP second projection weights"),
        (12, "final LayerNorm gamma"),
    ] {
        assert_parameter_gradient_matches_finite_differences(index, name);
    }
}
