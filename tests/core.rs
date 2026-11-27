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

#[cfg(feature = "cuda")]
#[test]
fn cuda_matmul_matches_cpu() {
    use mini_gpt::cuda;
    let (m, k, n) = (37usize, 29usize, 41usize);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 19) as f32 * 0.25 - 2.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 5 + 1) % 23) as f32 * 0.125 - 1.0)
        .collect();
    let gpu = cuda::matmul(&a, &b, m, k, n).expect("matmul CUDA");
    let mut cpu = vec![0.0f32; m * n];
    for row in 0..m {
        for inner in 0..k {
            let value = a[row * k + inner];
            for column in 0..n {
                cpu[row * n + column] += value * b[inner * n + column];
            }
        }
    }
    let max_error = gpu
        .iter()
        .zip(&cpu)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(max_error < 1e-3, "error máximo {max_error}");
}
