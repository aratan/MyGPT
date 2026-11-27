//! Telemetría de interpretabilidad: atención, flujo residual, MLP y logit lens.
//!
//! El motor solo paga el coste de capturar la traza cuando se llama a
//! `GPT::forward_traced`; la inferencia normal no cambia.

use crate::error::{Error, Result};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

/// Resumen de las activaciones de la capa MLP tras GELU.
#[derive(Clone, Debug)]
pub struct MlpTrace {
    pub mean: f32,
    pub max: f32,
    pub active_fraction: f32,
    pub top_neurons: Vec<(usize, f32)>,
}

/// Datos de un bloque transformer.
#[derive(Clone, Debug)]
pub struct BlockTrace {
    /// `[cabeza][consulta * seq + clave]`, lote 0.
    pub attention: Vec<Vec<f32>>,
    /// Entropía media (en nats) de cada cabeza.
    pub attention_entropy: Vec<f32>,
    pub mlp: MlpTrace,
}

/// Datos de una capa: bloque + logit lens sobre su salida residual.
#[derive(Clone, Debug)]
pub struct LayerTrace {
    pub residual_input_norm: f32,
    pub residual_output_norm: f32,
    pub block: BlockTrace,
    /// `(token, probabilidad)` del logit lens en la última posición.
    pub logit_lens: Vec<(u32, f32)>,
}

/// Traza completa de un forward.
#[derive(Clone, Debug)]
pub struct ActivationTrace {
    pub tokens: Vec<u32>,
    pub layers: Vec<LayerTrace>,
    pub final_predictions: Vec<(u32, f32)>,
    pub top_k: usize,
}

impl ActivationTrace {
    pub fn sequence_length(&self) -> usize {
        if self.tokens.is_empty() {
            0
        } else {
            self.tokens.len()
        }
    }
}

/// Media de la norma L2 de cada vector de token.
pub fn mean_l2(values: &[f32], rows: usize, dimension: usize) -> f32 {
    if rows == 0 || dimension == 0 {
        return 0.0;
    }
    let mut total = 0.0f64;
    for row in 0..rows {
        let slice = &values[row * dimension..row * dimension + dimension];
        let norm = slice
            .iter()
            .map(|&value| f64::from(value) * f64::from(value))
            .sum::<f64>()
            .sqrt();
        total += norm;
    }
    (total / rows as f64) as f32
}

/// Entropía de Shannon (nats) de una distribución.
pub fn entropy(probabilities: &[f32]) -> f32 {
    let mut value = 0.0f64;
    for &probability in probabilities {
        let probability = f64::from(probability);
        if probability > 0.0 {
            value -= probability * probability.ln();
        }
    }
    value as f32
}

/// Top-k del logit lens para la última posición de un tensor `[1, seq, vocab]`.
pub fn logit_lens_top(
    features: &Tensor,
    output_weights: &Tensor,
    top_k: usize,
) -> Result<Vec<(u32, f32)>> {
    if features.rank() != 3 || output_weights.rank() != 2 {
        return Err(Error::new("logit lens requiere features 3D y pesos 2D"));
    }
    let seq_len = features.shape()[1];
    let embedding_dim = features.shape()[2];
    let vocab_size = output_weights.shape()[1];
    if output_weights.shape()[0] != embedding_dim || seq_len == 0 {
        return Err(Error::new("Dimensiones incompatibles en logit lens"));
    }
    let position = seq_len - 1;
    let feature = &features.data()[position * embedding_dim..(position + 1) * embedding_dim];
    let weights = output_weights.data();
    let mut logits = vec![0.0f32; vocab_size];
    for (vocab, logit) in logits.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        for (dimension, &value) in feature.iter().enumerate() {
            sum += value * weights[dimension * vocab_size + vocab];
        }
        *logit = sum;
    }
    top_predictions(&logits, top_k)
}

/// Top-k del último token de unos logits `[batch, seq, vocab]`.
pub fn logits_last_top(logits: &Tensor, top_k: usize) -> Result<Vec<(u32, f32)>> {
    if logits.rank() != 3 {
        return Err(Error::new("logits debe ser 3D"));
    }
    let seq_len = logits.shape()[1];
    let vocab_size = logits.shape()[2];
    if seq_len == 0 {
        return Err(Error::new("logits sin posiciones"));
    }
    let start = (seq_len - 1) * vocab_size;
    top_predictions(&logits.data()[start..start + vocab_size], top_k)
}

/// Softmax estable y top-k ordenado por probabilidad.
pub fn top_predictions(logits: &[f32], top_k: usize) -> Result<Vec<(u32, f32)>> {
    if logits.is_empty() {
        return Err(Error::new("top_predictions necesita al menos un logit"));
    }
    let count = top_k.clamp(1, logits.len());
    let mut maximum = f32::NEG_INFINITY;
    for &value in logits {
        if !value.is_finite() {
            return Err(Error::new("Los logits deben ser finitos"));
        }
        maximum = maximum.max(value);
    }
    let mut weights = Vec::with_capacity(logits.len());
    let mut total = 0.0f64;
    for &value in logits {
        let weight = ((f64::from(value) - f64::from(maximum)).exp()) as f32;
        total += f64::from(weight);
        weights.push(weight);
    }
    let mut ranked: Vec<(u32, f32)> = weights
        .iter()
        .enumerate()
        .map(|(index, &weight)| (index as u32, (f64::from(weight) / total) as f32))
        .collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(left.0.cmp(&right.0))
    });
    ranked.truncate(count);
    Ok(ranked)
}

fn truncate_label(label: &str, limit: usize) -> String {
    let mut output = String::new();
    for character in label.chars() {
        if output.chars().count() >= limit {
            output.push('…');
            break;
        }
        if character.is_control() {
            output.push('·');
        } else {
            output.push(character);
        }
    }
    if output.is_empty() {
        output.push('·');
    }
    output
}

/// Informe de terminal con tabla por capa y mapa de atención de una cabeza.
pub fn render_terminal(
    trace: &ActivationTrace,
    labels: &HashMap<u32, String>,
    layer: usize,
    head: usize,
) -> Result<String> {
    if trace.layers.is_empty() {
        return Err(Error::new("La traza no contiene capas"));
    }
    if layer >= trace.layers.len() {
        return Err(Error::new("--trace-layer fuera de rango"));
    }
    if head >= trace.layers[layer].block.attention.len() {
        return Err(Error::new("--trace-head fuera de rango"));
    }
    let label = |id: u32| {
        labels
            .get(&id)
            .cloned()
            .unwrap_or_else(|| format!("#{id}"))
    };
    let sequence = trace.tokens.len();
    let mut output = String::new();
    writeln!(output, "Traza de activaciones ({} tokens)", sequence)?;
    writeln!(output, "\nCapa  ‖res.in‖  ‖res.out‖   Δ       MLP activa  MLP máx  logit lens")?;
    for (index, layer_trace) in trace.layers.iter().enumerate() {
        let delta = layer_trace.residual_output_norm - layer_trace.residual_input_norm;
        let top = layer_trace
            .logit_lens
            .first()
            .map(|(id, probability)| format!("\"{}\" {:.1}%", truncate_label(&label(*id), 12), probability * 100.0))
            .unwrap_or_else(|| "-".to_string());
        writeln!(
            output,
            "{index:>4}  {:>8.2}  {:>9.2}  {delta:>+7.2}  {:>9.1}%  {:>7.2}  {top}",
            layer_trace.residual_input_norm,
            layer_trace.residual_output_norm,
            layer_trace.block.mlp.active_fraction * 100.0,
            layer_trace.block.mlp.max,
        )?;
    }

    let selected = &trace.layers[layer];
    writeln!(
        output,
        "\nAtención · capa {layer} · cabeza {head} (entropía media {:.2} nats)",
        selected.block.attention_entropy[head]
    )?;

    let window = 16usize.min(sequence);
    let start = sequence - window;
    let labels_window: Vec<String> = trace.tokens[start..]
        .iter()
        .map(|&id| truncate_label(&label(id), 5))
        .collect();
    let weights = &selected.block.attention[head];
    writeln!(
        output,
        "clave →   {}",
        labels_window
            .iter()
            .enumerate()
            .map(|(index, text)| format!("{index:>2}:{text:<5}"))
            .collect::<Vec<_>>()
            .join(" ")
    )?;
    for query in start..sequence {
        let row = &weights[query * sequence..(query + 1) * sequence];
        let maximum = row[..=query]
            .iter()
            .copied()
            .fold(0.0f32, f32::max)
            .max(1e-9);
        let mut line = String::new();
        for key in start..sequence {
            let value = if key <= query { row[key] } else { 0.0 };
            let level = ((value / maximum).sqrt() * 4.0).round() as usize;
            let symbol = ['·', '░', '▒', '▓', '█'][level.min(4)];
            write!(line, "  {symbol}  ")?;
        }
        writeln!(
            output,
            "{query:>2}:{:<5} {line}",
            truncate_label(&label(trace.tokens[query]), 5)
        )?;
    }

    writeln!(output, "\nPredicción final:")?;
    for (rank, (id, probability)) in trace.final_predictions.iter().enumerate() {
        writeln!(
            output,
            "  {}. \"{}\" (id={id}) {:.2}%",
            rank + 1,
            truncate_label(&label(*id), 24),
            probability * 100.0
        )?;
    }
    Ok(output)
}

fn json_escape(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if (character as u32) < 0x20 => {
                let _ = write!(output, "\\u{:04x}", character as u32);
            }
            character => output.push(character),
        }
    }
    output
}

fn number(value: f32) -> String {
    if value.is_finite() {
        format!("{value:.5}")
    } else {
        "0".to_string()
    }
}

fn trace_json(
    trace: &ActivationTrace,
    labels: &HashMap<u32, String>,
    model_name: &str,
) -> String {
    let mut json = String::new();
    json.push_str("{\"model\":\"");
    json.push_str(&json_escape(model_name));
    json.push_str("\",\"tokens\":[");
    for (index, token) in trace.tokens.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        let _ = write!(json, "{token}");
    }
    json.push_str("],\"tokenLabels\":[");
    for (index, token) in trace.tokens.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        let text = labels
            .get(token)
            .cloned()
            .unwrap_or_else(|| format!("#{token}"));
        json.push('"');
        json.push_str(&json_escape(&text));
        json.push('"');
    }
    json.push_str("],\"labels\":{");
    let mut ids: Vec<u32> = labels.keys().copied().collect();
    ids.sort_unstable();
    for (index, id) in ids.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        let _ = write!(json, "\"{id}\":\"{}\"", json_escape(&labels[id]));
    }
    json.push_str("},\"layers\":[");
    for (layer_index, layer) in trace.layers.iter().enumerate() {
        if layer_index > 0 {
            json.push(',');
        }
        let _ = write!(
            json,
            "{{\"residualIn\":{},\"residualOut\":{},\"layer\":{layer_index},\"attention\":[",
            number(layer.residual_input_norm),
            number(layer.residual_output_norm)
        );
        for (head_index, head) in layer.block.attention.iter().enumerate() {
            if head_index > 0 {
                json.push(',');
            }
            json.push('[');
            for (value_index, value) in head.iter().enumerate() {
                if value_index > 0 {
                    json.push(',');
                }
                let _ = write!(json, "{}", number(*value));
            }
            json.push(']');
        }
        json.push_str("],\"entropy\":[");
        for (index, value) in layer.block.attention_entropy.iter().enumerate() {
            if index > 0 {
                json.push(',');
            }
            let _ = write!(json, "{}", number(*value));
        }
        json.push_str("],\"mlp\":{");
        let mlp = &layer.block.mlp;
        let _ = write!(
            json,
            "\"mean\":{},\"max\":{},\"activeFraction\":{},\"top\":[",
            number(mlp.mean),
            number(mlp.max),
            number(mlp.active_fraction)
        );
        for (index, (neuron, value)) in mlp.top_neurons.iter().enumerate() {
            if index > 0 {
                json.push(',');
            }
            let _ = write!(json, "{{\"neuron\":{neuron},\"value\":{}}}", number(*value));
        }
        json.push_str("]},\"logitLens\":[");
        for (index, (id, probability)) in layer.logit_lens.iter().enumerate() {
            if index > 0 {
                json.push(',');
            }
            let _ = write!(json, "[{id},{}]", number(*probability));
        }
        json.push_str("]}");
    }
    json.push_str("],\"final\":[");
    for (index, (id, probability)) in trace.final_predictions.iter().enumerate() {
        if index > 0 {
            json.push(',');
        }
        let _ = write!(json, "[{id},{}]", number(*probability));
    }
    json.push_str("]}");
    json
}

/// HTML autocontenido con el mapa de atención, el logit lens y las métricas.
pub fn render_html(
    trace: &ActivationTrace,
    labels: &HashMap<u32, String>,
    model_name: &str,
) -> Result<String> {
    if trace.layers.is_empty() {
        return Err(Error::new("La traza no contiene capas"));
    }
    let data = trace_json(trace, labels, model_name);
    let mut html = String::from(HTML_START);
    html.push_str("const DATA = ");
    html.push_str(&data);
    html.push_str(";\n");
    html.push_str(HTML_SCRIPT);
    Ok(html)
}

pub fn write_html(
    path: &Path,
    trace: &ActivationTrace,
    labels: &HashMap<u32, String>,
    model_name: &str,
) -> Result<()> {
    let html = render_html(trace, labels, model_name)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, html)?;
    Ok(())
}

const HTML_START: &str = r#"<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Traza de activaciones · mini GPT</title>
<style>
  :root { color-scheme: dark; --bg:#10141c; --panel:#171d28; --line:#2a3444; --text:#e6edf7; --muted:#93a1b5; --accent:#5cc8ff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
  header { padding:18px 22px 10px; border-bottom:1px solid var(--line); }
  h1 { margin:0 0 4px; font-size:19px; font-weight:600; }
  .muted { color:var(--muted); }
  main { display:grid; grid-template-columns: minmax(320px, 1fr) 360px; gap:18px; padding:18px 22px 30px; align-items:start; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; }
  .controls { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin-bottom:12px; }
  label { display:flex; gap:6px; align-items:center; color:var(--muted); font-size:12px; }
  select, input[type=range] { accent-color: var(--accent); background:#0d121a; color:var(--text); border:1px solid var(--line); border-radius:7px; padding:5px 7px; }
  canvas { width:100%; height:auto; display:block; background:#0b0f16; border-radius:8px; }
  .stats { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap:8px; margin:10px 0 4px; }
  .stat { background:#111722; border:1px solid var(--line); border-radius:8px; padding:8px 10px; }
  .stat b { display:block; font-size:16px; font-weight:600; }
  .stat span { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--muted); margin:16px 0 8px; }
  .list { display:flex; flex-direction:column; gap:5px; }
  .row { display:grid; grid-template-columns: 22px 1fr auto; gap:8px; align-items:center; font-size:12.5px; }
  .row .bar { height:6px; border-radius:3px; background:#1d2838; overflow:hidden; }
  .row .bar i { display:block; height:100%; background:linear-gradient(90deg,#2d7fb8,#5cc8ff); }
  .row .prob { color:var(--muted); font-variant-numeric: tabular-nums; }
  .token { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; white-space:pre; }
  #tooltip { position:fixed; pointer-events:none; background:#0a0e14; border:1px solid var(--line); border-radius:6px; padding:4px 7px; font-size:12px; opacity:0; transition:opacity .08s; z-index:10; }
  .legend { display:flex; gap:4px; align-items:center; color:var(--muted); font-size:11px; margin-top:8px; }
  .legend i { width:18px; height:8px; border-radius:2px; display:inline-block; }
</style>
</head>
<body>
<header>
  <h1>Traza de activaciones</h1>
  <div class="muted" id="subtitle"></div>
</header>
<main>
  <section class="panel">
    <div class="controls">
      <label>Capa <select id="layer"></select></label>
      <label>Cabeza <select id="head"></select></label>
      <label><input type="checkbox" id="normalize" checked> normalizar por fila</label>
    </div>
    <canvas id="heatmap" width="760" height="760"></canvas>
    <div class="legend"><span>baja</span><i style="background:#101722"></i><i style="background:#1e3a55"></i><i style="background:#2d6491"></i><i style="background:#4198cf"></i><i style="background:#7ad8ff"></i><span>alta</span><span id="hover"></span></div>
    <div class="stats">
      <div class="stat"><span>‖residual‖ entrada</span><b id="res-in">–</b></div>
      <div class="stat"><span>‖residual‖ salida</span><b id="res-out">–</b></div>
      <div class="stat"><span>MLP activa</span><b id="mlp-active">–</b></div>
      <div class="stat"><span>MLP máximo</span><b id="mlp-max">–</b></div>
    </div>
  </section>
  <aside class="panel">
    <h2>Entropía por cabeza</h2>
    <div class="list" id="entropy"></div>
    <h2>Logit lens · última posición</h2>
    <div class="list" id="lens"></div>
    <h2>Neuronas MLP más activas</h2>
    <div class="list" id="neurons"></div>
    <h2>Predicción final</h2>
    <div class="list" id="final"></div>
  </aside>
</main>
<div id="tooltip"></div>
"#;

const HTML_SCRIPT: &str = r#"
const layerSelect = document.getElementById('layer');
const headSelect = document.getElementById('head');
const canvas = document.getElementById('heatmap');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');
const normalize = document.getElementById('normalize');
const subtitle = document.getElementById('subtitle');

const LABELS = DATA.labels;
function label(id) { return LABELS[String(id)] !== undefined ? LABELS[String(id)] : ('#' + id); }
function short(text, max) {
  const clean = String(text).replace(/[\n\r\t]/g, ' ');
  return clean.length > max ? clean.slice(0, max - 1) + '…' : clean;
}
subtitle.textContent = DATA.model + ' · ' + DATA.tokens.length + ' tokens · ' + DATA.layers.length + ' capas';

DATA.layers.forEach((_, index) => {
  const option = document.createElement('option');
  option.value = String(index);
  option.textContent = 'capa ' + index;
  layerSelect.appendChild(option);
});
layerSelect.value = String(DATA.layers.length - 1);

function headsOf(layer) { return DATA.layers[layer].attention.length; }
function refreshHeads() {
  const layer = Number(layerSelect.value);
  headSelect.innerHTML = '';
  for (let head = 0; head < headsOf(layer); head++) {
    const option = document.createElement('option');
    option.value = String(head);
    option.textContent = 'cabeza ' + head;
    headSelect.appendChild(option);
  }
  headSelect.value = '0';
}

function color(value) {
  // Azul oscuro -> cian claro.
  const t = Math.max(0, Math.min(1, value));
  const r = Math.round(16 + t * 106);
  const g = Math.round(23 + t * 193);
  const b = Math.round(34 + t * 221);
  return 'rgb(' + r + ',' + g + ',' + b + ')';
}

function draw() {
  const layerIndex = Number(layerSelect.value);
  const headIndex = Number(headSelect.value);
  const layer = DATA.layers[layerIndex];
  const weights = layer.attention[headIndex];
  const n = DATA.tokens.length;
  const dpr = window.devicePixelRatio || 1;
  const size = canvas.clientWidth || 760;
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const marginLeft = Math.min(120, size * 0.22);
  const marginTop = 34;
  const grid = Math.max(40, size - marginLeft - 12);
  const cell = grid / Math.max(1, n);
  ctx.clearRect(0, 0, size, size);
  ctx.font = '10px ui-monospace, monospace';
  ctx.textBaseline = 'middle';

  const step = Math.max(1, Math.ceil(n / 28));
  for (let row = 0; row < n; row++) {
    const rowWeights = weights.slice(row * n, (row + 1) * n);
    let max = 0;
    for (let col = 0; col <= row; col++) max = Math.max(max, rowWeights[col]);
    for (let col = 0; col < n; col++) {
      const value = col <= row ? rowWeights[col] : 0;
      const normalized = normalize.checked ? value / Math.max(1e-9, max) : value;
      ctx.fillStyle = col <= row ? color(Math.sqrt(normalized)) : '#0b0f16';
      ctx.fillRect(marginLeft + col * cell, marginTop + row * cell, Math.ceil(cell), Math.ceil(cell));
    }
    if (row % step === 0) {
      ctx.fillStyle = '#93a1b5';
      ctx.textAlign = 'right';
      ctx.fillText(short(DATA.tokenLabels[row], 14), marginLeft - 6, marginTop + (row + 0.5) * cell);
    }
  }
  ctx.textAlign = 'center';
  for (let col = 0; col < n; col += step) {
    ctx.save();
    ctx.translate(marginLeft + (col + 0.5) * cell, marginTop - 8);
    ctx.rotate(-Math.PI / 4);
    ctx.fillStyle = '#93a1b5';
    ctx.fillText(short(DATA.tokenLabels[col], 12), 0, 0);
    ctx.restore();
  }

  document.getElementById('res-in').textContent = layer.residualIn.toFixed(2);
  document.getElementById('res-out').textContent = layer.residualOut.toFixed(2);
  document.getElementById('mlp-active').textContent = (layer.mlp.activeFraction * 100).toFixed(1) + '%';
  document.getElementById('mlp-max').textContent = layer.mlp.max.toFixed(2);

  const entropy = document.getElementById('entropy');
  entropy.innerHTML = '';
  layer.entropy.forEach((value, index) => {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<span class="muted">' + index + '</span><div class="bar"><i style="width:' +
      Math.min(100, (value / Math.max(0.001, Math.log(n))) * 100) + '%"></i></div><span class="prob">' +
      value.toFixed(2) + '</span>';
    entropy.appendChild(row);
  });

  const lens = document.getElementById('lens');
  lens.innerHTML = '';
  layer.logitLens.forEach(([id, probability]) => {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<span class="muted">·</span><span class="token">' + escapeHtml(short(label(id), 22)) +
      '</span><span class="prob">' + (probability * 100).toFixed(2) + '%</span>';
    lens.appendChild(row);
  });

  const neurons = document.getElementById('neurons');
  neurons.innerHTML = '';
  layer.mlp.top.forEach(({ neuron, value }) => {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<span class="muted">' + neuron + '</span><div class="bar"><i style="width:' +
      Math.min(100, Math.abs(value) * 25) + '%"></i></div><span class="prob">' + value.toFixed(3) + '</span>';
    neurons.appendChild(row);
  });

  const finalList = document.getElementById('final');
  finalList.innerHTML = '';
  DATA.final.forEach(([id, probability], index) => {
    const row = document.createElement('div');
    row.className = 'row';
    row.innerHTML = '<span class="muted">' + (index + 1) + '</span><span class="token">' + escapeHtml(short(label(id), 22)) +
      '</span><span class="prob">' + (probability * 100).toFixed(2) + '%</span>';
    finalList.appendChild(row);
  });

  canvas.onmousemove = (event) => {
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const col = Math.floor((x - marginLeft) / cell);
    const row = Math.floor((y - marginTop) / cell);
    if (row < 0 || row >= n || col < 0 || col >= n || col > row) {
      tooltip.style.opacity = 0;
      return;
    }
    const value = weights[row * n + col];
    tooltip.textContent = short(DATA.tokenLabels[row], 18) + ' → ' + short(DATA.tokenLabels[col], 18) +
      ': ' + value.toFixed(4);
    tooltip.style.left = (event.clientX + 12) + 'px';
    tooltip.style.top = (event.clientY + 12) + 'px';
    tooltip.style.opacity = 1;
  };
  canvas.onmouseleave = () => { tooltip.style.opacity = 0; };
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

layerSelect.onchange = () => { refreshHeads(); draw(); };
headSelect.onchange = draw;
normalize.onchange = draw;
window.addEventListener('resize', draw);
refreshHeads();
draw();
"#;
