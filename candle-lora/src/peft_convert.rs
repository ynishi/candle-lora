//! PEFT (Parameter-Efficient Fine-Tuning) format conversion utilities
//!
//! This module provides functionality to convert between HuggingFace PEFT format
//! and candle-lora format for seamless integration with PEFT adapters.

use candle_core::{DType, Device, Result, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// candle-lora naming prefixes for different layer types
/// Based on: https://github.com/EricLBuehler/candle-lora/blob/main/candle-lora-transformers/src/llama.rs
#[derive(Debug, Clone)]
pub enum CandleLoraPrefix {
    /// For embedding and lm_head layers (vocab_size related)
    Llama,
    /// For CausalSelfAttention layers (q_proj, k_proj, v_proj, o_proj)
    LlamaCsa,
    /// For transformer Block layers
    LlamaBlock,
}

impl CandleLoraPrefix {
    /// Get the string representation for VarBuilder path
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Llama => "lora_llama",
            Self::LlamaCsa => "lora_llama_csa",
            Self::LlamaBlock => "lora_llama_block",
        }
    }

    /// Determine prefix based on PEFT layer name
    pub fn from_peft_layer_name(name: &str) -> Self {
        if name.contains("embed_tokens") || name.contains("lm_head") {
            Self::Llama
        } else if name.contains("self_attn")
            && (name.contains("q_proj")
                || name.contains("k_proj")
                || name.contains("v_proj")
                || name.contains("o_proj"))
        {
            Self::LlamaCsa
        } else {
            Self::LlamaBlock
        }
    }
}

/// PEFT adapter_config.json structure
#[derive(Debug, Deserialize)]
pub struct PeftConfig {
    pub r: usize,
    pub lora_alpha: f64,
    #[serde(default)]
    pub lora_dropout: f64,
    pub target_modules: Vec<String>,
    pub peft_type: String,
    #[serde(default)]
    pub base_model_name_or_path: String,
}

/// Convert PEFT format LoRA weights to candle-lora format
///
/// This function takes a PEFT format safetensors file and converts it to
/// the candle-lora naming convention.
///
/// # Arguments
/// * `peft_path` - Path to PEFT format safetensors file (e.g., adapter_model.safetensors)
/// * `output_path` - Path where the converted safetensors will be saved
/// * `prefix` - Prefix for the converted tensors (e.g., "lora_llama")
/// * `device` - Device to load tensors on
///
/// # Example
/// ```no_run
/// use candle_core::Device;
/// use candle_lora::convert_peft_to_candle_lora;
///
/// let device = Device::Cpu;
/// convert_peft_to_candle_lora(
///     "path/to/adapter_model.safetensors",
///     "path/to/converted.safetensors",
///     "lora_llama",
///     &device
/// ).unwrap();
/// ```
pub fn convert_peft_to_candle_lora(
    peft_path: &str,
    output_path: &str,
    prefix: &str,
    device: &Device,
) -> Result<()> {
    // Load the PEFT safetensors file
    let peft_tensors = candle_core::safetensors::load(peft_path, device)?;

    // Group LoRA pairs
    let mut lora_pairs: Vec<(String, Tensor, Tensor)> = Vec::new();
    let mut processed_keys = std::collections::HashSet::new();

    for (name, tensor) in peft_tensors.iter() {
        if name.contains(".lora_A.weight") && !processed_keys.contains(name) {
            let base_name = name.replace(".lora_A.weight", "");
            let b_name = format!("{}.lora_B.weight", base_name);

            if let Some(lora_b_tensor) = peft_tensors.get(&b_name) {
                processed_keys.insert(name.clone());
                processed_keys.insert(b_name.clone());
                lora_pairs.push((base_name, tensor.clone(), lora_b_tensor.clone()));
            }
        }
    }

    // Sort for consistent ordering
    lora_pairs.sort_by(|a, b| a.0.cmp(&b.0));

    // Convert to candle-lora format
    let mut candle_tensors = HashMap::new();

    for (idx, (_peft_name, lora_a, lora_b)) in lora_pairs.iter().enumerate() {
        let a_name = format!("{}.a{}.weight", prefix, idx);
        let b_name = format!("{}.b{}.weight", prefix, idx);

        candle_tensors.insert(a_name, lora_a.clone());
        candle_tensors.insert(b_name, lora_b.clone());
    }

    // Save as safetensors
    candle_core::safetensors::save(&candle_tensors, output_path)?;

    Ok(())
}

/// Convert PEFT directory to candle-lora format
///
/// This function takes a PEFT format directory (containing adapter_config.json
/// and adapter_model.safetensors) and converts it to candle-lora format.
///
/// # Arguments
/// * `peft_dir` - Path to PEFT format directory
/// * `output_path` - Path where the converted safetensors will be saved
/// * `prefix` - Prefix for the converted tensors (e.g., "lora_llama")
/// * `device` - Device to load tensors on
pub fn convert_peft_dir_to_candle_lora(
    peft_dir: &str,
    output_path: &str,
    prefix: &str,
    device: &Device,
) -> Result<()> {
    let peft_path = Path::new(peft_dir);

    // Check for adapter files
    let adapter_path = peft_path.join("adapter_model.safetensors");
    let adapter_path_alt = peft_path.join("adapter.safetensors");

    let weights_path = if adapter_path.exists() {
        adapter_path
    } else if adapter_path_alt.exists() {
        adapter_path_alt
    } else {
        return Err(candle_core::Error::Msg(
            "No adapter weights found (tried adapter_model.safetensors and adapter.safetensors)"
                .to_string(),
        ));
    };

    // Load and display config if available
    let config_path = peft_path.join("adapter_config.json");
    if config_path.exists() {
        if let Ok(config_str) = std::fs::read_to_string(&config_path) {
            if let Ok(_config) = serde_json::from_str::<PeftConfig>(&config_str) {
                // Config loaded successfully
            }
        }
    }

    convert_peft_to_candle_lora(weights_path.to_str().unwrap(), output_path, prefix, device)
}

/// Convert PEFT format to candle-lora format with layer type awareness
///
/// This advanced conversion function automatically categorizes LoRA weights by layer type
/// and assigns appropriate prefixes for candle-lora compatibility.
///
/// # Arguments
/// * `peft_path` - Path to PEFT format safetensors file
/// * `output_path` - Path where the converted safetensors will be saved
/// * `device` - Device to load tensors on
/// * `add_dummy_embeddings` - Whether to add dummy embedding tensors if not present
pub fn convert_peft_to_candle_lora_typed(
    peft_path: &str,
    output_path: &str,
    device: &Device,
    add_dummy_embeddings: bool,
) -> Result<()> {
    // Load the PEFT safetensors file
    let peft_tensors = candle_core::safetensors::load(peft_path, device)?;

    // Group LoRA pairs
    let mut lora_pairs: Vec<(String, Tensor, Tensor)> = Vec::new();
    let mut processed_keys = std::collections::HashSet::new();

    for (name, tensor) in peft_tensors.iter() {
        if name.contains(".lora_A.weight") && !processed_keys.contains(name) {
            let base_name = name.replace(".lora_A.weight", "");
            let b_name = format!("{}.lora_B.weight", base_name);

            if let Some(lora_b_tensor) = peft_tensors.get(&b_name) {
                processed_keys.insert(name.clone());
                processed_keys.insert(b_name.clone());
                lora_pairs.push((base_name, tensor.clone(), lora_b_tensor.clone()));
            }
        }
    }

    // Group weights by prefix type
    let mut llama_weights = Vec::new();
    let mut llama_csa_weights = Vec::new();
    let mut llama_block_weights = Vec::new();

    for (key, lora_a, lora_b) in &lora_pairs {
        let prefix_type = CandleLoraPrefix::from_peft_layer_name(key);
        match prefix_type {
            CandleLoraPrefix::Llama => llama_weights.push((key, lora_a, lora_b)),
            CandleLoraPrefix::LlamaCsa => llama_csa_weights.push((key, lora_a, lora_b)),
            CandleLoraPrefix::LlamaBlock => llama_block_weights.push((key, lora_a, lora_b)),
        }
    }

    // Sort each group for consistent ordering
    llama_weights.sort_by_key(|(key, _, _)| *key);
    llama_csa_weights.sort_by_key(|(key, _, _)| *key);
    llama_block_weights.sort_by_key(|(key, _, _)| *key);

    // Convert to candle-lora format
    let mut candle_tensors = HashMap::new();

    // Helper closure to process each group
    let mut process_group = |weights: Vec<(&String, &Tensor, &Tensor)>,
                             prefix: CandleLoraPrefix| {
        let mut counter = 0;
        for (_key, lora_a, lora_b) in weights {
            let a_name = format!("{}.a{}.weight", prefix.as_str(), counter);
            let b_name = format!("{}.b{}.weight", prefix.as_str(), counter);

            candle_tensors.insert(a_name.clone(), lora_a.clone());
            candle_tensors.insert(b_name.clone(), lora_b.clone());
            counter += 1;
        }
    };

    if !llama_weights.is_empty() {
        process_group(llama_weights, CandleLoraPrefix::Llama);
    }
    if !llama_csa_weights.is_empty() {
        process_group(llama_csa_weights, CandleLoraPrefix::LlamaCsa);
    }
    if !llama_block_weights.is_empty() {
        process_group(llama_block_weights, CandleLoraPrefix::LlamaBlock);
    }

    // Add dummy embedding LoRA tensors if not present and requested
    if add_dummy_embeddings {
        let has_llama_tensors = candle_tensors.keys().any(|k| k.starts_with("lora_llama."));
        if !has_llama_tensors {
            // Default sizes for TinyLlama, but should be configurable
            let vocab_size = 32000;
            let hidden_size = 2048;
            let rank = 4; // Default rank, should match actual config

            let dummy_a = Tensor::zeros((rank, vocab_size), DType::F32, device)?;
            let dummy_b = Tensor::zeros((hidden_size, rank), DType::F32, device)?;

            candle_tensors.insert("lora_llama.a0.weight".to_string(), dummy_a);
            candle_tensors.insert("lora_llama.b0.weight".to_string(), dummy_b);
        }
    }

    // Save as safetensors
    candle_core::safetensors::save(&candle_tensors, output_path)?;

    Ok(())
}

/// Convert PEFT directory to candle-lora format with layer type awareness
///
/// This function takes a PEFT format directory and converts it using the typed conversion.
///
/// # Arguments
/// * `peft_dir` - Path to PEFT format directory
/// * `output_path` - Path where the converted safetensors will be saved
/// * `device` - Device to load tensors on
/// * `add_dummy_embeddings` - Whether to add dummy embedding tensors if not present
pub fn convert_peft_dir_to_candle_lora_typed(
    peft_dir: &str,
    output_path: &str,
    device: &Device,
    add_dummy_embeddings: bool,
) -> Result<()> {
    let peft_path = Path::new(peft_dir);

    // Check for adapter files
    let adapter_path = peft_path.join("adapter_model.safetensors");
    let adapter_path_alt = peft_path.join("adapter.safetensors");

    let weights_path = if adapter_path.exists() {
        adapter_path
    } else if adapter_path_alt.exists() {
        adapter_path_alt
    } else {
        return Err(candle_core::Error::Msg(
            "No adapter weights found (tried adapter_model.safetensors and adapter.safetensors)"
                .to_string(),
        ));
    };

    // Load and display config if available
    let config_path = peft_path.join("adapter_config.json");
    if config_path.exists() {
        if let Ok(config_str) = std::fs::read_to_string(&config_path) {
            if let Ok(_config) = serde_json::from_str::<PeftConfig>(&config_str) {
                // Config loaded successfully
            }
        }
    }

    convert_peft_to_candle_lora_typed(
        weights_path.to_str().unwrap(),
        output_path,
        device,
        add_dummy_embeddings,
    )
}
