//! PEFT (Parameter-Efficient Fine-Tuning) format conversion utilities
//!
//! This module provides functionality to convert between HuggingFace PEFT format
//! and candle-lora format for seamless integration with PEFT adapters.

use candle_core::{Device, Result, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

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
    println!("🔄 Converting PEFT to candle-lora format...");
    println!("   📂 Input: {}", peft_path);
    println!("   📁 Output: {}", output_path);
    println!("   🏷️  Prefix: {}", prefix);

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

        println!("   ✅ Converted LoRA pair {}", idx);
    }

    // Save as safetensors
    candle_core::safetensors::save(&candle_tensors, output_path)?;

    println!("✅ Conversion complete!");
    println!("   📁 Saved: {}", output_path);
    println!("   🔢 Total LoRA pairs: {}", lora_pairs.len());

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
            if let Ok(config) = serde_json::from_str::<PeftConfig>(&config_str) {
                println!("📋 PEFT Config:");
                println!("   r: {}", config.r);
                println!("   alpha: {}", config.lora_alpha);
                println!("   target_modules: {:?}", config.target_modules);
            }
        }
    }

    convert_peft_to_candle_lora(weights_path.to_str().unwrap(), output_path, prefix, device)
}
