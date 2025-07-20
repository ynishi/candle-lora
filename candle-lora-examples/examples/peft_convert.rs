//! Example of converting PEFT format LoRA weights to candle-lora format

use candle_core::{DType, Device, Tensor};
use candle_lora::convert_peft_to_candle_lora;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;

    // Create a dummy PEFT format file
    let peft_path = "dummy_peft_adapter.safetensors";
    let output_path = "converted_candle_lora.safetensors";

    println!("📝 Creating dummy PEFT format LoRA weights...");

    let mut peft_tensors = HashMap::new();

    // Simulate typical PEFT naming pattern
    let layers = vec![
        "base_model.model.layers.0.self_attn.q_proj",
        "base_model.model.layers.0.self_attn.k_proj",
        "base_model.model.layers.0.self_attn.v_proj",
        "base_model.model.layers.1.self_attn.q_proj",
    ];

    for layer in &layers {
        // LoRA rank = 16, hidden_size = 128 (small for testing)
        let lora_a = Tensor::randn(0.0, 0.02, (128, 16), &device)?;
        let lora_b = Tensor::zeros((16, 128), DType::F32, &device)?;

        peft_tensors.insert(format!("{}.lora_A.weight", layer), lora_a);
        peft_tensors.insert(format!("{}.lora_B.weight", layer), lora_b);
    }

    // Save as PEFT format
    candle_core::safetensors::save(&peft_tensors, peft_path)?;
    println!("✅ Created dummy PEFT file: {}", peft_path);
    println!("   Total tensors: {}", peft_tensors.len());
    println!("   LoRA pairs: {}", peft_tensors.len() / 2);

    // Convert to candle-lora format
    println!("\n🔄 Converting to candle-lora format...");
    convert_peft_to_candle_lora(peft_path, output_path, "lora_llama", &device)?;

    // Load and verify the converted file
    println!("\n📋 Verifying converted file...");
    let converted = candle_core::safetensors::load(output_path, &device)?;

    println!("Converted tensors:");
    for (name, tensor) in &converted {
        println!("  {} - shape: {:?}", name, tensor.dims());
    }

    // Clean up
    std::fs::remove_file(peft_path)?;
    std::fs::remove_file(output_path)?;
    println!("\n🧹 Cleaned up temporary files");

    println!("\n✅ PEFT conversion example completed successfully!");

    Ok(())
}
