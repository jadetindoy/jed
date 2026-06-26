"""
test_model.py
-------------
A script to instantiate the JedAI transformer model, print its architecture,
measure parameter count, perform a causal forward pass, and test token generation.
"""

import sys
import io
import torch
import yaml
from pathlib import Path

# Force UTF-8 output on Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from model.config import ModelConfig
from model.model import JedAI

def main():
    print("=== Initializing JedAI Transformer Model ===")

    # 1. Load config
    config_path = Path("configs/50m.yaml")
    if config_path.exists():
        print(f"Loading configuration from {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        print("Config file not found, using default configuration parameters")
        cfg = {
            "vocab_size": 32000,
            "d_model": 512,
            "n_heads": 8,
            "n_layers": 8,
            "max_seq_len": 1024,
        }

    # Use a slightly smaller config for quick local testing if desired,
    # but let's use the actual 50M settings.
    model_cfg = ModelConfig(**{
        k: v for k, v in cfg.items()
        if k in ModelConfig.__dataclass_fields__
    })
    
    print("\n--- Model Configuration ---")
    for k, v in model_cfg.to_dict().items():
        print(f"  {k}: {v}")

    # 2. Instantiate model
    print("\nInstantiating JedAI model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JedAI(model_cfg).to(device)
    
    # 3. Print Architecture & Params
    print("\n--- Model Architecture ---")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_emb_params = model.num_parameters(non_embedding=True)
    
    print(f"\n  Total parameters:      {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters:  {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"  Non-embedding params:  {non_emb_params:,} ({non_emb_params/1e6:.2f}M)")

    # 4. Dummy Forward Pass
    print("\n--- Running Causal Forward Pass ---")
    batch_size = 2
    seq_len = 128
    
    # Generate random input tokens
    dummy_input = torch.randint(0, model_cfg.vocab_size, (batch_size, seq_len), device=device)
    print(f"Input shape: {dummy_input.shape} (batch_size={batch_size}, seq_len={seq_len})")
    
    # Forward pass with labels to get logits and loss
    logits, loss = model(dummy_input, labels=dummy_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Self-prediction loss: {loss.item():.4f}")
    
    assert logits.shape == (batch_size, seq_len, model_cfg.vocab_size), "Logits shape mismatch!"
    assert loss is not None, "Loss should be computed!"
    print("Causal Forward Pass OK!")

    # 5. Token Generation Demonstration
    print("\n--- Testing Autoregressive Token Generation ---")
    prompt_len = 5
    prompt = torch.randint(0, model_cfg.vocab_size, (1, prompt_len), device=device)
    print(f"Prompt input token IDs: {prompt.tolist()[0]}")
    
    max_new_tokens = 10
    generated = model.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.8,
        top_k=40,
    )
    print(f"Generated output token IDs (len={generated.shape[1]}): {generated.tolist()[0]}")
    assert generated.shape == (1, prompt_len + max_new_tokens), "Generated shape mismatch!"
    print("Token Generation OK!")
    
    print("\nAll model verification tests passed successfully!")

if __name__ == "__main__":
    main()
