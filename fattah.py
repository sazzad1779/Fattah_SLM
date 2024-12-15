from src.Fattah import fattahModel, fattahForCausalLM, fattahConfig

# Step 1: Initialize the configuration
configuration = fattahConfig(
    vocab_size=32000,          # Adjust vocab size based on your tokenizer
    hidden_size=576,           # Embedding dimension
    num_hidden_layers=6,       # Number of transformer layers
    num_key_value_heads=8,     # Number of attention heads
    intermediate_size=1536,    # Dimension of feedforward layers
    max_position_embeddings=1024,  # Maximum sequence length
)

# Step 2: Initialize the base model
model = fattahModel(configuration)

# Step 3: Save the base model and configuration
save_path = "./fattah_base_model"
model.save_pretrained(save_path)       # Save the model weights
configuration.save_pretrained(save_path)  # Save the configuration

print(f"Base model and configuration saved at {save_path}")
