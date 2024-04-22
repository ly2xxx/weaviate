from transformers import AutoModel
#https://huggingface.co/sentence-transformers/clip-ViT-B-32
# Define the model name
model_name = "sentence-transformers/clip-ViT-L-14"

# Load the model
model = AutoModel.from_pretrained(model_name)

# Save the model to a local directory
model.save_pretrained("clip-ViT-L-14-local")

print(f"Model '{model_name}' downloaded and saved locally.")
