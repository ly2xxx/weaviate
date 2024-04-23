from sentence_transformers import SentenceTransformer
import re

def extract_model_name(text):
  pattern = r'/([^/]+)$'

  match = re.search(pattern, text)

  if match:
    return match.group(1)

#https://huggingface.co/sentence-transformers/clip-ViT-B-32
# Define the model name
model_name = "sentence-transformers/clip-ViT-B-16"

# Load the model
model = SentenceTransformer(model_name)

# Save the model to a local directory
short_name = extract_model_name(model_name)
model.save(short_name)

print(f"Model '{short_name}' downloaded and saved locally.")
