from sentence_transformers import SentenceTransformer, util
from PIL import Image

import weaviate

from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

text = "Dog"
inputs = processor(text=text, return_tensors="pt")
with torch.no_grad():
    text_embedding = model.get_text_features(**inputs).detach().numpy().squeeze()

print(text_embedding)

#V4
client = weaviate.connect_to_local()

# Convert the numpy.ndarray to a list
embedding_list = text_embedding.tolist()

# Get the collection
images = client.collections.get("OpenAIImagesV4")

number_of_results = 1
response = images.query.near_vector(
    near_vector=embedding_list,
    return_properties=["name", "path"],
    limit=number_of_results
)

def display_picture(picture_obj):
    print(picture_obj)
    path_value = picture_obj.properties["path"]
    Image.open(path_value).show(title="Response")

for i in range(number_of_results):
    display_picture(response.objects[i])

client.close()