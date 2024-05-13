from sentence_transformers import SentenceTransformer, util
from PIL import Image

import weaviate

from transformers import CLIPProcessor, CLIPModel
import torch

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_image(image_path: str, model, preprocess):
    image = Image.open(image_path)
    inputs = preprocess(images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    image_embedding = outputs.detach().cpu().numpy().squeeze()
    return image_embedding

#V4
client = weaviate.connect_to_local()

query_picture = "resources/eiffel-tower-night.jpg"
Image.open(query_picture).show(title="Query for...")

embedding=encode_image(query_picture, model, preprocess)

# Convert the numpy.ndarray to a list
embedding_list = embedding.tolist()

# Get the collection
images = client.collections.get("OpenAIImagesV4")

response = images.query.near_vector(
    near_vector=embedding_list,
    return_properties=["name", "path"],
    limit=1
)

first_obj = response.objects[0]
print(first_obj)
path_value = first_obj.properties["path"]
Image.open(path_value).show(title="Response")

client.close()