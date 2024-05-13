import weaviate
from weaviate.classes.config import Property, DataType
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import PIL
from typing import List, Union
from transformers import CLIPProcessor, CLIPModel
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

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

# def encode_images(images: Union[List[str], List[PIL.Image.Image]], batch_size: int):
#     def transform_fn(el):
#         if isinstance(el['image'], PIL.Image.Image):
#             imgs = el['image']
#         else:
#             imgs = [Image().decode_example(_) for _ in el['image']]
#         return preprocess(images=imgs, return_tensors='pt')
        
#     dataset = Dataset.from_dict({'image': images})
#     dataset = dataset.cast_column('image',Image(decode=False)) if isinstance(images[0], str) else dataset       
#     dataset.set_format('torch')
#     dataset.set_transform(transform_fn)
#     dataloader = DataLoader(dataset, batch_size=batch_size)
#     image_embeddings = []
#     pbar = tqdm(total=len(images) // batch_size, position=0)
#     with torch.no_grad():
#         for batch in dataloader:
#             batch = {k:v.to(device) for k,v in batch.items()}
#             image_embeddings.extend(model.get_image_features(**batch).detach().cpu().numpy())
#             pbar.update(1)
#         pbar.close()
#     return np.stack(image_embeddings)

#V4
client = weaviate.connect_to_local()
#https://weaviate.io/developers/weaviate/manage-data/collections
# clean up
COLLECTION_NAME = "OpenAIImagesV4"
client.collections.delete(COLLECTION_NAME)

client.collections.create(
    COLLECTION_NAME, 
    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="path", data_type=DataType.TEXT),
    ]
)

images = ['cat.jpg', 'eiffel-tower-day.jpg', 'two_dogs_in_snow.jpg'] 

images_collection = client.collections.get(COLLECTION_NAME)
#batch insert
with images_collection.batch.dynamic() as batch:
    batch.batch_size=10
    for imgurl in images:
        img_path = f"resources/{imgurl}"
        img_emb = encode_image(img_path, model, preprocess)#np.array(encode_images([img_path],1))
        print(img_emb)
        batch.add_object(properties={
            "name": imgurl,
            "path": img_path,
        },
        vector=img_emb.tolist())
    #single insert
    # uuid = images_collection.data.insert(
    #     properties={
    #         "name": imgurl,
    #         "path": img_path,
    #     },
    #     vector=img_emb.tolist()
    # )

    # print(uuid)    

client.close()
