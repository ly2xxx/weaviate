https://weaviate.io/developers/weaviate/search/image

### Prerequisites
- Python 3.9 or higher
- Git
- Docker desktop

### Installation
Create a virtual environment :
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Install the required dependencies in the virtual environment :
```bash
pip install -r requirements.txt
```

### Running
```bash
docker compose -f "docker-compose.yml" up -d --build
```

```bash
python 05-upload_images_v4.py
```

```bash
python 06-query_images_v4.py
```

https://nayakpplaban.medium.com/ask-questions-to-your-images-using-langchain-and-python-1aeb30f38751 (Salesforce/blip-image-captioning-large)

https://medium.com/aimonks/semantic-search-hands-on-text-to-image-search-using-clip-and-faiss-c1d387f66d00 (openai/clip-vit-base-patch32)