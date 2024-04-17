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
python 05-upload_images.py
```

```bash
python 06-query_images_v4.py
```