import os
from huggingface_hub import HfApi
import requests
import io
import pyarrow.parquet as pq
import psutil

def print_mem(label=""):
    process = psutil.Process(os.getpid())
    print(f"[{label}] Memory: {process.memory_info().rss / 1e9:.2f} GB")

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO = "rookierufus/ego10k-vjepa-latents"

api = HfApi(token=HF_TOKEN)
files = [f for f in api.list_repo_files(repo_id=REPO, repo_type="dataset", token=HF_TOKEN) if f.endswith('.parquet')]
print(f"Found {len(files)} parquet files.")

if files:
    filename = files[0]
    print(f"Downloading {filename}...")
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{filename}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    print_mem("Before request")
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    print_mem("After request")
    
    print(f"Downloaded {len(resp.content)} bytes.")
    bytes_io = io.BytesIO(resp.content)
    print_mem("Before pyarrow table")
    table = pq.read_table(bytes_io)
    print_mem("After pyarrow table")
    
    print(f"Table rows: {table.num_rows}")
    
    batch = table.to_batches()[0]
    d = batch.to_pydict()
    print("Keys:", d.keys())
    print("First item factory_id:", d['factory_id'][0])
    print("First item latent_bytes len:", len(d['latent_bytes'][0]))
