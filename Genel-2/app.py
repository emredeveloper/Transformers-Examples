import os
import polars as pl
from huggingface_hub import login

# Log in to Hugging Face using the environment variable
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if hf_token:
    login(hf_token)
else:
    print("Warning: HUGGINGFACE_TOKEN environment variable not found. You may not have access to private models.")

# Hugging Face'ten doğru dosyayı yüklemek için veri kümesinin yolunu doğru şekilde kontrol edin
try:
    df = pl.read_parquet('hf://datasets/HuggingFaceM4/the_cauldron/textcaps/train-00011-of-00012-baf9399db4a7051d.parquet')
    print("Dataset loaded!")
    print(df.head())
except Exception as e:
    print("An error occurred while loading the dataset:", e)
