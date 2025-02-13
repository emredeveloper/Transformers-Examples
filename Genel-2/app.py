import polars as pl

from huggingface_hub import login

# Hugging Face'e giriş yapmak
login("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")



# Hugging Face'ten doğru dosyayı yüklemek için veri kümesinin yolunu doğru şekilde kontrol edin
try:
    df = pl.read_parquet('hf://datasets/HuggingFaceM4/the_cauldron/textcaps/train-00011-of-00012-baf9399db4a7051d.parquet')
    print("Veri kümesi yüklendi!")
    print(df.head())
except Exception as e:
    print("Veri kümesi yüklenirken bir hata oluştu:", e)
