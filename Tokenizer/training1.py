import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from typing import List


# Örnek Türkçe metinler
texts = [
    "Merhaba, nasılsın?",
    "Bugün hava çok güzel.",
    "Python programlama dili çok popüler.",
    "Derin öğrenme, yapay zekanın bir dalıdır.",
    "Python, metin işlemede önemli bir adımdır."
]



class TurkishTokenizer:
    def __init__(self):
        self.vocab = {}  # Token -> ID
        self.id_to_token = {}  # ID -> Token
        self.next_id = 0  # Yeni token ID'si
        self.unk_token = "<UNK>"  # Bilinmeyen token
        self.pad_token = "<PAD>"  # Dolgu tokenı
        self.special_tokens = [self.unk_token, self.pad_token]

        # Özel tokenları ekle
        for token in self.special_tokens:
            self.add_token(token)

    def add_token(self, token: str) -> int:
        """
        Yeni bir token ekler ve bir ID atar.
        """
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]

    def tokenize(self, text: str) -> List[int]:
        """
        Metni tokenlara ayırır ve token ID'lerini döndürür.
        """
        # Metni küçük harfe çevir ve noktalama işaretlerini ayır
        text = text.lower()
        tokens = re.findall(r"\w+|\S", text)  # Kelimeler ve noktalama işaretleri
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.unk_token])  # Bilinmeyen token
        return token_ids

    def detokenize(self, token_ids: List[int]) -> str:
        """
        Token ID'lerini metne dönüştürür.
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append(self.unk_token)
        return " ".join(tokens)

    def build_vocab(self, texts: List[str]):
        """
        Metinler üzerinden kelime dağarcığı oluşturur.
        """
        counter = Counter()
        for text in texts:
            text = text.lower()
            tokens = re.findall(r"\w+|\S", text)
            counter.update(tokens)

        # En sık kullanılan tokenları ekle
        for token, _ in counter.most_common():
            self.add_token(token)
            
            
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        token_ids = self.tokenizer.tokenize(text)
        return torch.tensor(token_ids, dtype=torch.long)

# Tokenizer'ı ve veri setini oluştur
tokenizer = TurkishTokenizer()
tokenizer.build_vocab(texts)  # Kelime dağarcığını oluştur

dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Tokenizer'ı test et
test_text = "Bugün hava çok güzel."
token_ids = tokenizer.tokenize(test_text)
print(f"Token ID'leri: {token_ids}")

# Token ID'lerini metne dönüştür
decoded_text = tokenizer.detokenize(token_ids)
print(f"Çözülen metin: {decoded_text}")

print(dataset)
# DataLoader üzerinden örnekler al
for batch in dataloader:
    print("Batch:", batch)