<<<<<<< HEAD
from typing import List

class SimpleTokenizer:
    """
    Basit bir tokenizer sınıfı. Bu sınıf, metni tokenlara ayırır ve tokenları tekrar metne dönüştürür.
    """

    def __init__(self):
        """
        Tokenizer'ı başlatır. Bu örnekte, boşluklara göre tokenlara ayırma işlemi yapılır.
        """
        self.vocab = {}  # Tokenları saklamak için bir sözlük
        self.id_to_token = {}  # ID'den tokena eşleme yapmak için bir sözlük
        self.next_id = 0  # Bir sonraki token ID'si

    def add_token(self, token: str) -> int:
        """
        Yeni bir token ekler ve bir ID atar.

        :param token: Eklenmek istenen token.
        :return: Token'a atanmış ID.
        """
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]

    def tokenize(self, text: str) -> List[int]:
        """
        Metni tokenlara ayırır ve token ID'lerini döndürür.

        :param text: Tokenlara ayrılacak metin.
        :return: Token ID'lerinin listesi.
        """
        tokens = text.split()  # Metni boşluklara göre ayır
        token_ids = []
        for token in tokens:
            token_id = self.add_token(token)  # Token'ı ekle ve ID'sini al
            token_ids.append(token_id)
        return token_ids

    def detokenize(self, token_ids: List[int]) -> str:
        """
        Token ID'lerini metne dönüştürür.

        :param token_ids: Token ID'lerinin listesi.
        :return: Tokenlardan oluşturulmuş metin.
        """
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "")  # ID'ye karşılık gelen token'ı al
            tokens.append(token)
        return " ".join(tokens)  # Tokenları birleştir ve metni oluştur

# Örnek kullanım
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()

    # Metni tokenlara ayır
    text = "Merhaba dünya! Bu bir örnek metin."
    token_ids = tokenizer.tokenize(text)
    print(f"Token ID'leri: {token_ids}")

    # Token ID'lerini metne dönüştür
    decoded_text = tokenizer.detokenize(token_ids)
    print(f"Çözülen metin: {decoded_text}")""
    
    
=======
from typing import List

class SimpleTokenizer:
    """
    Basit bir tokenizer sınıfı. Bu sınıf, metni tokenlara ayırır ve tokenları tekrar metne dönüştürür.
    """

    def __init__(self):
        """
        Tokenizer'ı başlatır. Bu örnekte, boşluklara göre tokenlara ayırma işlemi yapılır.
        """
        self.vocab = {}  # Tokenları saklamak için bir sözlük
        self.id_to_token = {}  # ID'den tokena eşleme yapmak için bir sözlük
        self.next_id = 0  # Bir sonraki token ID'si

    def add_token(self, token: str) -> int:
        """
        Yeni bir token ekler ve bir ID atar.

        :param token: Eklenmek istenen token.
        :return: Token'a atanmış ID.
        """
        if token not in self.vocab:
            self.vocab[token] = self.next_id
            self.id_to_token[self.next_id] = token
            self.next_id += 1
        return self.vocab[token]

    def tokenize(self, text: str) -> List[int]:
        """
        Metni tokenlara ayırır ve token ID'lerini döndürür.

        :param text: Tokenlara ayrılacak metin.
        :return: Token ID'lerinin listesi.
        """
        tokens = text.split()  # Metni boşluklara göre ayır
        token_ids = []
        for token in tokens:
            token_id = self.add_token(token)  # Token'ı ekle ve ID'sini al
            token_ids.append(token_id)
        return token_ids

    def detokenize(self, token_ids: List[int]) -> str:
        """
        Token ID'lerini metne dönüştürür.

        :param token_ids: Token ID'lerinin listesi.
        :return: Tokenlardan oluşturulmuş metin.
        """
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id, "")  # ID'ye karşılık gelen token'ı al
            tokens.append(token)
        return " ".join(tokens)  # Tokenları birleştir ve metni oluştur

# Örnek kullanım
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()

    # Metni tokenlara ayır
    text = "Merhaba dünya! Bu bir örnek metin."
    token_ids = tokenizer.tokenize(text)
    print(f"Token ID'leri: {token_ids}")

    # Token ID'lerini metne dönüştür
    decoded_text = tokenizer.detokenize(token_ids)
    print(f"Çözülen metin: {decoded_text}")""
    
    
>>>>>>> 6e953b09772621b8cc37bb192b05fdf7daad2d9a
