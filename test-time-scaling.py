import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from collections import Counter
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

class TestTimeScaling:
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate_single_response(self, prompt, max_new_tokens=100, temperature=0.8):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response

    def generate_multiple_responses(self, prompt, num_samples=7, max_new_tokens=100):
        responses = []
        for i in range(num_samples):
            temp = 0.7 + (i * 0.1)
            response = self.generate_single_response(prompt, max_new_tokens, temp)
            responses.append(response)
        return responses

    def best_of_n_sampling(self, prompt, num_samples=7, max_new_tokens=100):
        responses = self.generate_multiple_responses(prompt, num_samples, max_new_tokens)
        def score_response(response):
            words = response.split()
            unique_words = len(set(words))
            sentence_count = len(re.findall(r'[.!?]+', response))
            score = len(words) * 0.5 + unique_words * 1.5 + sentence_count * 2
            return score
        best_response = max(responses, key=score_response)
        return best_response

    def iterative_refinement(self, prompt, iterations=5, max_new_tokens=100):
        current_response = self.generate_single_response(prompt, max_new_tokens)
        for i in range(iterations):
            refinement_prompt = f"{prompt}\n\nÖnceki yanıt: {current_response}\n\nBu yanıtı daha iyi hale getir:"
            current_response = self.generate_single_response(refinement_prompt, max_new_tokens, temperature=0.6)
        return current_response

    def consensus_sampling(self, prompt, num_samples=7, max_new_tokens=100):
        responses = self.generate_multiple_responses(prompt, num_samples, max_new_tokens)
        all_words = []
        for response in responses:
            words = re.findall(r'\b\w+\b', response.lower())
            all_words.extend(words)
        word_counts = Counter(all_words)
        common_words = [word for word, count in word_counts.most_common(10)]
        def consensus_score(response):
            response_words = set(re.findall(r'\b\w+\b', response.lower()))
            return len(response_words.intersection(set(common_words)))
        best_consensus = max(responses, key=consensus_score)
        return best_consensus

def score_response(response):
    words = response.split()
    unique_words = len(set(words))
    sentence_count = len(re.findall(r'[.!?]+', response))
    score = len(words) * 0.5 + unique_words * 1.5 + sentence_count * 2
    return {
        "word_count": len(words),
        "unique_words": unique_words,
        "score": score
    }

def benchmark_strategies(model_name="Qwen/Qwen3-0.6B"):
    console = Console()
    tts = TestTimeScaling(model_name)
    prompt = "Yapay zeka gelecekte insanlığın hayatını nasıl etkileyecek?"
    results = []

    strategies = [
        ("Best-of-N", lambda: tts.best_of_n_sampling(prompt, num_samples=7, max_new_tokens=80)),
        ("İteratif İyileştirme", lambda: tts.iterative_refinement(prompt, iterations=5, max_new_tokens=80)),
        ("Konsensüs Örneklemesi", lambda: tts.consensus_sampling(prompt, num_samples=7, max_new_tokens=80)),
    ]

    for name, func in strategies:
        console.print(f"\n[bold cyan]⏳ {name} başlatıldı...[/bold cyan]")
        start = time.time()
        response = func()
        end = time.time()
        metrics = score_response(response)
        metrics.update({
            "strategy": name,
            "time_sec": round(end - start, 2),
            "response": response
        })
        results.append(metrics)
        console.print(Panel(f"[bold green]{name} Yanıtı:[/bold green]\n{response}", expand=False))

    df = pd.DataFrame(results)
    table = Table(title="Benchmark Sonuçları", show_header=True, header_style="bold magenta")
    table.add_column("Strateji", style="cyan")
    table.add_column("Süre (sn)", justify="right", style="green")
    table.add_column("Kelime Sayısı", justify="right", style="yellow")
    table.add_column("Benzersiz Kelimeler", justify="right", style="yellow")
    table.add_column("Skor", justify="right", style="red")

    for _, row in df.iterrows():
        table.add_row(
            row["strategy"],
            f"{row['time_sec']:.2f}",
            str(row["word_count"]),
            str(row["unique_words"]),
            f"{row['score']:.2f}"
        )

    console.print(table)
    return df

if __name__ == "__main__":
    benchmark_strategies()