import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SigmoidGateExamples(nn.Module):
    """Farklı sigmoid gate örnekleri"""
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Basit sigmoid gate
        self.gate_linear = nn.Linear(input_dim, hidden_dim)
        
        # LSTM gates
        self.lstm_gates = nn.Linear(input_dim + hidden_dim, 4 * hidden_dim)
        
        # GRU gates
        self.gru_gates = nn.Linear(input_dim + hidden_dim, 3 * hidden_dim)
        
        # Gated Linear Unit (GLU)
        self.glu_linear = nn.Linear(input_dim, 2 * hidden_dim)
        
        # Highway Network gate
        self.highway_gate = nn.Linear(input_dim, input_dim)
        self.highway_transform = nn.Linear(input_dim, input_dim)
    
    def simple_gate(self, x):
        """Basit sigmoid gate örneği"""
        # Gate değeri hesapla (0-1 arası)
        gate = torch.sigmoid(self.gate_linear(x))
        
        # Gate'i uygula: çıktı = gate * input
        output = gate * x[:, :self.hidden_dim]
        
        return output, gate
    
    def lstm_gates_example(self, x, h, c):
        """LSTM'deki 4 sigmoid gate"""
        # x: input, h: hidden state, c: cell state
        combined = torch.cat([x, h], dim=1)
        gates = self.lstm_gates(combined)
        
        # 4 gate'e ayır
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Sigmoid gates
        i = torch.sigmoid(i)  # Input gate: neyi hatırlayacağız
        f = torch.sigmoid(f)  # Forget gate: neyi unutacağız
        o = torch.sigmoid(o)  # Output gate: neyi çıktı olarak vereceğiz
        g = torch.tanh(g)     # Candidate values (gate değil)
        
        # Yeni cell state
        c_new = f * c + i * g
        
        # Yeni hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new, {'input': i, 'forget': f, 'output': o}
    
    def gru_gates_example(self, x, h):
        """GRU'daki sigmoid gates"""
        combined = torch.cat([x, h], dim=1)
        gates = self.gru_gates(combined)
        
        # 3 kısma ayır
        r, z, n = gates.chunk(3, dim=1)
        
        # Reset gate: önceki bilginin ne kadarını kullanacağız
        r = torch.sigmoid(r)
        
        # Update gate: yeni ve eski bilgiyi nasıl birleştireceğiz
        z = torch.sigmoid(z)
        
        # Yeni hidden state adayı
        n = torch.tanh(n)
        
        # Yeni hidden state
        h_new = (1 - z) * n + z * h
        
        return h_new, {'reset': r, 'update': z}
    
    def glu_example(self, x):
        """Gated Linear Unit (GLU)"""
        # Linear dönüşüm
        output = self.glu_linear(x)
        
        # İkiye böl
        a, b = output.chunk(2, dim=1)
        
        # GLU: a * sigmoid(b)
        return a * torch.sigmoid(b)
    
    def highway_gate_example(self, x):
        """Highway Network gate"""
        # Transform gate (T): ne kadar dönüşüm uygulayacağız
        T = torch.sigmoid(self.highway_gate(x))
        
        # Dönüştürülmüş veri
        H = torch.relu(self.highway_transform(x))
        
        # Highway formülü: y = T * H + (1 - T) * x
        # T=1: tamamen dönüşüm, T=0: girdiyi olduğu gibi geçir
        output = T * H + (1 - T) * x
        
        return output, T


class AttentionGate(nn.Module):
    """Attention mekanizmasında sigmoid gate kullanımı"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim, 1)
    
    def forward(self, query, keys, values):
        """
        query: [batch, hidden_dim]
        keys: [batch, seq_len, hidden_dim]
        values: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = keys.shape
        
        # Query'yi genişlet
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Attention hesapla
        combined = torch.cat([query_expanded, keys], dim=2)
        attention_hidden = torch.tanh(self.attention_linear(combined))
        
        # Sigmoid gate ile attention weights
        attention_scores = self.gate_linear(attention_hidden).squeeze(-1)
        attention_weights = torch.sigmoid(attention_scores)
        
        # Normalize (opsiyonel - soft attention için)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum
        attended = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return attended, attention_weights


class SigmoidGatingMechanism(nn.Module):
    """Genel amaçlı sigmoid gating mekanizması"""
    
    def __init__(self, input_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # Her expert için bir ağ
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, input_dim) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_experts)
        )
    
    def forward(self, x):
        # Expert çıktıları
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Gate değerleri (sigmoid)
        gates = torch.sigmoid(self.gate_network(x))
        gates = gates.unsqueeze(-1)
        
        # Ağırlıklı toplam
        output = (gates * expert_outputs).sum(dim=1)
        
        return output, gates.squeeze(-1)


def visualize_sigmoid_gate():
    """Sigmoid fonksiyonunu ve gate davranışını görselleştir"""
    x = np.linspace(-10, 10, 1000)
    sigmoid = 1 / (1 + np.exp(-x))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Sigmoid fonksiyonu
    axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2)
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Sigmoid Fonksiyonu')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('σ(x)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Gate çarpımı etkisi
    input_signal = np.sin(x)
    gated_signal = sigmoid * input_signal
    
    axes[0, 1].plot(x, input_signal, 'g-', label='Giriş sinyali', alpha=0.7)
    axes[0, 1].plot(x, sigmoid, 'r-', label='Gate değeri', alpha=0.7)
    axes[0, 1].plot(x, gated_signal, 'b-', label='Gate * Sinyal', linewidth=2)
    axes[0, 1].set_title('Gate Çarpımı Etkisi')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Farklı gate değerleri
    gate_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gate_values)))
    
    for gate, color in zip(gate_values, colors):
        axes[1, 0].plot(x, gate * np.sin(x), color=color, label=f'Gate={gate}')
    
    axes[1, 0].set_title('Farklı Gate Değerlerinin Etkisi')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Gate * sin(x)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. LSTM gate dinamikleri
    time_steps = 50
    forget_gate = np.random.beta(5, 2, time_steps)
    input_gate = np.random.beta(2, 5, time_steps)
    output_gate = np.random.beta(3, 3, time_steps)
    
    axes[1, 1].plot(forget_gate, 'r-', label='Forget gate', linewidth=2)
    axes[1, 1].plot(input_gate, 'g-', label='Input gate', linewidth=2)
    axes[1, 1].plot(output_gate, 'b-', label='Output gate', linewidth=2)
    axes[1, 1].set_title('LSTM Gate Dinamikleri (Örnek)')
    axes[1, 1].set_xlabel('Zaman adımı')
    axes[1, 1].set_ylabel('Gate değeri')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('sigmoid_gates_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_gate_effects():
    """Gate'lerin etkilerini göster"""
    print("=== Sigmoid Gate Etkileri Demonstrasyonu ===\n")
    
    # Örnek veri
    batch_size = 2
    input_dim = 4
    hidden_dim = 4
    
    x = torch.randn(batch_size, input_dim)
    h = torch.randn(batch_size, hidden_dim)
    c = torch.randn(batch_size, hidden_dim)
    
    # Model oluştur
    model = SigmoidGateExamples(input_dim, hidden_dim)
    
    # 1. Basit gate
    print("1. Basit Sigmoid Gate:")
    output, gate = model.simple_gate(x)
    print(f"   Giriş boyutu: {x.shape}")
    print(f"   Gate değerleri: {gate[0, :4].detach().numpy()}")
    print(f"   Çıktı: {output[0, :4].detach().numpy()}\n")
    
    # 2. LSTM gates
    print("2. LSTM Gates:")
    h_new, c_new, lstm_gates = model.lstm_gates_example(x, h, c)
    print(f"   Input gate ortalaması: {lstm_gates['input'].mean().item():.3f}")
    print(f"   Forget gate ortalaması: {lstm_gates['forget'].mean().item():.3f}")
    print(f"   Output gate ortalaması: {lstm_gates['output'].mean().item():.3f}\n")
    
    # 3. GRU gates
    print("3. GRU Gates:")
    h_new, gru_gates = model.gru_gates_example(x, h)
    print(f"   Reset gate ortalaması: {gru_gates['reset'].mean().item():.3f}")
    print(f"   Update gate ortalaması: {gru_gates['update'].mean().item():.3f}\n")
    
    # 4. Highway gate
    print("4. Highway Gate:")
    output, transform_gate = model.highway_gate_example(x)
    print(f"   Transform gate ortalaması: {transform_gate.mean().item():.3f}")
    print(f"   Bypass oranı: {(1 - transform_gate).mean().item():.3f}\n")
    
    # 5. Expert gating
    print("5. Expert Gating:")
    expert_model = SigmoidGatingMechanism(input_dim, num_experts=4)
    output, expert_gates = expert_model(x)
    print(f"   Expert gate değerleri: {expert_gates[0].detach().numpy()}")
    print(f"   En aktif expert: {expert_gates[0].argmax().item()}")


if __name__ == "__main__":
    # Görselleştirme
    visualize_sigmoid_gate()
    
    # Demonstrasyon
    demonstrate_gate_effects()
    
    print("\n✅ Sigmoid gates demonstrasyonu tamamlandı!")