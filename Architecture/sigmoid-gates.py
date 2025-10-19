import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class SigmoidGateExamples(nn.Module):
    """Examples of different sigmoid gate mechanisms."""
    
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
        """Basic sigmoid gate example."""
        # Compute gate value between 0 and 1
        gate = torch.sigmoid(self.gate_linear(x))
        
        # Apply the gate: output = gate * input
        output = gate * x[:, :self.hidden_dim]
        
        return output, gate
    
    def lstm_gates_example(self, x, h, c):
        """The four sigmoid gates used in an LSTM."""
        # x: input, h: hidden state, c: cell state
        combined = torch.cat([x, h], dim=1)
        gates = self.lstm_gates(combined)
        
        # Split into the four gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Sigmoid gates
        i = torch.sigmoid(i)  # Input gate: what we keep in memory
        f = torch.sigmoid(f)  # Forget gate: what we discard
        o = torch.sigmoid(o)  # Output gate: what we expose as output
        g = torch.tanh(g)     # Candidate values (not a sigmoid gate)
        
        # Updated cell state
        c_new = f * c + i * g
        
        # Updated hidden state
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new, {'input': i, 'forget': f, 'output': o}
    
    def gru_gates_example(self, x, h):
        """Sigmoid gates inside a GRU."""
        combined = torch.cat([x, h], dim=1)
        gates = self.gru_gates(combined)
        
        # Split into three sections
        r, z, n = gates.chunk(3, dim=1)
        
        # Reset gate: how much of the previous state to use
        r = torch.sigmoid(r)
        
        # Update gate: how to mix new and old information
        z = torch.sigmoid(z)
        
        # Candidate hidden state
        n = torch.tanh(n)
        
        # Updated hidden state
        h_new = (1 - z) * n + z * h
        
        return h_new, {'reset': r, 'update': z}
    
    def glu_example(self, x):
        """Gated Linear Unit (GLU)."""
        # Linear projection
        output = self.glu_linear(x)
        
        # Split in two
        a, b = output.chunk(2, dim=1)
        
        # GLU: a * sigmoid(b)
        return a * torch.sigmoid(b)
    
    def highway_gate_example(self, x):
        """Highway Network gate."""
        # Transform gate (T): how much of the transformed signal to use
        T = torch.sigmoid(self.highway_gate(x))
        
        # Transformed data
        H = torch.relu(self.highway_transform(x))
        
        # Highway formula: y = T * H + (1 - T) * x
        # T=1: full transform, T=0: passthrough the input
        output = T * H + (1 - T) * x
        
        return output, T


class AttentionGate(nn.Module):
    """Attention mechanism augmented with sigmoid gating."""
    
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
        
        # Broadcast the query across the sequence dimension
        query_expanded = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute the attention projection
        combined = torch.cat([query_expanded, keys], dim=2)
        attention_hidden = torch.tanh(self.attention_linear(combined))
        
        # Attention weights from the sigmoid gate
        attention_scores = self.gate_linear(attention_hidden).squeeze(-1)
        attention_weights = torch.sigmoid(attention_scores)
        
        # Normalize (optional, for soft attention)
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        # Weighted sum
        attended = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        
        return attended, attention_weights


class SigmoidGatingMechanism(nn.Module):
    """General-purpose sigmoid gating mechanism."""
    
    def __init__(self, input_dim, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # A small network for each expert
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
        # Collect expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Sigmoid gate values
        gates = torch.sigmoid(self.gate_network(x))
        gates = gates.unsqueeze(-1)
        
        # Weighted sum of expert outputs
        output = (gates * expert_outputs).sum(dim=1)
        
        return output, gates.squeeze(-1)


def visualize_sigmoid_gate():
    """Visualize the sigmoid function and gate behaviour."""
    x = np.linspace(-10, 10, 1000)
    sigmoid = 1 / (1 + np.exp(-x))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Sigmoid function
    axes[0, 0].plot(x, sigmoid, 'b-', linewidth=2)
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Sigmoid Function')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('σ(x)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Effect of multiplying by the gate
    input_signal = np.sin(x)
    gated_signal = sigmoid * input_signal
    
    axes[0, 1].plot(x, input_signal, 'g-', label='Input signal', alpha=0.7)
    axes[0, 1].plot(x, sigmoid, 'r-', label='Gate value', alpha=0.7)
    axes[0, 1].plot(x, gated_signal, 'b-', label='Gate * Signal', linewidth=2)
    axes[0, 1].set_title('Effect of Gate Multiplication')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Different gate values
    gate_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    colors = plt.cm.viridis(np.linspace(0, 1, len(gate_values)))
    
    for gate, color in zip(gate_values, colors):
        axes[1, 0].plot(x, gate * np.sin(x), color=color, label=f'Gate={gate}')

    axes[1, 0].set_title('Effect of Different Gate Values')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Gate * sin(x)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. LSTM gate dynamics
    time_steps = 50
    forget_gate = np.random.beta(5, 2, time_steps)
    input_gate = np.random.beta(2, 5, time_steps)
    output_gate = np.random.beta(3, 3, time_steps)
    
    axes[1, 1].plot(forget_gate, 'r-', label='Forget gate', linewidth=2)
    axes[1, 1].plot(input_gate, 'g-', label='Input gate', linewidth=2)
    axes[1, 1].plot(output_gate, 'b-', label='Output gate', linewidth=2)
    axes[1, 1].set_title('LSTM Gate Dynamics (Sample)')
    axes[1, 1].set_xlabel('Time step')
    axes[1, 1].set_ylabel('Gate value')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('sigmoid_gates_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_gate_effects():
    """Showcase how different gates behave."""
    print("=== Sigmoid Gate Effects Demonstration ===\n")
    
    # Sample data
    batch_size = 2
    input_dim = 4
    hidden_dim = 4
    
    x = torch.randn(batch_size, input_dim)
    h = torch.randn(batch_size, hidden_dim)
    c = torch.randn(batch_size, hidden_dim)
    
    # Build model
    model = SigmoidGateExamples(input_dim, hidden_dim)
    
    # 1. Simple gate
    print("1. Simple Sigmoid Gate:")
    output, gate = model.simple_gate(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Gate values: {gate[0, :4].detach().numpy()}")
    print(f"   Output: {output[0, :4].detach().numpy()}\n")
    
    # 2. LSTM gates
    print("2. LSTM Gates:")
    h_new, c_new, lstm_gates = model.lstm_gates_example(x, h, c)
    print(f"   Input gate mean: {lstm_gates['input'].mean().item():.3f}")
    print(f"   Forget gate mean: {lstm_gates['forget'].mean().item():.3f}")
    print(f"   Output gate mean: {lstm_gates['output'].mean().item():.3f}\n")
    
    # 3. GRU gates
    print("3. GRU Gates:")
    h_new, gru_gates = model.gru_gates_example(x, h)
    print(f"   Reset gate mean: {gru_gates['reset'].mean().item():.3f}")
    print(f"   Update gate mean: {gru_gates['update'].mean().item():.3f}\n")
    
    # 4. Highway gate
    print("4. Highway Gate:")
    output, transform_gate = model.highway_gate_example(x)
    print(f"   Transform gate mean: {transform_gate.mean().item():.3f}")
    print(f"   Bypass rate: {(1 - transform_gate).mean().item():.3f}\n")
    
    # 5. Expert gating
    print("5. Expert Gating:")
    expert_model = SigmoidGatingMechanism(input_dim, num_experts=4)
    output, expert_gates = expert_model(x)
    print(f"   Expert gate values: {expert_gates[0].detach().numpy()}")
    print(f"   Most active expert: {expert_gates[0].argmax().item()}")


if __name__ == "__main__":
    # Visualization
    visualize_sigmoid_gate()

    # Demonstration
    demonstrate_gate_effects()

    print("\n✅ Sigmoid gates demonstration completed!")