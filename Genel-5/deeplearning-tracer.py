import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint
from rich import box

# Initialize rich console
console = Console()

def print_section(title, color="cyan"):
    """Print a section header with rich formatting"""
    console.rule(f"[bold {color}]{title}", style=color)

# --- 1. Define a Deeper PyTorch Model ---
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        """
        A deep multi-layer perceptron (MLP) model

        Args:
            input_dim: Input dimension
            hidden_dims: List containing the hidden layer sizes
            output_dim: Output dimension
            dropout_rate: Dropout rate (default: 0.1)
        """
        super().__init__()
        self.layers = nn.ModuleList()

        # Input layer
        prev_dim = input_dim

        # Create hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, output_dim)

        # Weight initialization
        self._init_weights()

        # Display model information
        self._print_model_info(input_dim, hidden_dims, output_dim, dropout_rate)

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def _print_model_info(self, input_dim, hidden_dims, output_dim, dropout_rate):
        """Display information about the model architecture"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        info_table = Table(show_header=False, box=box.ROUNDED, show_edge=False)
        info_table.add_column("Feature", style="cyan", no_wrap=True)
        info_table.add_column("Value", style="green")

        info_table.add_row("Model Type", "Deep Multi-Layer Perceptron (MLP)")
        info_table.add_row("Total Parameters", f"{total_params:,}")
        info_table.add_row("Input Dimension", str(input_dim))
        info_table.add_row("Hidden Layers", " → ".join(map(str, hidden_dims)))
        info_table.add_row("Output Dimension", str(output_dim))
        info_table.add_row("Dropout Rate", str(dropout_rate))

        console.print(Panel(
            info_table,
            title="[bold green]Model Configuration[/]",
            border_style="green",
            padding=(1, 2)
        ))

    def forward(self, x):
        """Forward pass"""
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)

        # Output layer
        x = self.output_layer(x)
        return x

# --- Global Storage and State Management for Hooks ---
# In a real application you would likely manage this state in a cleaner way (e.g., inside a class).
hook_state = {
    "captured_activation": None,    # Stores the captured activation
    "is_intervention_mode": False,  # Indicates whether we are in intervention mode
    "neuron_to_modify_idx": 0,    # Which neuron's activation to intervene on
    "intervention_value": 0.0     # The value to inject during the intervention
}

# --- 2. Apply a Hook to Capture and Modify Activations ---
def activation_hook_fn(module, input_args, output_tensor):
    """
    This is a PyTorch forward hook.
    If 'is_intervention_mode' is False, it captures the layer's output activation.
    If 'is_intervention_mode' is True, it modifies the activation of a specified neuron.
    """
    global hook_state

    if not hook_state["is_intervention_mode"]:
        # Normal (capture) mode: store the activation
        hook_state["captured_activation"] = output_tensor.clone().detach()
        return None  # Do not modify the output
    else:
        # Intervention mode: modify the activation
        modified_output = output_tensor.clone()  # Clone before modifying

        # For example, change the activation of the first neuron (assuming batch_size=1)
        # The output tensor is expected to have shape [batch_size, num_features]
        if modified_output.ndim == 2 and modified_output.shape[0] == 1:  # e.g. [1, hidden_dim]
            neuron_idx = hook_state["neuron_to_modify_idx"]
            if 0 <= neuron_idx < modified_output.shape[1]:
                modified_output[0, neuron_idx] = hook_state["intervention_value"]
                hook_state["captured_activation"] = modified_output.clone().detach()  # Store the modified activation
                return modified_output  # Return the modified activation
            else:
                print(f"Warning: Neuron index {neuron_idx} is out of bounds.")
                return None  # Fall back to the original activation on error
        else:
            print(f"Warning: The hook expected an activation shaped like [1, num_features], received: {modified_output.shape}")
            return None  # Fall back to the original activation on error

# --- Model and Data Setup ---
input_dim = 10
hidden_dims = [64, 32, 16]  # Deeper architecture
output_dim = 2
dropout_rate = 0.1

# Create the model
model = DeepMLP(input_dim, hidden_dims, output_dim, dropout_rate)

# Detect the available device (use GPU if present)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Display a model summary
console.print(f"\n[bold]Model loaded to {device}.[/]")
console.print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Register hooks on all ReLU layers
hook_handles = []
for i, layer in enumerate(model.layers):
    if isinstance(layer, nn.ReLU):
        handle = layer.register_forward_hook(activation_hook_fn)
        hook_handles.append(handle)
        print(f"Hook added to ReLU layer: {i}")

if not hook_handles:
    raise ValueError("No ReLU layers found in the model!")

# Create random input data (batch_size=1 for simplicity)
dummy_input = torch.randn(1, input_dim).to(device)

# Display information about the input data
input_info = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
input_info.add_column("Feature", style="cyan")
input_info.add_column("Value", style="green")
input_info.add_row("Input Shape", f"{tuple(dummy_input.shape)}")
input_info.add_row("Minimum", f"{dummy_input.min().item():.4f}")
input_info.add_row("Maximum", f"{dummy_input.max().item():.4f}")
input_info.add_row("Mean", f"{dummy_input.mean().item():.4f}")
input_info.add_row("Standard Deviation", f"{dummy_input.std().item():.4f}")

console.print(Panel(
    input_info,
    title="[bold blue]Input Data Statistics[/]",
    border_style="blue",
    padding=(1, 2)
))

# Show the first five features
input_sample = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
input_sample.add_column("Feature Index", style="cyan")
input_sample.add_column("Value", style="green")

for i, val in enumerate(dummy_input.squeeze().cpu().numpy()[:5]):
    input_sample.add_row(f"{i}", f"{val:.6f}")

console.print(Panel(
    input_sample,
    title="[bold blue]Input Data (First 5 Features)[/]",
    border_style="blue",
    padding=(1, 2)
))
print_section("🔧 Model and Data Setup")
console.print(f"[bold]Model Architecture:[/] [cyan]Input: {input_dim}[/] → [green]Hidden: {hidden_dims}[/] → [yellow]Output: {output_dim}[/]")
console.print(f"[bold]Input Sample:[/] {dummy_input.squeeze().tolist()[:5]}... [dim](showing the first 5 features)[/dim]\n")

# --- 3. "Clean Run": Capture the baseline activations and output ---
print_section("🔍 Clean Run (No Intervention)")

hook_state["is_intervention_mode"] = False
with torch.no_grad():
    original_output = model(dummy_input)
    clean_hidden_activation = hook_state["captured_activation"]

# Table showing hidden layer activations
table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
table.add_column("Neuron", style="dim", width=12)
table.add_column("Activation Value", justify="right")

for i, val in enumerate(clean_hidden_activation.squeeze().tolist()):
    table.add_row(f"Neuron {i}", f"{val:.4f}")

console.print(Panel.fit(
    table,
    title="[bold]Hidden Layer Activations (Post-ReLU)",
    border_style="green",
    padding=(1, 2)
))

console.print(f"\n[bold]Model Output:[/] {original_output.squeeze().tolist()}")
console.rule(style="dim")

# --- 4. "Intervention Run": Change an activation and observe the effect ---
print_section("🔧 Intervention Run")

# Intervention settings
neuron_idx = 0
new_value = 10.0

hook_state["is_intervention_mode"] = True
hook_state["neuron_to_modify_idx"] = neuron_idx
hook_state["intervention_value"] = new_value

with torch.no_grad():
    intervened_output = model(dummy_input)
    intervened_hidden_activation = hook_state["captured_activation"]

# Intervention summary
console.print(f"[bold]Intervention Details:[/]")
console.print(f"  • [yellow]Target Neuron:[/] [bold]{neuron_idx}[/]")
console.print(f"  • [yellow]New Value:[/] [bold]{new_value}[/]")

# Table with modified activations
modified_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
modified_table.add_column("Neuron", style="dim", width=12)
modified_table.add_column("Previous Value", justify="right")
modified_table.add_column("New Value", justify="right")
modified_table.add_column("Status", justify="center")

for i, (orig, new) in enumerate(zip(
    clean_hidden_activation.squeeze().tolist(),
    intervened_hidden_activation.squeeze().tolist()
)):
    modified = i == neuron_idx
    status = "[bold red]✗ Modified" if modified else "[green]✓ Unchanged"
    orig_val = f"[strike dim]{orig:.4f}[/]" if modified else f"{orig:.4f}"
    new_val = f"[bold red]{new:.4f}" if modified else f"{new:.4f}"

    modified_table.add_row(
        f"Neuron {i}",
        orig_val,
        new_val,
        status
    )

console.print(Panel.fit(
    modified_table,
    title="[bold]Hidden Layer Comparison",
    border_style="yellow",
    padding=(1, 2)
))

console.print(f"\n[bold]New Model Output:[/] {intervened_output.squeeze().tolist()}")
console.rule(style="dim")

# --- 5. Compare Results ---
print_section("📊 Comparing Results")

# Output comparison table
output_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
output_table.add_column("Output Neuron", style="dim", width=12)
output_table.add_column("Original Value", justify="right")
output_table.add_column("New Value", justify="right")
output_table.add_column("Difference", justify="right")

orig_outputs = original_output.squeeze().tolist()
new_outputs = intervened_output.squeeze().tolist()
diffs = torch.abs(original_output - intervened_output).squeeze().tolist()

for i, (orig, new, diff) in enumerate(zip(orig_outputs, new_outputs, diffs)):
    diff_style = "[red]" if diff > 0.1 else "[green]"
    output_table.add_row(
        f"Output {i}",
        f"{orig:.6f}",
        f"{new:.6f}",
        f"{diff_style}{diff:.6f}"
    )

console.print(Panel.fit(
    output_table,
    title="[bold]Output Comparison",
    border_style="blue",
    padding=(1, 2)
))

# Summary statistics
console.print("\n[bold]📈 Summary Statistics:[/]")
console.print(f"  • [yellow]Total Absolute Difference:[/] {torch.sum(torch.abs(original_output - intervened_output)):.6f}")
console.print(f"  • [yellow]Maximum Difference:[/] {torch.max(torch.abs(original_output - intervened_output)):.6f}")
console.print(f"  • [yellow]Mean Absolute Difference:[/] {torch.mean(torch.abs(original_output - intervened_output)):.6f}")

# Information about hook cleanup
console.print("\n[dim]Note: Hooks have been removed successfully.[/dim]")


# Always remove hooks when you're done, especially if you are repeatedly
# executing cells in a notebook environment.
for handle in hook_handles:
    handle.remove()
