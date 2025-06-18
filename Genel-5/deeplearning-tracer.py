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

# --- 1. Daha Derin bir PyTorch Modeli Tanımla ---
class DeepMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.1):
        """
        Derin bir çok katmanlı algılayıcı (MLP) modeli
        
        Args:
            input_dim: Giriş boyutu
            hidden_dims: Gizli katman boyutlarını içeren liste
            output_dim: Çıkış boyutu
            dropout_rate: Dropout oranı (varsayılan: 0.1)
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Giriş katmanı
        prev_dim = input_dim
        
        # Gizli katmanları oluştur
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
            
        # Çıkış katmanı
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Ağırlık başlatma
        self._init_weights()
        
        # Model bilgilerini göster
        self._print_model_info(input_dim, hidden_dims, output_dim, dropout_rate)
    
    def _init_weights(self):
        """Ağırlıkları Xavier/Glorot başlatma yöntemiyle başlat"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def _print_model_info(self, input_dim, hidden_dims, output_dim, dropout_rate):
        """Model yapısı hakkında bilgi göster"""
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info_table = Table(show_header=False, box=box.ROUNDED, show_edge=False)
        info_table.add_column("Özellik", style="cyan", no_wrap=True)
        info_table.add_column("Değer", style="green")
        
        info_table.add_row("Model Türü", "Derin Çok Katmanlı Algılayıcı (MLP)")
        info_table.add_row("Toplam Parametre", f"{total_params:,}")
        info_table.add_row("Giriş Boyutu", str(input_dim))
        info_table.add_row("Gizli Katmanlar", " → ".join(map(str, hidden_dims)))
        info_table.add_row("Çıkış Boyutu", str(output_dim))
        info_table.add_row("Dropout Oranı", str(dropout_rate))
        
        console.print(Panel(
            info_table,
            title="[bold green]Model Yapılandırması[/]",
            border_style="green",
            padding=(1, 2)
        ))
    
    def forward(self, x):
        """İleri yayılım"""
        # Gizli katmanlardan geçir
        for layer in self.layers:
            x = layer(x)
            
        # Çıkış katmanı
        x = self.output_layer(x)
        return x

# --- Kanca (Hook) için Global Depolama ve Durum Yönetimi ---
# Gerçek bir uygulamada bu durumu daha temiz yönetmek istersiniz (örneğin bir sınıf içinde).
hook_state = {
    "captured_activation": None,    # Yakalanan aktivasyonu saklamak için
    "is_intervention_mode": False,  # Müdahale modunda olup olmadığımızı belirtir
    "neuron_to_modify_idx": 0,    # Hangi nöronun aktivasyonuna müdahale edileceği
    "intervention_value": 0.0     # Müdahale edilecek yeni değer
}

# --- 2. Aktivasyonları Yakalamak ve Değiştirmek için bir Kanca (Hook) Uygula ---
def activation_hook_fn(module, input_args, output_tensor):
    """
    Bu bir PyTorch ileri (forward) kancasıdır.
    Eğer 'is_intervention_mode' False ise, katmanın çıkış aktivasyonunu yakalar.
    Eğer 'is_intervention_mode' True ise, belirtilen bir nöronun aktivasyonunu değiştirir.
    """
    global hook_state

    if not hook_state["is_intervention_mode"]:
        # Normal (yakalama) mod: Aktivasyonu sakla
        hook_state["captured_activation"] = output_tensor.clone().detach()
        # print(f"Kanca (Yakalama): {module} çıkışı yakalandı: {hook_state['captured_activation']}")
        return None # Çıkışı değiştirme, orijinali kullanılsın
    else:
        # Müdahale modu: Aktivasyonu değiştir
        modified_output = output_tensor.clone() # Değişiklik yapmadan önce klonla!

        # Örneğin, ilk nöronun (batch_size=1 varsayımıyla) aktivasyonunu değiştir
        # output_tensor'un şekli [batch_size, num_features] beklenir
        if modified_output.ndim == 2 and modified_output.shape[0] == 1: # [1, hidden_dim] gibi
            neuron_idx = hook_state["neuron_to_modify_idx"]
            if 0 <= neuron_idx < modified_output.shape[1]:
                # print(f"Kanca (Müdahale): {module} Nöron {neuron_idx} orijinal değeri: {modified_output[0, neuron_idx]}")
                modified_output[0, neuron_idx] = hook_state["intervention_value"]
                # print(f"Kanca (Müdahale): {module} Nöron {neuron_idx} yeni değeri: {modified_output[0, neuron_idx]}")
                hook_state["captured_activation"] = modified_output.clone().detach() # Değiştirilmiş aktivasyonu da sakla
                return modified_output # Değiştirilmiş aktivasyonu döndür
            else:
                print(f"Uyarı: Nöron indeksi {neuron_idx} sınırlar dışında.")
                return None # Bir sorun varsa orijinali döndür
        else:
            print(f"Uyarı: Kanca, [1, num_features] şeklinde aktivasyon bekliyordu, gelen: {modified_output.shape}")
            return None # Bir sorun varsa orijinali döndür

# --- Model ve Veri Kurulumu ---
input_dim = 10
hidden_dims = [64, 32, 16]  # Daha derin mimari
output_dim = 2
dropout_rate = 0.1

# Modeli oluştur
model = DeepMLP(input_dim, hidden_dims, output_dim, dropout_rate)

# Kullanılabilir cihazı belirle (GPU varsa onu kullan)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Model özetini göster
console.print(f"\n[bold]Model {device} cihazına yüklendi.[/]")
console.print(f"Eğitilebilir parametre sayısı: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Tüm ReLU katmanlarına kancaları kaydet
hook_handles = []
for i, layer in enumerate(model.layers):
    if isinstance(layer, nn.ReLU):
        handle = layer.register_forward_hook(activation_hook_fn)
        hook_handles.append(handle)
        print(f"ReLU katmanına kanca eklendi: {i}")

if not hook_handles:
    raise ValueError("Modelde hiç ReLU katmanı bulunamadı!")

# Rastgele bir girdi verisi oluştur (basitlik için batch_size=1)
dummy_input = torch.randn(1, input_dim).to(device)

# Girdi verisi hakkında bilgi
input_info = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
input_info.add_column("Özellik", style="cyan")
input_info.add_column("Değer", style="green")
input_info.add_row("Girdi Boyutu", f"{tuple(dummy_input.shape)}")
input_info.add_row("Min Değer", f"{dummy_input.min().item():.4f}")
input_info.add_row("Maksimum Değer", f"{dummy_input.max().item():.4f}")
input_info.add_row("Ortalama", f"{dummy_input.mean().item():.4f}")
input_info.add_row("Standart Sapma", f"{dummy_input.std().item():.4f}")

console.print(Panel(
    input_info,
    title="[bold blue]Girdi Verisi İstatistikleri[/]",
    border_style="blue",
    padding=(1, 2)
))

# İlk 5 özelliği göster
input_sample = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
input_sample.add_column("Özellik İndeksi", style="cyan")
input_sample.add_column("Değer", style="green")

for i, val in enumerate(dummy_input.squeeze().cpu().numpy()[:5]):
    input_sample.add_row(f"{i}", f"{val:.6f}")

console.print(Panel(
    input_info,
    title="[bold blue]Girdi Verisi (İlk 5 Özellik)[/]",
    border_style="blue",
    padding=(1, 2)
))
print_section("🔧 Model ve Veri Kurulumu")
console.print(f"[bold]Model Yapısı:[/] [cyan]Input: {input_dim}[/] → [green]Hidden: {hidden_dim}[/] → [yellow]Output: {output_dim}[/]")
console.print(f"[bold]Girdi Verisi:[/] {dummy_input.squeeze().tolist()[:5]}... [dim](ilk 5 özellik gösteriliyor)[/dim]\n")

# --- 3. "Temiz Çalıştırma": Temel aktivasyonları ve çıktıyı al ---
print_section("🔍 Temiz Çalıştırma (Müdahalesiz)")

hook_state["is_intervention_mode"] = False
with torch.no_grad():
    original_output = model(dummy_input)
    clean_hidden_activation = hook_state["captured_activation"]

# Gizli katman aktivasyonlarını gösteren tablo
table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
table.add_column("Nöron", style="dim", width=12)
table.add_column("Aktivasyon Değeri", justify="right")

for i, val in enumerate(clean_hidden_activation.squeeze().tolist()):
    table.add_row(f"Nöron {i}", f"{val:.4f}")

console.print(Panel.fit(
    table,
    title="[bold]Gizli Katman Aktivasyonları (ReLU Sonrası)",
    border_style="green",
    padding=(1, 2)
))

console.print(f"\n[bold]Model Çıktısı:[/] {original_output.squeeze().tolist()}")
console.rule(style="dim")

# --- 4. "Müdahale Çalıştırması": Bir aktivasyonu değiştir ve etkiyi gör ---
print_section("🔧 Müdahale Çalıştırması")

# Müdahale ayarları
neuron_idx = 0
new_value = 10.0

hook_state["is_intervention_mode"] = True
hook_state["neuron_to_modify_idx"] = neuron_idx
hook_state["intervention_value"] = new_value

with torch.no_grad():
    intervened_output = model(dummy_input)
    intervened_hidden_activation = hook_state["captured_activation"]

# Müdahale özeti
console.print(f"[bold]Müdahale Detayları:[/]")
console.print(f"  • [yellow]Hedef Nöron:[/] [bold]{neuron_idx}[/]")
console.print(f"  • [yellow]Yeni Değer:[/] [bold]{new_value}[/]")

# Müdahale edilmiş aktivasyonlar tablosu
modified_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
modified_table.add_column("Nöron", style="dim", width=12)
modified_table.add_column("Önceki Değer", justify="right")
modified_table.add_column("Yeni Değer", justify="right")
modified_table.add_column("Durum", justify="center")

for i, (orig, new) in enumerate(zip(
    clean_hidden_activation.squeeze().tolist(),
    intervened_hidden_activation.squeeze().tolist()
)):
    modified = i == neuron_idx
    status = "[bold red]✗ Değiştirildi" if modified else "[green]✓ Aynı"
    orig_val = f"[strike dim]{orig:.4f}[/]" if modified else f"{orig:.4f}"
    new_val = f"[bold red]{new:.4f}" if modified else f"{new:.4f}"
    
    modified_table.add_row(
        f"Nöron {i}",
        orig_val,
        new_val,
        status
    )

console.print(Panel.fit(
    modified_table,
    title="[bold]Gizli Katman Karşılaştırması",
    border_style="yellow",
    padding=(1, 2)
))

console.print(f"\n[bold]Yeni Model Çıktısı:[/] {intervened_output.squeeze().tolist()}")
console.rule(style="dim")

# --- 5. Karşılaştır ---
print_section("📊 Sonuçların Karşılaştırılması")

# Çıktı karşılaştırma tablosu
output_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
output_table.add_column("Çıktı Nöronu", style="dim", width=12)
output_table.add_column("Orijinal Değer", justify="right")
output_table.add_column("Yeni Değer", justify="right")
output_table.add_column("Fark", justify="right")

orig_outputs = original_output.squeeze().tolist()
new_outputs = intervened_output.squeeze().tolist()
diffs = torch.abs(original_output - intervened_output).squeeze().tolist()

for i, (orig, new, diff) in enumerate(zip(orig_outputs, new_outputs, diffs)):
    diff_style = "[red]" if diff > 0.1 else "[green]"
    output_table.add_row(
        f"Çıktı {i}",
        f"{orig:.6f}",
        f"{new:.6f}",
        f"{diff_style}{diff:.6f}"
    )

console.print(Panel.fit(
    output_table,
    title="[bold]Çıktı Karşılaştırması",
    border_style="blue",
    padding=(1, 2)
))

# Özet istatistikler
console.print("\n[bold]📈 Özet İstatistikler:[/]")
console.print(f"  • [yellow]Toplam Mutlak Fark:[/] {torch.sum(torch.abs(original_output - intervened_output)):.6f}")
console.print(f"  • [yellow]Maksimum Fark:[/] {torch.max(torch.abs(original_output - intervened_output)):.6f}")
console.print(f"  • [yellow]Ortalama Mutlak Fark:[/] {torch.mean(torch.abs(original_output - intervened_output)):.6f}")

# Kanca temizliği hakkında bilgi
console.print("\n[dim]Not: Kanca başarıyla kaldırıldı.[/dim]")


# Kancayı işiniz bittiğinde kaldırmayı unutmayın,
# özellikle bir notebook'ta hücreleri tekrar tekrar çalıştırıyorsanız.
hook_handle.remove()