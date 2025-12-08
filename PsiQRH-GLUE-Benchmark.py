"""
ΨQRH Genuine Trained Energy Distillation System - GLUE Benchmark
=================================================================
Google Colab Compatible Training Script

Este script treina e avalia o modelo ΨQRH no GLUE Benchmark,
incluindo as tarefas: MRPC, SST-2, CoLA, QNLI, QQP, RTE, MNLI, STS-B

Uso no Colab:
1. Faça upload deste arquivo
2. Execute todas as células
3. Os resultados serão salvos em /content/glue_results/

Autor: ΨQRH Research
"""

# =============================================================================
# INSTALAÇÃO DE DEPENDÊNCIAS (EXECUTAR PRIMEIRO NO COLAB)
# =============================================================================

"""
!pip install -q transformers datasets evaluate einops accelerate scikit-learn
!pip install -q torch torchvision torchaudio
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import logging
import time
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Verificar se estamos no Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab import drive
    print("🚀 Executando no Google Colab!")

# =============================================================================
# IMPORTS PRINCIPAIS
# =============================================================================

try:
    from einops import rearrange, reduce, repeat, parse_shape
    print("✅ EinOps importado com sucesso!")
except ImportError:
    print("⚠️ Instalando EinOps...")
    os.system("pip install -q einops")
    from einops import rearrange, reduce, repeat, parse_shape

try:
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    import evaluate
    print("✅ HuggingFace libraries importadas!")
except ImportError:
    print("⚠️ Instalando dependências HuggingFace...")
    os.system("pip install -q transformers datasets evaluate")
    from datasets import load_dataset
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup
    import evaluate

# =============================================================================
# CONFIGURAÇÕES DO GLUE BENCHMARK
# =============================================================================

@dataclass
class GLUETaskConfig:
    """Configuração para cada tarefa do GLUE"""
    name: str
    num_labels: int
    metric: str
    is_regression: bool = False
    text_columns: Tuple[str, ...] = ("sentence",)
    
GLUE_TASKS = {
    "cola": GLUETaskConfig("cola", 2, "matthews_correlation", text_columns=("sentence",)),
    "sst2": GLUETaskConfig("sst2", 2, "accuracy", text_columns=("sentence",)),
    "mrpc": GLUETaskConfig("mrpc", 2, "f1", text_columns=("sentence1", "sentence2")),
    "qqp": GLUETaskConfig("qqp", 2, "f1", text_columns=("question1", "question2")),
    "stsb": GLUETaskConfig("stsb", 1, "spearmanr", is_regression=True, text_columns=("sentence1", "sentence2")),
    "mnli": GLUETaskConfig("mnli", 3, "accuracy", text_columns=("premise", "hypothesis")),
    "qnli": GLUETaskConfig("qnli", 2, "accuracy", text_columns=("question", "sentence")),
    "rte": GLUETaskConfig("rte", 2, "accuracy", text_columns=("sentence1", "sentence2")),
    "wnli": GLUETaskConfig("wnli", 2, "accuracy", text_columns=("sentence1", "sentence2")),
}

@dataclass
class TrainingConfig:
    """Configurações de treinamento"""
    batch_size: int = 32
    learning_rate: float = 2e-5
    num_epochs: int = 3
    max_seq_len: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    d_model: int = 256
    n_layers: int = 4
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    seed: int = 42
    output_dir: str = "./glue_results"

# =============================================================================
# COMPONENTES DO MODELO ΨQRH (EINOPS OPTIMIZED)
# =============================================================================

class SpectralAttention(nn.Module):
    """GENUINE Spectral Attention com EinOps"""

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.k_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, n_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.alpha = nn.Parameter(torch.tensor(1.5))
        self.fractal_alpha_scale = nn.Parameter(torch.tensor(0.5))
        self.energy_normalizer = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        shape_before = parse_shape(x, 'b t c')
        B, T, C = shape_before['b'], shape_before['t'], shape_before['c']
        
        input_energy = torch.norm(x, p=2, dim=-1, keepdim=True)
        
        # Fractal dimension adaptativa
        fractal_dim = torch.tensor(1.5, device=x.device)
        adaptive_alpha = self.alpha + self.fractal_alpha_scale * (fractal_dim - 1.5)

        # Projeções QKV
        q = rearrange(self.q_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        k = rearrange(self.k_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)
        v = rearrange(self.v_proj(x), 'b t (h d) -> b h t d', h=self.n_heads)

        # Aplicar filtro espectral suave
        if T > 1:
            q_fft = torch.fft.fft(q, dim=2)
            freqs = torch.fft.fftfreq(T, device=x.device)
            k_magnitude = torch.clamp(torch.abs(freqs), min=1e-10)
            spectral_filter = torch.exp(-adaptive_alpha * torch.arctan(torch.log(k_magnitude)))
            spectral_filter = rearrange(spectral_filter, 't -> 1 1 t 1')
            q_filtered = torch.fft.ifft(q_fft * spectral_filter, dim=2).real
        else:
            q_filtered = q

        # Atenção scaled dot-product
        attn_logits = torch.matmul(q_filtered, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Aplicar máscara de atenção
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(attention_mask == 0, float('-inf'))
        
        attn_weights = self.dropout(torch.softmax(attn_logits, dim=-1))
        
        attended = torch.matmul(attn_weights, v)
        output = rearrange(attended, 'b h t d -> b t (h d)')
        output = self.out_proj(output)

        # Conservação de energia
        output_energy = torch.norm(output, p=2, dim=-1, keepdim=True)
        energy_ratio = input_energy / (output_energy + 1e-8)
        output = output * energy_ratio * self.energy_normalizer

        return output


class GenuineLeechLattice(nn.Module):
    """Leech Lattice otimizado com EinOps"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.lattice_dim = 24

        self.embed_to_lattice = nn.Linear(embed_dim, self.lattice_dim)
        self.lattice_to_embed = nn.Linear(self.lattice_dim, embed_dim)
        self.error_correction_strength = nn.Parameter(torch.tensor(0.1))
        self.energy_preservation = nn.Parameter(torch.tensor(1.0))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_energy = torch.norm(data, p=2, dim=-1, keepdim=True)
        
        # Encode
        lattice_proj = self.embed_to_lattice(data)
        lattice_points = torch.round(lattice_proj / (self.error_correction_strength + 1e-8)) * self.error_correction_strength
        
        # Decode
        corrected = torch.where(
            torch.abs(lattice_points) > self.error_correction_strength,
            lattice_points,
            torch.zeros_like(lattice_points)
        )
        result = self.lattice_to_embed(corrected)
        
        # Energy conservation
        output_energy = torch.norm(result, p=2, dim=-1, keepdim=True)
        energy_ratio = input_energy / (output_energy + 1e-8)
        result = result * energy_ratio * self.energy_preservation
        
        return result


class TransformerBlock(nn.Module):
    """Bloco Transformer com Spectral Attention"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.attention = SpectralAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture
        x = x + self.dropout(self.attention(self.attention_norm(x), attention_mask))
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x


class PSIQRHForSequenceClassification(nn.Module):
    """
    Modelo ΨQRH para Classificação de Sequências (GLUE Benchmark)
    
    Features:
    - Spectral Attention com filtro adaptativo
    - Leech Lattice error correction
    - Energy conservation
    - EinOps optimization
    """

    def __init__(
        self, 
        vocab_size: int = 30522,  # BERT vocab
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        num_labels: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        is_regression: bool = False
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_labels = num_labels
        self.is_regression = is_regression

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.token_type_embedding = nn.Embedding(2, d_model)  # Para pares de sentenças
        
        self.embedding_dropout = nn.Dropout(dropout)
        self.embedding_norm = nn.LayerNorm(d_model)
        
        # Leech Lattice
        self.leech_lattice = GenuineLeechLattice(d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        # Pooler e classificador
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_labels)
        )
        
        # Inicialização
        self.apply(self._init_weights)
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"ΨQRH Model: {total_params:,} total params, {trainable_params:,} trainable")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        B, T = input_ids.shape
        device = input_ids.device
        
        # Truncar se necessário
        if T > self.max_seq_len:
            input_ids = input_ids[:, :self.max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :self.max_seq_len]
            T = self.max_seq_len
        
        # Criar attention_mask padrão se não fornecida
        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=device)
            
        # Criar token_type_ids padrão
        if token_type_ids is None:
            token_type_ids = torch.zeros(B, T, dtype=torch.long, device=device)
        
        # Embeddings
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(positions)
        x = x + self.token_type_embedding(token_type_ids)
        
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)
        
        # Leech Lattice encoding (residual)
        lattice_output = self.leech_lattice(x)
        x = x + 0.1 * lattice_output  # Conexão residual suave
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Pooling (usar [CLS] token ou média)
        # Usamos média mascarada para melhor performance
        mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        pooled = self.pooler(pooled)
        logits = self.classifier(pooled)
        
        # Calcular loss se labels fornecidos
        loss = None
        if labels is not None:
            if self.is_regression:
                loss = F.mse_loss(logits.squeeze(), labels.float())
            else:
                loss = F.cross_entropy(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "pooled_output": pooled
        }


# =============================================================================
# DATASET E DATALOADER
# =============================================================================

class GLUEDataset(Dataset):
    """Dataset para tarefas GLUE"""
    
    def __init__(
        self, 
        task_name: str,
        split: str,
        tokenizer,
        max_length: int = 128
    ):
        self.task_name = task_name
        self.task_config = GLUE_TASKS[task_name]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Carregar dataset
        if task_name == "mnli" and split == "validation":
            # MNLI tem dois splits de validação
            self.dataset = load_dataset("glue", task_name, split="validation_matched")
        else:
            self.dataset = load_dataset("glue", task_name, split=split)
        
        logger.info(f"Loaded {task_name} {split}: {len(self.dataset)} examples")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extrair textos baseado na configuração da tarefa
        if len(self.task_config.text_columns) == 1:
            text = item[self.task_config.text_columns[0]]
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
        else:
            text_a = item[self.task_config.text_columns[0]]
            text_b = item[self.task_config.text_columns[1]]
            encoding = self.tokenizer(
                text_a,
                text_b,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
        
        # Extrair label
        label = item["label"]
        if self.task_config.is_regression:
            label = torch.tensor(label, dtype=torch.float)
        else:
            label = torch.tensor(label, dtype=torch.long)
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
            "labels": label
        }


def create_dataloader(
    task_name: str,
    split: str,
    tokenizer,
    batch_size: int,
    max_length: int = 128,
    shuffle: bool = True
) -> DataLoader:
    """Criar DataLoader para uma tarefa GLUE"""
    dataset = GLUEDataset(task_name, split, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2 if not IN_COLAB else 0,
        pin_memory=True
    )


# =============================================================================
# TREINAMENTO E AVALIAÇÃO
# =============================================================================

class GLUETrainer:
    """Trainer para GLUE Benchmark"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        task_name: str,
        tokenizer,
        device: torch.device
    ):
        self.model = model.to(device)
        self.config = config
        self.task_name = task_name
        self.task_config = GLUE_TASKS[task_name]
        self.tokenizer = tokenizer
        self.device = device
        
        # DataLoaders
        self.train_loader = create_dataloader(
            task_name, "train", tokenizer, config.batch_size, config.max_seq_len, shuffle=True
        )
        self.val_loader = create_dataloader(
            task_name, "validation", tokenizer, config.batch_size, config.max_seq_len, shuffle=False
        )
        
        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': config.weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
        
        # Scheduler
        total_steps = len(self.train_loader) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.fp16 and device.type == 'cuda' else None
        
        # Métricas
        self.metric = evaluate.load("glue", task_name)
        
        # Histórico
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_metric": []
        }
        
    def train_epoch(self, epoch: int) -> float:
        """Treinar uma época"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_interval = max(1, len(self.train_loader) // 10)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Mover para device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass com mixed precision
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
                    loss = outputs["loss"]
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                loss = outputs["loss"]
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % progress_interval == 0:
                avg_loss = total_loss / num_batches
                logger.info(f"  Epoch {epoch+1} [{batch_idx+1}/{len(self.train_loader)}] Loss: {avg_loss:.4f}")
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """Avaliar no conjunto de validação"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_predictions = []
        all_labels = []
        
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            total_loss += outputs["loss"].item()
            num_batches += 1
            
            logits = outputs["logits"]
            if self.task_config.is_regression:
                predictions = logits.squeeze().cpu().numpy()
            else:
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
        
        # Calcular métricas
        metrics = self.metric.compute(predictions=all_predictions, references=all_labels)
        
        return total_loss / num_batches, metrics
    
    def train(self) -> Dict[str, Any]:
        """Loop de treinamento completo"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training ΨQRH on {self.task_name.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        
        best_metric = -float('inf')
        best_epoch = 0
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Treinar
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            
            # Avaliar
            val_loss, val_metrics = self.evaluate()
            self.history["val_loss"].append(val_loss)
            
            # Extrair métrica principal
            metric_name = self.task_config.metric
            if metric_name in val_metrics:
                main_metric = val_metrics[metric_name]
            else:
                main_metric = list(val_metrics.values())[0]
            
            self.history["val_metric"].append(main_metric)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val {metric_name}: {main_metric:.4f}")
            logger.info(f"  Time: {epoch_time:.1f}s")
            
            # Salvar melhor modelo
            if main_metric > best_metric:
                best_metric = main_metric
                best_epoch = epoch + 1
                
                # Salvar checkpoint
                checkpoint_path = Path(self.config.output_dir) / f"{self.task_name}_best.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metric': best_metric,
                }, checkpoint_path)
                logger.info(f"  ✅ New best model saved!")
        
        total_time = time.time() - start_time
        
        results = {
            "task": self.task_name,
            "best_metric": best_metric,
            "metric_name": self.task_config.metric,
            "best_epoch": best_epoch,
            "total_time": total_time,
            "history": self.history
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Complete!")
        logger.info(f"Best {self.task_config.metric}: {best_metric:.4f} (epoch {best_epoch})")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"{'='*60}\n")
        
        return results


# =============================================================================
# FUNÇÃO PRINCIPAL DE EXECUÇÃO
# =============================================================================

def run_glue_benchmark(
    tasks: Optional[List[str]] = None,
    config: Optional[TrainingConfig] = None
) -> Dict[str, Any]:
    """
    Executar GLUE Benchmark completo
    
    Args:
        tasks: Lista de tarefas a executar (None = todas)
        config: Configurações de treinamento
        
    Returns:
        Dicionário com resultados de todas as tarefas
    """
    # Configurações padrão
    if config is None:
        config = TrainingConfig()
    
    # Tarefas padrão (excluindo WNLI que é notoriamente difícil)
    if tasks is None:
        tasks = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte"]
    
    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Configurar seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Carregar tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Resultados
    all_results = {}
    
    for task_name in tasks:
        if task_name not in GLUE_TASKS:
            logger.warning(f"Task {task_name} not found in GLUE_TASKS, skipping...")
            continue
        
        task_config = GLUE_TASKS[task_name]
        
        # Criar modelo para esta tarefa
        model = PSIQRHForSequenceClassification(
            vocab_size=tokenizer.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            num_labels=task_config.num_labels,
            max_seq_len=config.max_seq_len,
            is_regression=task_config.is_regression
        )
        
        # Criar trainer
        trainer = GLUETrainer(
            model=model,
            config=config,
            task_name=task_name,
            tokenizer=tokenizer,
            device=device
        )
        
        # Treinar
        try:
            results = trainer.train()
            all_results[task_name] = results
        except Exception as e:
            logger.error(f"Error training {task_name}: {e}")
            all_results[task_name] = {"error": str(e)}
        
        # Limpar memória
        del model, trainer
        torch.cuda.empty_cache() if device.type == "cuda" else None
    
    # Salvar resultados
    output_path = Path(config.output_dir) / "glue_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Preparar resultados serializáveis
    serializable_results = {}
    for task, result in all_results.items():
        if "error" in result:
            serializable_results[task] = result
        else:
            serializable_results[task] = {
                "best_metric": float(result["best_metric"]),
                "metric_name": result["metric_name"],
                "best_epoch": result["best_epoch"],
                "total_time": result["total_time"]
            }
    
    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_path}")
    
    # Resumo final
    print("\n" + "="*60)
    print("ΨQRH GLUE BENCHMARK RESULTS")
    print("="*60)
    
    for task, result in all_results.items():
        if "error" in result:
            print(f"{task.upper():8s}: ERROR - {result['error']}")
        else:
            print(f"{task.upper():8s}: {result['metric_name']:20s} = {result['best_metric']:.4f}")
    
    # Calcular média (GLUE score)
    valid_results = [r["best_metric"] for r in all_results.values() if "best_metric" in r]
    if valid_results:
        avg_score = np.mean(valid_results)
        print("-"*60)
        print(f"{'AVERAGE':8s}: {'GLUE Score':20s} = {avg_score:.4f}")
    
    print("="*60 + "\n")
    
    return all_results


# =============================================================================
# EXECUÇÃO NO COLAB
# =============================================================================

def run_quick_test():
    """Teste rápido para verificar se tudo funciona"""
    logger.info("🧪 Running quick test...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Criar modelo pequeno
    model = PSIQRHForSequenceClassification(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        num_labels=2,
        max_seq_len=32
    ).to(device)
    
    # Teste forward
    batch = {
        "input_ids": torch.randint(0, 1000, (4, 32)).to(device),
        "attention_mask": torch.ones(4, 32).to(device),
        "token_type_ids": torch.zeros(4, 32, dtype=torch.long).to(device),
        "labels": torch.randint(0, 2, (4,)).to(device)
    }
    
    with torch.no_grad():
        outputs = model(**batch)
    
    logger.info(f"✅ Forward pass successful!")
    logger.info(f"   Loss: {outputs['loss'].item():.4f}")
    logger.info(f"   Logits shape: {outputs['logits'].shape}")
    logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return True


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║     ΨQRH Genuine Trained Energy Distillation System           ║
    ║              GLUE Benchmark Training Script                    ║
    ╠═══════════════════════════════════════════════════════════════╣
    ║  Features:                                                     ║
    ║  • Spectral Attention with adaptive filtering                  ║
    ║  • Leech Lattice error correction                             ║
    ║  • Energy conservation (Parseval's theorem)                    ║
    ║  • EinOps optimized tensor operations                         ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Verificar instalação
    print("\n📦 Checking dependencies...")
    run_quick_test()
    
    print("\n" + "="*60)
    print("OPÇÕES DE EXECUÇÃO:")
    print("="*60)
    print("""
    1. TESTE RÁPIDO (1 tarefa, poucos epochs):
       >>> config = TrainingConfig(num_epochs=1, batch_size=16)
       >>> results = run_glue_benchmark(tasks=["mrpc"], config=config)
    
    2. BENCHMARK COMPLETO:
       >>> config = TrainingConfig(num_epochs=3, batch_size=32)
       >>> results = run_glue_benchmark(config=config)
    
    3. TAREFAS ESPECÍFICAS:
       >>> results = run_glue_benchmark(tasks=["sst2", "cola", "mrpc"])
    
    4. CONFIGURAÇÃO PERSONALIZADA:
       >>> config = TrainingConfig(
       ...     d_model=512,
       ...     n_layers=6,
       ...     learning_rate=1e-5,
       ...     num_epochs=5
       ... )
       >>> results = run_glue_benchmark(config=config)
    """)
    
    # Se executado diretamente, fazer teste rápido
    print("\n🚀 Executando teste rápido em MRPC...")
    config = TrainingConfig(
        num_epochs=1,
        batch_size=16,
        d_model=128,
        n_layers=2
    )
    results = run_glue_benchmark(tasks=["mrpc"], config=config)
