"""
Passo 2 do Tech Challenge Fase 3 — Normalização + Split
========================================================

Este script:
1. Lê o dataset anonimizado: data/processed/medquad_anonimizado.jsonl
2. Aplica normalização adicional:
   - Encoding UTF-8 + NFC (caracteres compostos)
   - Whitespace collapse (remove indentação exagerada)
   - Truncamento de outputs > MAX_OUTPUT_CHARS (2500)
   - Remoção de caracteres de controle
3. Divide em train/val/test (90/5/5) com seed=42
4. Salva em:
   - data/processed/train.jsonl
   - data/processed/val.jsonl
   - data/processed/test.jsonl
5. Gera relatório em: data/processed/relatorio_split.txt

Uso:
    python src/data/02_normalizar_e_split.py

Pré-requisito:
    Ter rodado 01_anonimizar.py primeiro.

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
Data:  2026-08-31

Parâmetros (ver README.md para detalhes completos):
    MAX_OUTPUT_CHARS = 2500  # ~512-600 tokens, cabe em max_seq_length=4096
    SPLIT_RATIOS = (0.90, 0.05, 0.05)  # 90% train, 5% val, 5% test
    RANDOM_SEED = 42  # reprodutibilidade
"""

import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO DE PATHS (aceita env vars)
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent.parent))

INPUT_FILE = Path(os.environ.get(
    "INPUT_FILE",
    PROJECT_ROOT / "data" / "processed" / "medquad_anonimizado.jsonl"
))
OUTPUT_DIR = Path(os.environ.get(
    "OUTPUT_DIR",
    PROJECT_ROOT / "data" / "processed"
))
REPORT_FILE = OUTPUT_DIR / "relatorio_split.txt"

# ============================================================
# PARÂMETROS DE NORMALIZAÇÃO (todos explicados no README)
# ============================================================
MAX_OUTPUT_CHARS = 2500
"""
Tamanho máximo do output em caracteres.
Justificativa: max_seq_length=4096 tokens precisa acomodar:
- instruction (~200 chars ~ 50 tokens)
- input (~100 chars ~ 25 tokens)
- tokens de template Alpaca (~80 tokens)
- output (resto, até ~15.700 chars / ~3.700 tokens)
Usar 2500 chars (~500-600 tokens) deixa margem generosa.
Outputs médicos com info crítica estão nos primeiros 2500 chars.
"""

SPLIT_TRAIN = 0.90
SPLIT_VAL = 0.05
SPLIT_TEST = 0.05
"""
Proporções de divisão. 90/5/5 é o padrão para datasets de 10k-100k:
- train (90%): ajuste de pesos
- val (5%): early stopping e ajuste de hiperparâmetros
- test (5%): avaliação final honesta (modelo nunca viu durante treino)
"""

RANDOM_SEED = 42
"""
Seed do gerador aleatório. Garante reprodutibilidade:
rodar o script 2x produz exatamente os mesmos splits.
"""

# Caracteres de controle ASCII (exceto \n e \t que são whitespace legítimos)
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")

# Whitespace múltiplo
WHITESPACE_PATTERN = re.compile(r"\s+")


# ============================================================
# FUNÇÕES DE NORMALIZAÇÃO
# ============================================================
def normalizar_texto(texto: str) -> str:
    """
    Pipeline completo de normalização de texto.

    Ordem importa:
    1. NFC normalize (composição de caracteres Unicode)
    2. Remove control chars
    3. Collapse whitespace
    4. Strip bordas
    """
    if not texto:
        return texto

    # 1. NFC: ã (1 char) > a + ̃ (2 chars)
    texto = unicodedata.normalize("NFC", texto)

    # 2. Remove caracteres de controle (exceto \n, \t, \r)
    texto = CONTROL_CHARS_PATTERN.sub("", texto)

    # 3. Collapse whitespace: "  \n\n  " → " "
    texto = WHITESPACE_PATTERN.sub(" ", texto)

    # 4. Strip bordas
    texto = texto.strip()

    return texto


def truncar_output(texto: str, max_chars: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    """Trunca output se exceder max_chars. Retorna (texto, foi_truncado)."""
    if len(texto) <= max_chars:
        return texto, False
    # Truncar e adicionar marcador
    return texto[:max_chars].rsplit(" ", 1)[0] + " ...", True


def normalizar_exemplo(obj: dict) -> tuple[dict, dict]:
    """
    Normaliza um exemplo completo. Retorna (obj_normalizado, stats).
    stats contém info de quantos caracteres foram truncados.
    """
    stats = {"truncado": False, "chars_originais": 0, "chars_finais": 0}

    # Normalizar todos os campos de texto
    for campo in ["instruction", "input", "output"]:
        if campo in obj and obj[campo]:
            obj[campo] = normalizar_texto(obj[campo])

    # Truncar output se necessário
    if "output" in obj:
        stats["chars_originais"] = len(obj["output"])
        obj["output"], truncado = truncar_output(obj["output"])
        stats["truncado"] = truncado
        stats["chars_finais"] = len(obj["output"])

    return obj, stats


# ============================================================
# FUNÇÕES DE SPLIT
# ============================================================
def split_data(data: list, ratios: tuple = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST),
               seed: int = RANDOM_SEED) -> tuple[list, list, list]:
    """
    Divide dados em train/val/test com shuffle e seed fixa.

    Por que shuffle? O MedQuAD vem ordenado por tópico.
    Sem shuffle, train não veria certas categorias, criando viés.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios devem somar 1.0"

    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)

    n = len(data)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    train = [data[i] for i in indices[:train_end]]
    val = [data[i] for i in indices[train_end:val_end]]
    test = [data[i] for i in indices[val_end:]]

    return train, val, test


# ============================================================
# FUNÇÕES DE RELATÓRIO
# ============================================================
def gerar_relatorio(
    n_input: int,
    n_train: int,
    n_val: int,
    n_test: int,
    stats_truncamento: Counter,
    stats_tamanho: dict,
) -> str:
    """Gera relatório legível do split."""
    rel = []
    rel.append("=" * 70)
    rel.append("RELATÓRIO DE SPLIT — Passo 2: Normalização + Train/Val/Test")
    rel.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rel.append("=" * 70)
    rel.append("")
    rel.append("📊 ESTATÍSTICAS DE NORMALIZAÇÃO")
    rel.append("-" * 70)
    rel.append(f"  Linhas lidas (entrada):           {n_input:>8,}")
    truncados = stats_truncamento.get(True, 0)
    rel.append(f"  Outputs truncados (>2500 chars): {truncados:>8,}")
    rel.append(f"  Taxa de truncamento:              {truncados/n_input*100:>6.2f}%")
    rel.append("")

    # Estatísticas de tamanho
    rel.append("📏 TAMANHO DOS OUTPUTS (pós-normalização)")
    rel.append("-" * 70)
    rel.append(f"  Média:  {stats_tamanho['media']:>8.0f} chars")
    rel.append(f"  Mediana:{stats_tamanho['mediana']:>8.0f} chars")
    rel.append(f"  Mín:    {stats_tamanho['min']:>8} chars")
    rel.append(f"  Máx:    {stats_tamanho['max']:>8} chars")
    rel.append("")

    # Estatísticas de split
    rel.append("✂️  SPLIT 90/5/5 (seed=42)")
    rel.append("-" * 70)
    rel.append(f"  Train:  {n_train:>8,} amostras ({n_train/(n_train+n_val+n_test)*100:.2f}%)")
    rel.append(f"  Val:     {n_val:>7,} amostras ({n_val/(n_train+n_val+n_test)*100:.2f}%)")
    rel.append(f"  Test:    {n_test:>7,} amostras ({n_test/(n_train+n_val+n_test)*100:.2f}%)")
    rel.append(f"  TOTAL:  {n_train+n_val+n_test:>8,} amostras")
    rel.append("")

    rel.append("📋 PARÂMETROS UTILIZADOS")
    rel.append("-" * 70)
    rel.append(f"  MAX_OUTPUT_CHARS = {MAX_OUTPUT_CHARS}")
    rel.append(f"  SPLIT_RATIOS = ({SPLIT_TRAIN}, {SPLIT_VAL}, {SPLIT_TEST})")
    rel.append(f"  RANDOM_SEED = {RANDOM_SEED}")
    rel.append(f"  Encoding = UTF-8 + NFC")
    rel.append(f"  Whitespace = collapse")
    rel.append("")

    rel.append("📁 ARQUIVOS GERADOS")
    rel.append("-" * 70)
    rel.append(f"  {OUTPUT_DIR}/train.jsonl  ({n_train:,} amostras)")
    rel.append(f"  {OUTPUT_DIR}/val.jsonl    ({n_val:,} amostras)")
    rel.append(f"  {OUTPUT_DIR}/test.jsonl   ({n_test:,} amostras)")
    rel.append("")
    rel.append("=" * 70)
    rel.append("✅ Pipeline de dados completo.")
    rel.append("   Próximo passo: notebooks/02_finetuning.ipynb no Colab Pro.")
    rel.append("=" * 70)
    return "\n".join(rel)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("✂️  PASSO 2: NORMALIZAÇÃO + SPLIT TRAIN/VAL/TEST")
    print("=" * 70)
    print(f"\n📂 Input:  {INPUT_FILE}")
    print(f"📂 Output: {OUTPUT_DIR}/\n")

    # Validar
    if not INPUT_FILE.exists():
        print(f"❌ ERRO: arquivo anonimizado não encontrado:\n   {INPUT_FILE}")
        print("\n💡 Dica: rode primeiro: python src/data/01_anonimizar.py")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Carregar dataset anonimizado
    print("📥 Carregando dataset anonimizado...")
    data = []
    with INPUT_FILE.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                data.append(json.loads(linha))
    print(f"   Carregadas: {len(data):,} linhas\n")

    # 2. Normalizar cada exemplo
    print("🔧 Aplicando normalização adicional...")
    data_norm = []
    stats_truncamento = Counter()
    tamanhos_output = []

    for obj in data:
        obj_norm, stats = normalizar_exemplo(obj)
        stats_truncamento[stats["truncado"]] += 1
        tamanhos_output.append(stats["chars_finais"])
        data_norm.append(obj_norm)

    # Estatísticas de tamanho
    tamanhos_ordenados = sorted(tamanhos_output)
    stats_tamanho = {
        "media": sum(tamanhos_output) / len(tamanhos_output),
        "mediana": tamanhos_ordenados[len(tamanhos_ordenados)//2],
        "min": min(tamanhos_output),
        "max": max(tamanhos_output),
    }

    print(f"   Outputs truncados: {stats_truncamento.get(True, 0):,}")
    print(f"   Tamanho médio:     {stats_tamanho['media']:.0f} chars")
    print(f"   Tamanho máximo:    {stats_tamanho['max']:,} chars\n")

    # 3. Split
    print("✂️  Dividindo em train/val/test (90/5/5, seed=42)...")
    train, val, test = split_data(data_norm)
    print(f"   Train: {len(train):,} amostras")
    print(f"   Val:   {len(val):,} amostras")
    print(f"   Test:  {len(test):,} amostras\n")

    # 4. Salvar 3 arquivos
    print("💾 Salvando arquivos finais...")
    for nome, dados in [("train.jsonl", train), ("val.jsonl", val), ("test.jsonl", test)]:
        path = OUTPUT_DIR / nome
        with path.open("w", encoding="utf-8") as f:
            for obj in dados:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        size_kb = path.stat().st_size / 1024
        print(f"   ✅ {nome:12} {len(dados):>7,} amostras ({size_kb:>7.1f} KB)")

    # 5. Relatório
    relatorio = gerar_relatorio(
        n_input=len(data_norm),
        n_train=len(train),
        n_val=len(val),
        n_test=len(test),
        stats_truncamento=stats_truncamento,
        stats_tamanho=stats_tamanho,
    )
    REPORT_FILE.write_text(relatorio, encoding="utf-8")

    print(f"\n📄 Relatório salvo em: {REPORT_FILE}")
    print("\n" + "=" * 70)
    print("🎯 PRÓXIMO PASSO: subir train.jsonl pro Colab Pro e rodar")
    print("   notebooks/02_finetuning.ipynb")
    print("=" * 70)


if __name__ == "__main__":
    main()