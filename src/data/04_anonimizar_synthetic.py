"""
Passo 4 do Tech Challenge Fase 3 — Anonimização de Synthetic Clinical Notes
=========================================================================

Este script:
1. Lê o dataset Synthetic Clinical Notes (HuggingFace)
2. Detecta e substitui PHI-like por placeholders:
   - Nomes de pacientes: "Harvey D'Amore" → [NOME_PACIENTE]
   - Datas de nascimento: "September 12, 1964" → [DATA_NASCIMENTO]
   - IDs/MRN: "HD-AM-0901964" → [ID_PACIENTE]
   - Nomes de médicos: "Dr. Emily Thompson" → [NOME_MEDICO]
   - Endereços: "Springfield, IL" → [ENDERECO]
   - Telefones, e-mails (raros, mas possível)
3. Salva dataset anonimizado em formato JSONL (compatível com nosso pipeline)
4. Gera relatório de curadoria

Uso:
    python src/data/04_anonimizar_synthetic.py

Pré-requisito:
    Ter rodado os downloads (dados em data/raw/synthetic_clinical_notes/)

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
Data:  2026-08-31

NOTA: os dados são SINTÉTICOS (pacientes fictícios), mas anonimizamos por:
1. Boa prática LGPD
2. Evitar que o modelo aprenda formato de PHI
3. Defesa em auditoria
"""

import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk, Dataset

# ============================================================
# CONFIGURAÇÃO DE PATHS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent.parent))

INPUT_DIR = Path(os.environ.get(
    "INPUT_DIR",
    PROJECT_ROOT / "data" / "raw" / "synthetic_clinical_notes"
))
OUTPUT_FILE = Path(os.environ.get(
    "OUTPUT_FILE",
    PROJECT_ROOT / "data" / "processed" / "synthetic_clinical_notes_anonimizado.jsonl"
))
REPORT_FILE = Path(os.environ.get(
    "REPORT_FILE",
    PROJECT_ROOT / "data" / "processed" / "relatorio_anonimizacao_synthetic.txt"
))

# ============================================================
# PADRÕES DE PHI (calibrados no Synthetic Clinical Notes)
# ============================================================
# Estes regex são específicos pra estrutura de notas clínicas
# Formato típico: "Patient Name: <nome>"
PHI_PATTERNS: dict[str, re.Pattern] = {
    # Nomes de pacientes
    "patient_name": re.compile(
        r"Patient Name:\*?\*?\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)+)",
        re.IGNORECASE
    ),

    # Datas de nascimento (formato texto ou numérico)
    "dob": re.compile(
        r"(?:Date of Birth|DOB|Birth Date):\*?\*?\s+([\w\s,/-]+?)(?=\n|\*\*|\s\s|$)",
        re.IGNORECASE
    ),

    # Medical Record Number / IDs
    "mrn": re.compile(
        r"(?:Medical Record Number|MRN|Patient ID|MRN|Medical Record #):\*?\*?\s+(\S+)",
        re.IGNORECASE
    ),

    # Nomes de médicos
    "doctor_name": re.compile(
        r"(?:Dr\.|Dra\.|Dr |Doctor)\s+([A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)+)",
        re.IGNORECASE
    ),

    # Local/Endereço (cidade + estado)
    "location": re.compile(
        r"Location:\*?\*?\s+([^,\n]+,\s*[A-Z]{2})",
        re.IGNORECASE
    ),

    # Telefones
    "telefone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),

    # E-mails
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),

    # CEPs US (5 dígitos + -3 dígitos)
    "zip": re.compile(r"\b\d{5}-\d{4}\b"),
}

# ============================================================
# FUNÇÕES
# ============================================================
def anonimizar_texto(texto: str) -> tuple[str, dict[str, int]]:
    """
    Aplica todos os padrões de PHI no texto.
    Retorna (texto_anonimizado, contadores_por_tipo).
    """
    if not texto:
        return texto, {}

    contadores = {}
    for tipo, regex in PHI_PATTERNS.items():
        placeholder = f"[{tipo.upper()}]"
        texto, n = regex.subn(placeholder, texto)
        if n > 0:
            contadores[tipo] = n
    return texto, contadores


def anonimizar_exemplo(example: dict) -> tuple[dict, dict]:
    """
    Anonimiza um exemplo (note + encounter_data).
    Retorna (exemplo_anonimizado, contadores).
    """
    phi_total = Counter()
    exemplo = dict(example)  # cópia

    # 1. Anonimizar o campo 'note' (string ou lista)
    note = exemplo.get("note", "")
    if isinstance(note, list):
        for i, n in enumerate(note):
            n_anon, cont = anonimizar_texto(str(n))
            exemplo["note"][i] = n_anon
            phi_total.update(cont)
    else:
        note_anon, cont = anonimizar_texto(str(note))
        exemplo["note"] = note_anon
        phi_total.update(cont)

    # 2. Anonimizar encounter_data (dict)
    encounter = exemplo.get("encounter_data", {})
    if isinstance(encounter, dict):
        encounter_anon = {}
        for k, v in encounter.items():
            v_anon, cont = anonimizar_texto(str(v))
            encounter_anon[k] = v_anon
            phi_total.update(cont)
        exemplo["encounter_data"] = encounter_anon
    elif isinstance(encounter, str):
        encounter_anon, cont = anonimizar_texto(encounter)
        exemplo["encounter_data"] = encounter_anon
        phi_total.update(cont)

    return exemplo, dict(phi_total)


def gerar_relatorio(
    n_input: int,
    n_output: int,
    phi_total: Counter,
    exemplos_antes_depois: list[dict],
) -> str:
    """Gera relatório legível."""
    rel = []
    rel.append("=" * 70)
    rel.append("RELATÓRIO DE ANONIMIZAÇÃO — Synthetic Clinical Notes")
    rel.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rel.append("=" * 70)
    rel.append("")
    rel.append("📊 ESTATÍSTICAS GERAIS")
    rel.append("-" * 70)
    rel.append(f"  Amostras lidas:     {n_input:>8,}")
    rel.append(f"  Amostras escritas:   {n_output:>8,}")
    rel.append("")
    rel.append("🔒 PHI SUBSTITUÍDO POR TIPO")
    rel.append("-" * 70)
    if phi_total:
        for tipo, count in phi_total.most_common():
            rel.append(f"  {tipo:<20} {count:>6,} substituições")
        rel.append(f"  {'TOTAL':<20} {sum(phi_total.values()):>6,} substituições")
    else:
        rel.append("  Nenhum PHI detectado.")
    rel.append("")
    if exemplos_antes_depois:
        rel.append("📋 EXEMPLO ANTES/DEPOIS")
        rel.append("-" * 70)
        for i, ex in enumerate(exemplos_antes_depois[:2], 1):
            rel.append(f"\n--- Exemplo {i} ---")
            rel.append("ANTES (trecho):")
            rel.append(f"  {ex['antes']}")
            rel.append("\nDEPOIS (trecho):")
            rel.append(f"  {ex['depois']}")
    rel.append("")
    rel.append("=" * 70)
    rel.append("✅ Anonimização concluída.")
    rel.append("=" * 70)
    return "\n".join(rel)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🔒 PASSO 4: ANONIMIZAÇÃO DO SYNTHETIC CLINICAL NOTES")
    print("=" * 70)
    print(f"\n📂 Input:  {INPUT_DIR}")
    print(f"📂 Output: {OUTPUT_FILE}\n")

    if not INPUT_DIR.exists():
        print(f"❌ ERRO: dataset não encontrado em {INPUT_DIR}")
        print("💡 Rode os downloads primeiro")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Carregar dataset HuggingFace
    print("📥 Carregando dataset HuggingFace...")
    ds = load_from_disk(str(INPUT_DIR))
    print(f"   Splits: {list(ds.keys())}")

    # Processar todos os splits
    phi_total = Counter()
    total_input = 0
    total_output = 0
    exemplos_antes_depois = []

    with OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for split_name in ds.keys():
            split_data = ds[split_name]
            print(f"\n🔄 Processando split '{split_name}' ({len(split_data):,} amostras)...")

            for i, ex in enumerate(split_data):
                total_input += 1
                ex_anon, cont = anonimizar_exemplo(dict(ex))
                phi_total.update(cont)

                # Guardar primeiro exemplo pra relatório
                if i == 0 and len(exemplos_antes_depois) < 2:
                    note_orig = ex.get("note", "")
                    note_anon = ex_anon.get("note", "")
                    if isinstance(note_orig, list):
                        note_orig = " ".join(str(x) for x in note_orig)
                        note_anon = " ".join(str(x) for x in note_anon)
                    exemplos_antes_depois.append({
                        "antes": str(note_orig)[:400],
                        "depois": str(note_anon)[:400],
                    })

                # Salvar como JSONL (apenas campos úteis)
                record = {
                    "note": ex_anon.get("note", ""),
                    "encounter_data": ex_anon.get("encounter_data", {}),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_output += 1

                if (i + 1) % 500 == 0:
                    print(f"   ... {i+1:,}/{len(split_data):,}")

    # Relatório
    relatorio = gerar_relatorio(total_input, total_output, phi_total, exemplos_antes_depois)
    REPORT_FILE.write_text(relatorio, encoding="utf-8")

    print()
    print("=" * 70)
    print("✅ PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)
    print(f"  Amostras lidas:     {total_input:>8,}")
    print(f"  Amostras escritas:   {total_output:>8,}")
    print(f"\n🔒 PHI substituído (total): {sum(phi_total.values()):,} ocorrências")
    for tipo, count in phi_total.most_common(5):
        print(f"     • {tipo}: {count:,}")

    print(f"\n📄 Relatório: {REPORT_FILE}")
    print(f"📄 Dataset limpo: {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print("🎯 PRÓXIMO PASSO: indexar no ChromaDB pra RAG")
    print("=" * 70)


if __name__ == "__main__":
    main()