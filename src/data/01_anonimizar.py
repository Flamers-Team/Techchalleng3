"""
Passo 1 do Tech Challenge Fase 3 — Pipeline de Dados
====================================================

Este script:
1. Lê o dataset bruto: data/raw/medquad_finetuning.jsonl
2. Aplica anonimização (regex) removendo PHI/PII
3. Filtra outputs de baixa qualidade (< 50 chars ou que repetem pergunta)
4. Remove duplicatas exatas
5. Salva em: data/processed/medquad_anonimizado.jsonl
6. Gera relatório de curadoria em: data/processed/relatorio_curadoria.txt

Uso:
    python src/data/01_anonimizar.py

Requisitos:
    - Arquivo de entrada em data/raw/medquad_finetuning.jsonl
    - Python 3.10+

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
Data:  2026-08-31
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# ============================================================
# CONFIGURAÇÃO DE PATHS
# ============================================================
# Por padrão, assume estrutura: projeto/data/raw/, projeto/data/processed/
# Pode ser sobrescrito via variáveis de ambiente se rodar em outro lugar.
import os

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent.parent))

# Permite apontar input/output customizados via env vars
INPUT_FILE = Path(os.environ.get(
    "INPUT_FILE",
    PROJECT_ROOT / "data" / "raw" / "medquad_finetuning.jsonl"
))
OUTPUT_FILE = Path(os.environ.get(
    "OUTPUT_FILE",
    PROJECT_ROOT / "data" / "processed" / "medquad_anonimizado.jsonl"
))
REPORT_FILE = Path(os.environ.get(
    "REPORT_FILE",
    PROJECT_ROOT / "data" / "processed" / "relatorio_curadoria.txt"
))

# ============================================================
# PADRÕES DE PHI/PII (calibrados no MedQuAD/NIH)
# ============================================================
# Cada padrão detecta um tipo de identificador e substitui por um placeholder.
# Os placeholders preservam a estrutura da frase sem revelar o conteúdo.

PHI_PATTERNS: dict[str, re.Pattern] = {
    # Telefones US formato (XXX) XXX-XXXX ou XXX-XXX-XXXX ou XXX.XXX.XXXX
    "telefone": re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"),

    # SSN americano: 123-45-6789
    "ssn": re.compile(r"\b\d{3}-?\d{2}-?\d{4}\b"),

    # CPF brasileiro: 123.456.789-00 ou 12345678900
    "cpf": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),

    # E-mail: usuario@dominio.com
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),

    # URLs: http://... ou https://...
    "url": re.compile(r"https?://[\w./\-?=&%#]+"),

    # Datas numéricas: 12/05/2024, 3/30/2011, 15-03-2024
    "data_numerica": re.compile(r"\b\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}\b"),

    # CEP brasileiro: 70000-000 ou 70000000
    "cep": re.compile(r"\b\d{5}-?\d{3}\b"),

    # Números de prontuário comuns (PTO-1234, PRONT-5678)
    "prontuario": re.compile(r"\b(?:prontuário|pront|p[rt]o|pto)[\s\-_]*[nºn°]?[\s\-_]*\d+\b", re.IGNORECASE),

    # Cartão Nacional de Saúde (15 dígitos)
    "cns": re.compile(r"\b\d{15}\b"),
}

# ============================================================
# FUNÇÕES PRINCIPAIS
# ============================================================
def anonimizar(texto: str) -> tuple[str, dict[str, int]]:
    """
    Aplica todos os padrões de PHI no texto.
    Retorna (texto_anonimizado, dict_contadores_por_tipo).
    """
    if not texto:
        return texto, {}

    contadores = {}
    for tipo, regex in PHI_PATTERNS.items():
        texto, n = regex.subn(f"[{tipo.upper()}]", texto)
        if n > 0:
            contadores[tipo] = n
    return texto, contadores


def normalizar_espacos(texto: str) -> str:
    """Colapsa múltiplos espaços/newlines em um único espaço."""
    return re.sub(r"\s+", " ", texto).strip()


def eh_resposta_invalida(instruction: str, output: str) -> tuple[bool, str]:
    """
    Verifica se a resposta é lixo. Retorna (eh_invalida, motivo).
    """
    # 1. Output vazio ou muito curto
    if len(output.strip()) < 50:
        return True, f"output_curto ({len(output.strip())} chars)"

    # 2. Output repete literalmente a pergunta
    instr_norm = instruction.strip().rstrip("?.!").lower()
    out_norm = output.strip().rstrip("?.!").lower()
    if instr_norm == out_norm:
        return True, "output_repete_pergunta"

    # 3. Output é só uma reformulação da pergunta (heurística simples)
    if len(out_norm) < 80 and out_norm.startswith(("how might", "what are", "is it", "are there")):
        return True, "output_reformulacao_pergunta"

    # 4. Placeholders comuns do XML original (NIH usa "Topics", "FAQs" como seção)
    placeholders_lixo = [
        "topics",
        "frequently asked questions",
        "faqs",
        "key points",
    ]
    out_lower = output.strip().lower()
    if out_lower in placeholders_lixo:
        return True, f"placeholder_xml ({output.strip()[:30]})"

    return False, ""


def processar_linha(obj: dict) -> tuple[dict | None, dict, str]:
    """
    Processa uma linha: anonimiza + filtra. Retorna (obj_processado, contadores_phi, status).

    Status pode ser: 'ok', 'invalida', 'duplicada'.
    """
    # 1. Anonimizar instruction, input, output
    phi_total = Counter()
    for campo in ["instruction", "input", "output"]:
        if campo in obj and obj[campo]:
            obj[campo], cont = anonimizar(obj[campo])
            phi_total.update(cont)

    # 2. Normalizar espaços
    for campo in ["instruction", "input", "output"]:
        if campo in obj:
            obj[campo] = normalizar_espacos(obj[campo])

    # 3. Verificar qualidade
    invalida, motivo = eh_resposta_invalida(obj.get("instruction", ""), obj.get("output", ""))
    if invalida:
        return None, dict(phi_total), f"invalida:{motivo}"

    return obj, dict(phi_total), "ok"


def gerar_relatorio(stats: dict, phi_total_global: Counter,
                    linhas_descartadas_motivos: Counter) -> str:
    """Gera relatório legível de curadoria."""
    relatorio = []
    relatorio.append("=" * 70)
    relatorio.append("RELATÓRIO DE CURADORIA — Passo 1: Anonimização MedQuAD")
    relatorio.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    relatorio.append("=" * 70)
    relatorio.append("")
    relatorio.append("📊 ESTATÍSTICAS GERAIS")
    relatorio.append("-" * 70)
    relatorio.append(f"  Linhas lidas (entrada):         {stats['lidas']:>8,}")
    relatorio.append(f"  Linhas escritas (saída):        {stats['escritas']:>8,}")
    relatorio.append(f"  Taxa de aproveitamento:         {stats['taxa_aprov']:>7.2f}%")
    relatorio.append("")
    relatorio.append("🗑️  DESCARTE POR FILTRO DE QUALIDADE")
    relatorio.append("-" * 70)
    for motivo, count in linhas_descartadas_motivos.most_common():
        relatorio.append(f"  {motivo:<35} {count:>6,}")
    relatorio.append(f"  {'TOTAL DESCARTADAS':<35} {stats['descartadas']:>6,}")
    relatorio.append("")
    relatorio.append("🔒 PHI/PII SUBSTITUÍDO POR TIPO")
    relatorio.append("-" * 70)
    if phi_total_global:
        for tipo, count in phi_total_global.most_common():
            relatorio.append(f"  {tipo:<20} {count:>6,} substituições")
        relatorio.append(f"  {'TOTAL':<20} {sum(phi_total_global.values()):>6,} substituições")
    else:
        relatorio.append("  Nenhum PHI detectado.")
    relatorio.append("")
    relatorio.append("📋 CONFIGURAÇÃO")
    relatorio.append("-" * 70)
    relatorio.append(f"  Input:  {INPUT_FILE}")
    relatorio.append(f"  Output: {OUTPUT_FILE}")
    relatorio.append(f"  Patterns ativos: {len(PHI_PATTERNS)}")
    relatorio.append("")
    relatorio.append("=" * 70)
    relatorio.append("✅ Curadoria concluída.")
    relatorio.append("=" * 70)
    return "\n".join(relatorio)


# ============================================================
# MAIN
# ============================================================
def main():
    """Executa o pipeline completo de anonimização."""
    print("=" * 70)
    print("🔒 PASSO 1: ANONIMIZAÇÃO + CURADORIA DO MEDQUAD")
    print("=" * 70)
    print(f"\n📂 Input:  {INPUT_FILE}")
    print(f"📂 Output: {OUTPUT_FILE}\n")

    # Validar paths
    if not INPUT_FILE.exists():
        print(f"❌ ERRO: arquivo de entrada não encontrado:\n   {INPUT_FILE}")
        print("\n💡 Dica: verifique se o arquivo está em data/raw/medquad_finetuning.jsonl")
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Contadores globais
    lidas = 0
    escritas = 0
    descartadas = 0
    vistas = set()  # para detectar duplicatas exatas
    phi_global = Counter()
    descartes_motivo = Counter()

    # Processar linha a linha
    with INPUT_FILE.open(encoding="utf-8") as fin, OUTPUT_FILE.open("w", encoding="utf-8") as fout:
        for linha in fin:
            linha = linha.strip()
            if not linha:
                continue
            lidas += 1

            try:
                obj = json.loads(linha)
            except json.JSONDecodeError:
                descartadas += 1
                descartes_motivo["json_invalido"] += 1
                continue

            # 1. Processar (anonimiza + normaliza + valida qualidade)
            obj_proc, phi, status = processar_linha(obj)
            phi_global.update(phi)

            if obj_proc is None:
                descartadas += 1
                descartes_motivo[status.replace("invalida:", "")] += 1
                continue

            # 2. Detectar duplicata exata (instruction + output)
            chave_dedup = (obj_proc["instruction"], obj_proc["output"])
            if chave_dedup in vistas:
                descartadas += 1
                descartes_motivo["duplicata_exata"] += 1
                continue
            vistas.add(chave_dedup)

            # 3. Escrever linha válida
            fout.write(json.dumps(obj_proc, ensure_ascii=False) + "\n")
            escritas += 1

            # Log de progresso a cada 2000 linhas
            if lidas % 2000 == 0:
                print(f"   ... processadas {lidas:,} linhas ({escritas:,} mantidas)")

    # Estatísticas finais
    taxa = (escritas / lidas * 100) if lidas > 0 else 0.0
    stats = {
        "lidas": lidas,
        "escritas": escritas,
        "descartadas": descartadas,
        "taxa_aprov": taxa,
    }

    # Imprimir resumo
    print("\n" + "=" * 70)
    print("✅ PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)
    print(f"  Linhas lidas:      {lidas:>8,}")
    print(f"  Linhas escritas:   {escritas:>8,}")
    print(f"  Linhas descartadas:{descartadas:>8,}")
    print(f"  Taxa aproveitamento: {taxa:>6.2f}%")
    print(f"\n🔒 PHI substituído (total): {sum(phi_global.values()):,} ocorrências")
    for tipo, n in phi_global.most_common(5):
        print(f"     • {tipo}: {n:,}")

    # Gerar relatório
    relatorio = gerar_relatorio(stats, phi_global, descartes_motivo)
    REPORT_FILE.write_text(relatorio, encoding="utf-8")
    print(f"\n📄 Relatório salvo em: {REPORT_FILE}")
    print(f"📄 Dataset limpo em:   {OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print("🎯 PRÓXIMO PASSO: rodar 02_normalizar.py")
    print("=" * 70)


if __name__ == "__main__":
    main()