"""
Passo 2.5 do Tech Challenge Fase 3 — Validação Qualitativa
==========================================================

Este script faz 4 tipos de validação no dataset pré-treino:

1. ESTATÍSTICAS DESCRITIVAS — distribuições de tamanho, completude
2. AMOSTRAGEM ALEATÓRIA — lê 50 amostras reais para inspeção manual
3. DETECÇÃO DE ANOMALIAS — outputs suspeitos que escaparam dos filtros
4. COERÊNCIA TEMÁTICA — verifica se instruction+input são coerentes com output

Gera relatório completo em: data/processed/relatorio_validacao.txt
Gera arquivo de amostras em: data/processed/amostras_para_revisao.json

Uso:
    python src/data/03_validar_qualidade.py

Pré-requisito:
    Ter rodado 01_anonimizar.py e 02_normalizar_e_split.py

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
Data:  2026-08-31
"""

import json
import os
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ============================================================
# CONFIGURAÇÃO
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", SCRIPT_DIR.parent.parent))

INPUT_FILE = Path(os.environ.get(
    "INPUT_FILE",
    PROJECT_ROOT / "data" / "processed" / "train.jsonl"
))
REPORT_FILE = PROJECT_ROOT / "data" / "processed" / "relatorio_validacao.txt"
AMOSTRAS_FILE = PROJECT_ROOT / "data" / "processed" / "amostras_para_revisao.json"

N_AMOSTRAS = 50
RANDOM_SEED = 42

# ============================================================
# 1. ESTATÍSTICAS DESCRITIVAS
# ============================================================
def estatisticas_descritivas(data: list) -> dict:
    """Calcula distribuições de tamanho e completude."""
    stats = {
        "n_total": len(data),
        "instr_lengths": [],
        "input_lengths": [],
        "output_lengths": [],
        "outputs_vazios": 0,
        "instrucoes_vazias": 0,
        "outputs_sem_sentenca_completa": 0,
        "outputs_com_placeholders": 0,
        "outputs_com_phi_residual": 0,
        "outputs_com_apenas_urls": 0,
    }

    # Padrões de detecção
    PHI_RESIDUAL = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|@[\w.-]+\.[\w]+|https?://[\w./-]+")
    PLACEHOLDER_LIXO = re.compile(r"^(topics|frequently asked questions|faqs|key points)\s*$", re.IGNORECASE)
    SO_URL = re.compile(r"^https?://[\w./\-?=&%#]+\s*$")
    SEM_SENTENCA = re.compile(r"[.!?]$")

    for item in data:
        instr = item.get("instruction", "")
        inp = item.get("input", "")
        out = item.get("output", "")

        stats["instr_lengths"].append(len(instr))
        stats["input_lengths"].append(len(inp))
        stats["output_lengths"].append(len(out))

        if not instr.strip():
            stats["instrucoes_vazias"] += 1
        if not out.strip():
            stats["outputs_vazios"] += 1
        if out and not SEM_SENTENCA.search(out):
            stats["outputs_sem_sentenca_completa"] += 1
        if out and PLACEHOLDER_LIXO.match(out.strip()):
            stats["outputs_com_placeholders"] += 1
        if out and PHI_RESIDUAL.search(out):
            stats["outputs_com_phi_residual"] += 1
        if out and SO_URL.match(out.strip()):
            stats["outputs_com_apenas_urls"] += 1

    return stats


def formatar_estatisticas(stats: dict) -> str:
    """Formata estatísticas para relatório legível."""
    lines = []
    lines.append("📊 ESTATÍSTICAS DESCRITIVAS")
    lines.append("-" * 70)
    lines.append(f"  Total de amostras: {stats['n_total']:,}")
    lines.append("")

    for campo, label in [
        ("instr_lengths", "Instruction"),
        ("input_lengths", "Input"),
        ("output_lengths", "Output"),
    ]:
        vals = sorted(stats[campo])
        if not vals:
            continue
        n = len(vals)
        lines.append(f"  📏 {label}:")
        lines.append(f"     Min:    {vals[0]:>6,} chars")
        lines.append(f"     Mediana:{vals[n//2]:>6,} chars")
        lines.append(f"     Média:  {sum(vals)/n:>6,.0f} chars")
        lines.append(f"     Máx:    {vals[-1]:>6,} chars")
        lines.append(f"     P95:    {vals[int(n*0.95)]:>6,} chars")
        lines.append(f"     P99:    {vals[int(n*0.99)]:>6,} chars")
        lines.append("")

    lines.append("  🚨 ANOMALIAS DETECTADAS:")
    lines.append(f"     Instruções vazias:                  {stats['instrucoes_vazias']:>6,}")
    lines.append(f"     Outputs vazios:                     {stats['outputs_vazios']:>6,}")
    lines.append(f"     Outputs sem pontuação final:        {stats['outputs_sem_sentenca_completa']:>6,}")
    lines.append(f"     Outputs com placeholder (Topics...): {stats['outputs_com_placeholders']:>6,}")
    lines.append(f"     Outputs com PHI residual:           {stats['outputs_com_phi_residual']:>6,}")
    lines.append(f"     Outputs só com URL:                 {stats['outputs_com_apenas_urls']:>6,}")
    return "\n".join(lines)


# ============================================================
# 2. DETECÇÃO DE ANOMALIAS
# ============================================================
def detectar_anomalias(data: list) -> list[dict]:
    """Encontra amostras suspeitas que escaparam dos filtros anteriores."""
    anomalias = []

    # Padrões
    PHI_RESIDUAL = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|@[\w.-]+\.[\w]+|https?://[\w./-]+")
    PLACEHOLDER_LIXO = re.compile(r"^(topics|frequently asked questions|faqs|key points|see more)\s*$", re.IGNORECASE)
    SO_PERGUNTA = re.compile(r"^(how might|what are|is it|are there|how many people|who is at risk)", re.IGNORECASE)
    REPETICAO = re.compile(r"^(.{20,})\1", re.IGNORECASE)  # mesma frase repetida

    for i, item in enumerate(data):
        instr = item.get("instruction", "")
        out = item.get("output", "")

        anomalias_item = []

        # 1. PHI não anonimizado
        if PHI_RESIDUAL.search(out):
            anomalias_item.append("phi_residual")

        # 2. Placeholder do XML
        if PLACEHOLDER_LIXO.match(out.strip()):
            anomalias_item.append("placeholder_xml")

        # 3. Output é só reformulação da pergunta
        if len(out) < 100 and SO_PERGUNTA.match(out.strip()):
            anomalias_item.append("reformulacao_pergunta")

        # 4. Output repete instruction textualmente
        instr_norm = instr.strip().rstrip("?.!").lower()
        out_norm = out.strip().rstrip("?.!").lower()
        if instr_norm and instr_norm == out_norm:
            anomalias_item.append("repeticao_literal_pergunta")

        # 5. Texto repetido (possível corrupção)
        if REPETICAO.search(out):
            anomalias_item.append("texto_repetido")

        # 6. Output muito curto apesar do filtro (50-100 chars sem ponto final)
        if 50 <= len(out) < 100 and not out.strip().endswith((".", "!", "?")):
            anomalias_item.append("muito_curto_sem_pontuacao")

        if anomalias_item:
            anomalias.append({
                "indice": i,
                "instruction": instr[:100],
                "output_preview": out[:150] + "..." if len(out) > 150 else out,
                "problemas": anomalias_item,
            })

    return anomalias


# ============================================================
# 3. COERÊNCIA TEMÁTICA
# ============================================================
def analisar_coerencia(data: list, n_amostras: int = 100) -> dict:
    """
    Verifica se instruction+input são coerentes com output.
    Heurística: contar palavras-chave do tópico que aparecem no output.
    """
    rng = random.Random(RANDOM_SEED)
    amostras = rng.sample(data, min(n_amostras, len(data)))

    resultados = {
        "total_analisado": len(amostras),
        "coerentes": 0,
        "parcialmente_coerentes": 0,
        "incoerentes": 0,
        "exemplos": [],
    }

    for item in amostras:
        instr = item.get("instruction", "")
        inp = item.get("input", "")
        out = item.get("output", "")

        # Extrair palavras-chave do tópico (do input)
        topic = inp.replace("Context / Topic:", "").strip()

        # Verificar se palavras do tópico aparecem no output
        topic_words = [w.lower() for w in re.findall(r"\b\w{4,}\b", topic) if len(w) > 3]
        out_lower = out.lower()
        topic_words_present = sum(1 for w in topic_words if w in out_lower)

        # Heurística simples de coerência
        ratio = topic_words_present / len(topic_words) if topic_words else 1.0

        if ratio > 0.5:
            resultados["coerentes"] += 1
            categoria = "coerente"
        elif ratio > 0.2:
            resultados["parcialmente_coerentes"] += 1
            categoria = "parcial"
        else:
            resultados["incoerentes"] += 1
            categoria = "incoerente"

        # Guardar exemplos de cada categoria
        if categoria != "coerente" and len(resultados["exemplos"]) < 5:
            resultados["exemplos"].append({
                "instruction": instr[:100],
                "topic": topic,
                "topic_words_encontradas": topic_words_present,
                "total_topic_words": len(topic_words),
                "categoria": categoria,
            })

    return resultados


# ============================================================
# 4. AMOSTRAGEM PARA REVISÃO MANUAL
# ============================================================
def gerar_amostras_revisao(data: list, n: int = N_AMOSTRAS) -> list[dict]:
    """Seleciona n amostras aleatórias estratificadas por tamanho de output."""
    rng = random.Random(RANDOM_SEED)

    # Divide em 3 estratos: curto / médio / longo
    curtos = [d for d in data if len(d.get("output", "")) < 500]
    medios = [d for d in data if 500 <= len(d.get("output", "")) < 1500]
    longos = [d for d in data if len(d.get("output", "")) >= 1500]

    # Pega proporcional
    n_curto = n // 3
    n_medio = n // 3
    n_longo = n - n_curto - n_medio

    amostras = []
    amostras.extend(rng.sample(curtos, min(n_curto, len(curtos))))
    amostras.extend(rng.sample(medios, min(n_medio, len(medios))))
    amostras.extend(rng.sample(longos, min(n_longo, len(longos))))

    rng.shuffle(amostras)

    # Adiciona metadata para revisão
    for i, a in enumerate(amostras):
        a["_revisao"] = {
            "indice": i + 1,
            "tamanho_categoria": (
                "curto" if len(a.get("output", "")) < 500
                else "medio" if len(a.get("output", "")) < 1500
                else "longo"
            ),
            "output_chars": len(a.get("output", "")),
            "instrucao_clara": "OK",  # placeholder para revisão humana
        }

    return amostras


# ============================================================
# RELATÓRIO
# ============================================================
def gerar_relatorio(
    stats: dict,
    stats_str: str,
    anomalias: list,
    coerencia: dict,
    amostras: list,
) -> str:
    """Monta relatório completo de validação."""
    rel = []
    rel.append("=" * 70)
    rel.append("RELATÓRIO DE VALIDAÇÃO QUALITATIVA — Passo 2.5")
    rel.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    rel.append(f"Dataset: {INPUT_FILE.name}")
    rel.append("=" * 70)
    rel.append("")
    rel.append(stats_str)
    rel.append("")

    # Anomalias
    rel.append("🚨 ANOMALIAS DETECTADAS (top 20 amostras)")
    rel.append("-" * 70)
    if anomalias:
        rel.append(f"Total encontrado: {len(anomalias):,} amostras com problemas")
        rel.append("")
        rel.append("Distribuição por tipo de problema:")
        problemas_count = Counter()
        for a in anomalias:
            for p in a["problemas"]:
                problemas_count[p] += 1
        for p, c in problemas_count.most_common():
            rel.append(f"  • {p:<35} {c:>6,} amostras")
        rel.append("")
        rel.append("Primeiras 20 anomalias (para revisão):")
        rel.append("")
        for i, a in enumerate(anomalias[:20], 1):
            rel.append(f"  [{i}] {', '.join(a['problemas'])}")
            rel.append(f"      Q: {a['instruction']}")
            rel.append(f"      R: {a['output_preview']}")
            rel.append("")
    else:
        rel.append("✅ Nenhuma anomalia crítica detectada!")
    rel.append("")

    # Coerência
    rel.append("🎯 ANÁLISE DE COERÊNCIA TEMÁTICA (amostra de " + str(coerencia["total_analisado"]) + ")")
    rel.append("-" * 70)
    rel.append(f"  Coerentes:              {coerencia['coerentes']:>4} ({coerencia['coerentes']/coerencia['total_analisado']*100:.1f}%)")
    rel.append(f"  Parcialmente coerentes: {coerencia['parcialmente_coerentes']:>4} ({coerencia['parcialmente_coerentes']/coerencia['total_analisado']*100:.1f}%)")
    rel.append(f"  Incoerentes:            {coerencia['incoerentes']:>4} ({coerencia['incoerentes']/coerencia['total_analisado']*100:.1f}%)")
    rel.append("")
    if coerencia["exemplos"]:
        rel.append("Exemplos de baixa coerência:")
        for ex in coerencia["exemplos"][:3]:
            rel.append(f"  • {ex['instruction'][:80]}")
            rel.append(f"    Tópico: {ex['topic']}")
            rel.append(f"    Palavras do tópico no output: {ex['topic_words_encontradas']}/{ex['total_topic_words']}")
            rel.append("")

    # Veredicto final
    rel.append("=" * 70)
    rel.append("📋 VEREDICTO PARA FINE-TUNING")
    rel.append("=" * 70)

    # Score baseado em anomalias e coerência
    score_anomalias = 100 - (len(anomalias) / stats["n_total"] * 100)
    score_coerencia = coerencia["coerentes"] / coerencia["total_analisado"] * 100
    score_final = (score_anomalias + score_coerencia) / 2

    if score_final >= 90:
        veredito = "🟢 EXCELENTE — Pronto para fine-tuning"
        rel.append(f"  Score de qualidade: {score_final:.1f}/100")
        rel.append(f"  Recomendação: {veredito}")
    elif score_final >= 75:
        veredito = "🟡 BOM — Pode fine-tunar, mas revise amostras críticas"
        rel.append(f"  Score de qualidade: {score_final:.1f}/100")
        rel.append(f"  Recomendação: {veredito}")
    else:
        veredito = "🔴 PREOCUPANTE — Voltar e revisar filtros do passo 1/2"
        rel.append(f"  Score de qualidade: {score_final:.1f}/100")
        rel.append(f"  Recomendação: {veredito}")

    rel.append("")
    rel.append("📁 PRÓXIMOS PASSOS:")
    rel.append(f"  1. Revisar manualmente: {AMOSTRAS_FILE.name}")
    rel.append(f"     ({len(amostras)} amostras estratificadas)")
    rel.append("  2. Se veredicto verde → seguir para fine-tuning no Colab Pro")
    rel.append("  3. Se veredicto amarelo → considerar adicionar dados sintéticos")
    rel.append("  4. Se veredicto vermelho → revisar filtros e re-rodar passos 1/2")
    rel.append("=" * 70)
    return "\n".join(rel)


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("🔬 PASSO 2.5: VALIDAÇÃO QUALITATIVA DO DATASET")
    print("=" * 70)
    print(f"\n📂 Input:  {INPUT_FILE}")

    if not INPUT_FILE.exists():
        print(f"\n❌ ERRO: arquivo não encontrado.")
        print("💡 Rode primeiro: 02_normalizar_e_split.py")
        return

    # Carregar dados
    print("\n📥 Carregando dataset...")
    data = []
    with INPUT_FILE.open(encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                data.append(json.loads(linha))
    print(f"   Carregadas: {len(data):,} amostras")

    # 1. Estatísticas descritivas
    print("\n📊 Calculando estatísticas descritivas...")
    stats = estatisticas_descritivas(data)
    stats_str = formatar_estatisticas(stats)
    print(f"   Outputs vazios: {stats['outputs_vazios']:,}")
    print(f"   Outputs com PHI residual: {stats['outputs_com_phi_residual']:,}")

    # 2. Detecção de anomalias
    print("\n🚨 Detectando anomalias...")
    anomalias = detectar_anomalias(data)
    print(f"   Anomalias encontradas: {len(anomalias):,}")

    # 3. Coerência temática
    print("\n🎯 Analisando coerência temática (amostra 100)...")
    coerencia = analisar_coerencia(data, n_amostras=100)
    print(f"   Coerentes: {coerencia['coerentes']} | Parciais: {coerencia['parcialmente_coerentes']} | Incoerentes: {coerencia['incoerentes']}")

    # 4. Amostras para revisão manual
    print(f"\n📋 Gerando {N_AMOSTRAS} amostras estratificadas para revisão...")
    amostras = gerar_amostras_revisao(data, n=N_AMOSTRAS)

    # Salvar amostras para revisão
    AMOSTRAS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with AMOSTRAS_FILE.open("w", encoding="utf-8") as f:
        json.dump(amostras, f, ensure_ascii=False, indent=2)
    print(f"   Salvas em: {AMOSTRAS_FILE}")

    # Gerar e salvar relatório
    relatorio = gerar_relatorio(stats, stats_str, anomalias, coerencia, amostras)
    REPORT_FILE.write_text(relatorio, encoding="utf-8")

    # Imprimir relatório
    print("\n" + "=" * 70)
    print("RELATÓRIO COMPLETO")
    print("=" * 70)
    print(relatorio)
    print(f"\n📄 Relatório salvo em: {REPORT_FILE}")


if __name__ == "__main__":
    main()