"""
Gradio App FINAL — Assistente Médico (Tech Challenge Fase 3)
=============================================================

Integra:
- LLM real (BioMistral fine-tuned) ou mock
- RAG (ChromaDB) com 3 vector stores
- Gerador de PDFs (ReportLab)
- Sistema de auditoria (SQLite)
- HITL (médico ratifica antes de gerar docs)

Uso:
    python src/ui/gradio_app.py
    # abre em http://127.0.0.1:7860
    # login: medico / demo123
"""

import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr

# Adicionar raiz ao path pra imports funcionarem
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ============================================================
# CONFIGURAÇÃO
# ============================================================
DB_PATH = Path("audit.db")
DOCS_DIR = Path("data/documents")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

SESSION_ID = str(uuid.uuid4())[:8]
USER_ID = "dr.demonstracao"

# ============================================================
# PIPELINE COMPLETO (integra tudo)
# ============================================================
def inicializar_componentes():
    """Inicializa LLM + Retriever + DocGenerator (lazy loading)."""
    global _llm, _retriever, _doc_gen
    if "_llm" not in globals():
        from src.llm.client import get_llm
        from src.rag.retriever import Retriever
        from src.docs.generator import DocumentGenerator

        _llm = get_llm()
        try:
            _retriever = Retriever()
        except Exception as e:
            print(f"⚠️  Retriever não carregado: {e}")
            _retriever = None
        _doc_gen = DocumentGenerator(output_dir=str(DOCS_DIR))


def processar_consulta(relato: str, nome_paciente: str, idade: str, sexo: str, progress=gr.Progress()):
    """Pipeline completo: LLM + RAG → síntese → validação."""
    inicializar_componentes()

    progress(0.1, desc="🔍 Etapa 1/5: Triagem clínica...")

    # Agente 1: Triagem (via src/agents/triagem.py)
    from src.agents.triagem import triar
    tri = triar(relato)
    log_event("triagem", agent="triagem", input_data=relato, output_data=tri)

    progress(0.3, desc="📚 Etapa 2/5: Buscando literatura (PMC)...")
    # RAG PMC (mock até indexar artigos reais)
    rag_pmc = []
    if _retriever:
        rag_pmc = _retriever.retrieve_pmc(relato, k=4)

    progress(0.5, desc="🏥 Etapa 3/5: Buscando base interna...")
    # RAG interno (ANVISA + CID-10 + Synthetic)
    rag_interno = []
    if _retriever:
        rag_interno = _retriever.retrieve_interno(relato, k=4)

    progress(0.7, desc="🧠 Etapa 4/5: Gerando síntese clínica...")
    # Agente 2: Síntese
    from src.agents.sintese import sintetizar
    sintese = sintetizar(relato, rag_pmc, rag_interno)
    log_event("sintese", agent="sintese", input_data=relato, output_data=sintese)

    progress(0.9, desc="✅ Etapa 5/5: Validando e formatando...")
    # Adicionar disclaimer
    sintese["disclaimer"] = (
        "⚕️ ATENÇÃO: Esta resposta foi gerada por IA e constitui APENAS "
        "uma sugestão. A validação do médico assistente é obrigatória "
        "antes de qualquer conduta clínica."
    )
    # Adicionar info da triagem
    sintese["triagem"] = tri

    # Agente 3: Validação
    from src.agents.validacao import validar
    validated = validar(sintese, tri, _llm)
    log_event("validacao", agent="validacao", input_data=sintese, output_data=validated)

    progress(1.0, desc="✅ Concluído!")

    return tri, rag_pmc, rag_interno, validated


def finalizar_consulta(decisao: str, texto_editado: str, sintese_atual: dict,
                       nome: str, idade: str, sexo: str):
    """Processa decisão do HITL e gera PDFs reais com ReportLab."""
    inicializar_componentes()

    if decisao == "rejeitar":
        log_event("hitl_decision", action="rejeitado")
        return "❌ Consulta rejeitada. Nenhum documento gerado.", None, None, None

    log_event("hitl_decision", action=decisao, texto_editado=texto_editado)

    # Preparar dados pra ReportLab
    texto_final = texto_editado if (decisao == "editar" and texto_editado) else str(sintese_atual)

    dados_pdf = {
        "paciente": nome or "Não informado",
        "idade": idade or "—",
        "sexo": sexo or "—",
        "medico": "Dr(a). Responsável",
        "crm": "000000-DF",
        "queixa_principal": texto_final[:500] if texto_final else "",
        "exame_fisico": "Conforme avaliação clínica (ver prontuário anterior)",
        "hipoteses": sintese_atual.get("hipoteses", []),
        "exames_sugeridos": sintese_atual.get("exames_sugeridos", []),
        "medicacoes_sugeridas": sintese_atual.get("medicacoes_sugeridas", []),
        "observacoes": sintese_atual.get("observacoes", ""),
        "dias_afastamento": "7",
        "motivo": "condição clínica",
    }

    # Gerar PDFs REAIS
    prontuario = _doc_gen.gerar_prontuario(dados_pdf)
    atestado = _doc_gen.gerar_atestado(dados_pdf)
    receita = _doc_gen.gerar_receita(dados_pdf)
    log_event("document_generated", document_type="prontuario+atestado+receita")

    return (
        f"✅ Documentos PDF gerados com sucesso!",
        prontuario,
        atestado,
        receita,
    )


def log_event(event_type: str, **kwargs):
    """Grava evento no SQLite."""
    try:
        from src.logging.audit import init_db, log_event
        from src.logging.schemas import LLMCallEvent, HITLDecisionEvent, DocumentGeneratedEvent

        init_db()

        if event_type in ("triagem", "sintese", "validacao"):
            event = LLMCallEvent(
                event_type="llm_call",
                session_id=SESSION_ID,
                user_id=USER_ID,
                agent=kwargs.get("agent", event_type),
                input_preview=str(kwargs.get("input_data", ""))[:500],
                output_preview=str(kwargs.get("output_data", ""))[:500],
                tokens_in=len(str(kwargs.get("input_data", ""))) // 4,
                tokens_out=len(str(kwargs.get("output_data", ""))) // 4,
            )
        elif event_type == "hitl_decision":
            event = HITLDecisionEvent(
                event_type="hitl_decision",
                session_id=SESSION_ID,
                user_id=USER_ID,
                action=kwargs.get("action", ""),
                texto_original=str(kwargs.get("texto_editado", ""))[:500],
            )
        elif event_type == "document_generated":
            event = DocumentGeneratedEvent(
                event_type="document_generated",
                session_id=SESSION_ID,
                user_id=USER_ID,
                document_type=kwargs.get("document_type", ""),
                file_path="generated",
            )
        else:
            return

        log_event(event)
    except Exception as e:
        print(f"Erro ao logar: {e}")


# ============================================================
# DASHBOARD
# ============================================================
def get_audit_dashboard(horas: int = 24) -> tuple[str, str]:
    """Retorna texto do dashboard + métricas."""
    if not DB_PATH.exists():
        return "📊 Banco ainda não inicializado. Faça uma consulta primeiro.", "0"

    try:
        from src.logging.dashboard import dashboard_resumo
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            dashboard_resumo(horas=horas)

        conn = sqlite3.connect(str(DB_PATH))
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        sessoes = conn.execute("SELECT COUNT(DISTINCT session_id) FROM events").fetchone()[0]
        conn.close()

        return buf.getvalue(), f"**{total:,}** eventos | **{sessoes}** sessões"
    except Exception as e:
        return f"Erro: {e}", "0"


def listar_documentos() -> str:
    """Lista PDFs gerados."""
    if not DOCS_DIR.exists():
        return "Nenhum documento gerado ainda."

    docs = sorted(DOCS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not docs:
        return "Nenhum documento gerado ainda."

    md = "📄 **Documentos PDF gerados (mais recentes primeiro):**\n\n"
    for doc in docs[:20]:
        mtime = datetime.fromtimestamp(doc.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size_kb = doc.stat().st_size / 1024
        md += f"- 📄 `{doc.name}` ({size_kb:.1f} KB) — {mtime}\n"
    return md


# ============================================================
# UI GRADIO
# ============================================================
with gr.Blocks(
    title="🏥 Assistente Médico Inteligente",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
) as demo:

    gr.Markdown(f"""
    # 🏥 Tech Challenge Fase 3 — Assistente Médico Inteligente

    **Sessão**: `{SESSION_ID}` | **Usuário**: `{USER_ID}` | **Status**: 🟢 Online

    Pipeline: Relato → Triagem → RAG (PMC + Interno) → Síntese → HITL → PDFs
    """)

    sintese_state = gr.State({})

    with gr.Tabs():

        # ============================================================
        # ABA 1: CONSULTA
        # ============================================================
        with gr.Tab("📋 Consulta"):
            gr.Markdown("### 1️⃣ Dados do Paciente e Relato Inicial")

            with gr.Row():
                with gr.Column():
                    nome = gr.Textbox(label="👤 Nome do Paciente", placeholder="Maria Silva")
                with gr.Column():
                    idade = gr.Textbox(label="🎂 Idade", placeholder="45")
                with gr.Column():
                    sexo = gr.Radio(["Feminino", "Masculino", "Outro"], label="⚧ Sexo")

            relato = gr.Textbox(
                label="📝 Relato Clínico",
                placeholder="Ex: Paciente relata dor torácica em aperto há 3h, irradiando para braço esquerdo...",
                lines=5,
            )

            iniciar_btn = gr.Button("🚀 Iniciar Consulta", variant="primary", size="lg")
            gr.Markdown("---")

            gr.Markdown("### 2️⃣ Resultado do Pipeline")
            with gr.Row():
                with gr.Column():
                    triagem_out = gr.JSON(label="🚨 Triagem (Agente 1)")
                with gr.Column():
                    sintese_out = gr.JSON(label="🧠 Síntese (Agentes 2+3)")

            with gr.Row():
                with gr.Column():
                    rag_pmc_out = gr.JSON(label="📚 RAG Literatura (PMC)")
                with gr.Column():
                    rag_interno_out = gr.JSON(label="🏥 RAG Base Interna")

            gr.Markdown("### 3️⃣ Decisão do Médico (HITL)")
            with gr.Row():
                aprovar_btn = gr.Button("✅ Aprovar como está", variant="primary")
                editar_btn = gr.Button("✏️ Editar texto", variant="secondary")
                rejeitar_btn = gr.Button("❌ Rejeitar", variant="stop")

            texto_editado = gr.Textbox(
                label="📝 Texto editado (se clicou em 'Editar')",
                lines=5,
                visible=False,
            )

            status_hitl = gr.Markdown("")
            gr.Markdown("### 4️⃣ Documentos PDF Gerados")
            with gr.Row():
                prontuario_file = gr.File(label="📄 Prontuário", visible=False)
                atestado_file = gr.File(label="📄 Atestado", visible=False)
                receita_file = gr.File(label="📄 Receita", visible=False)

            # Eventos
            def iniciar(rel, n, i, s):
                tri, rpmc, rint, sint = processar_consulta(rel, n, i, s)
                return tri, rpmc, rint, sint, sint

            iniciar_btn.click(
                iniciar, inputs=[relato, nome, idade, sexo],
                outputs=[triagem_out, rag_pmc_out, rag_interno_out, sintese_out, sintese_state],
            )

            editar_btn.click(lambda: gr.update(visible=True), outputs=[texto_editado])

            def on_aprovar(s):
                return finalizar_consulta("aprovado", "", s, nome.value, idade.value, sexo.value)

            def on_editar(t, s):
                return finalizar_consulta("editar", t, s, nome.value, idade.value, sexo.value)

            def on_rejeitar(s):
                return finalizar_consulta("rejeitar", "", s, nome.value, idade.value, sexo.value)

            def show_files(p, a, r):
                return (
                    gr.update(value=p, visible=True) if p else gr.update(visible=False),
                    gr.update(value=a, visible=True) if a else gr.update(visible=False),
                    gr.update(value=r, visible=True) if r else gr.update(visible=False),
                )

            aprovar_btn.click(
                on_aprovar, inputs=[sintese_state],
                outputs=[status_hitl, prontuario_file, atestado_file, receita_file],
            )
            editar_btn.click(
                on_editar, inputs=[texto_editado, sintese_state],
                outputs=[status_hitl, prontuario_file, atestado_file, receita_file],
            )
            rejeitar_btn.click(
                on_rejeitar, inputs=[sintese_state],
                outputs=[status_hitl, prontuario_file, atestado_file, receita_file],
            )

        # ============================================================
        # ABA 2: AUDITORIA
        # ============================================================
        with gr.Tab("📊 Auditoria"):
            gr.Markdown("### Logs de Auditoria (SQLite)")
            with gr.Row():
                horas_slider = gr.Slider(1, 168, value=24, step=1, label="Período (horas)")
                refresh_btn = gr.Button("🔄 Atualizar")

            metricas_out = gr.Markdown("Carregando...")
            dashboard_out = gr.Textbox(label="Dashboard completo", lines=20, max_lines=30)

            def refresh_dash(h):
                dash, m = get_audit_dashboard(int(h))
                return m, dash

            refresh_btn.click(refresh_dash, inputs=[horas_slider], outputs=[metricas_out, dashboard_out])
            demo.load(refresh_dash, inputs=[horas_slider], outputs=[metricas_out, dashboard_out])

        # ============================================================
        # ABA 3: DOCUMENTOS
        # ============================================================
        with gr.Tab("📁 Documentos"):
            gr.Markdown("### PDFs Gerados")
            docs_out = gr.Markdown(listar_documentos())
            refresh_docs_btn = gr.Button("🔄 Atualizar Lista")
            refresh_docs_btn.click(lambda: listar_documentos(), outputs=[docs_out])

        # ============================================================
        # ABA 4: CONFIG
        # ============================================================
        with gr.Tab("⚙️ Config"):
            gr.Markdown(f"""
            ### Sistema

            | Item | Valor |
            |------|-------|
            | **LLM** | BioMistral-7B + LoRA (modo mock até treinar) |
            | **RAG** | ChromaDB: ANVISA (43k) + CID-10 (12k) + Synthetic (3k) |
            | **PDFs** | ReportLab (prontuário, atestado, receita, laudo) |
            | **Auditoria** | SQLite em `{DB_PATH}` |
            | **Sessão** | `{SESSION_ID}` |
            | **Diretório docs** | `{DOCS_DIR}` |

            **Stack**: LangChain + LangGraph + ChromaDB + ReportLab + SQLite + Gradio
            """)


# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("🏥 INICIANDO ASSISTENTE MÉDICO INTELIGENTE (VERSÃO FINAL)")
    print("="*60)

    # Tentar inicializar componentes
    try:
        inicializar_componentes()
        print("✅ Todos os componentes inicializados!")
    except Exception as e:
        print(f"⚠️  Erro na inicialização: {e}")
        print("   Continuando com fallbacks (modo demo)...")

    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        auth=("medico", "demo123"),
        show_error=True,
    )
    print("\n✅ Servidor rodando!")
    print("   Local:    http://127.0.0.1:7860")
    print("   Mobile:   http://<seu-IP>:7860")
    print("   Login:    medico / demo123")