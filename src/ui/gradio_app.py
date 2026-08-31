"""
Gradio App — Assistente Médico Inteligente (Tech Challenge Fase 3)
===================================================================

Interface completa com 4 abas:
1. Consulta (relato → triagem → RAG → síntese → HITL → documentos)
2. Auditoria (dashboard de logs SQLite)
3. Documentos (lista de PDFs gerados)
4. Configurações (parâmetros do sistema)

Uso:
    python src/ui/gradio_app.py

Acesso:
    - Local:    http://127.0.0.1:7860
    - Mobile:   http://<IP-do-PC>:7860 (mesma WiFi)
    - Público:  share=True gera URL gradio.live (válida 72h)

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
Data:  2026-08-31
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import gradio as gr

# ============================================================
# CONFIGURAÇÃO
# ============================================================
DB_PATH = Path("audit.db")
DOCS_DIR = Path("data/documents")
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# Sessão atual (em produção seria por usuário logado)
SESSION_ID = str(uuid.uuid4())[:8]
USER_ID = "dr.demonstracao"

# ============================================================
# DADOS MOCK (em produção, viria do LangGraph + RAG + LLM)
# ============================================================
# Estes são placeholders pra demonstração visual.
# Quando o LangGraph estiver conectado, substituir por chamadas reais.

def triar_mock(relato: str) -> dict:
    """Mock do Agente 1 — Triagem."""
    keywords_emergencia = ["dor torácica", "infarto", "avc", "sepse", "parada"]
    keywords_urgente = ["febre alta", "sangramento", "dispneia", "vômito"]

    relato_lower = relato.lower()
    if any(k in relato_lower for k in keywords_emergencia):
        cat = "EMERGENCIA"
    elif any(k in relato_lower for k in keywords_urgente):
        cat = "URGENTE"
    else:
        cat = "ROTINA"

    return {
        "categoria": cat,
        "justificativa": f"Detectado padrão compatível com {cat} baseado em palavras-chave",
        "red_flags": ["dor intensa"] if cat == "EMERGENCIA" else [],
        "confianca": "media",
    }


def retrieve_rag_mock(query: str, source: str) -> list[dict]:
    """Mock do RAG retrieval (em produção, viria do ChromaDB)."""
    if source == "pmc":
        return [
            {"source": "PMC-12345", "content": f"Literatura científica sobre '{query}' — artigo A mostra que..."},
            {"source": "PMC-67890", "content": f"Estudo clínico randomizado (n=500) sobre '{query}' indica..."},
        ]
    else:
        return [
            {"source": "SOP-CARDIO-007", "content": f"Protocolo interno: ao receber queixa de '{query}', proceder com..."},
            {"source": "SOP-EMERG-002", "content": f"Fluxograma institucional para '{query}': passo1: triagem, passo2:..."},
        ]


def sintetizar_mock(relato: str, rag_pmc: list, rag_interno: list) -> dict:
    """Mock do Agente 2 — Síntese."""
    tri = triar_mock(relato)
    return {
        "hipoteses": [
            {"cid10": "I21", "nome": "Infarto Agudo do Miocárdio", "probabilidade": "alta", "justificativa": "Padrão clínico compatível", "fonte": "PMC-12345"},
            {"cid10": "I20", "nome": "Angina Instável", "probabilidade": "media", "justificativa": "Diagnóstico diferencial", "fonte": "SOP-CARDIO-007"},
        ],
        "exames_sugeridos": [
            {"nome": "ECG de 12 derivações", "justificativa": "Investigar isquemia", "fonte": "PMC-12345"},
            {"nome": "Troponina I", "justificativa": "Marcador de necrose miocárdica", "fonte": "PMC-67890"},
        ],
        "medicacoes_sugeridas": [
            {"nome": "AAS", "dose": "200mg", "frequencia": "dose única", "NOTA": "⚠️ VALIDAÇÃO MÉDICA OBRIGATÓRIA"},
            {"nome": "Atorvastatina", "dose": "40mg", "frequencia": "1x/dia", "NOTA": "⚠️ VALIDAÇÃO MÉDICA OBRIGATÓRIA"},
        ],
        "observacoes": "Caso requer investigação adicional antes de conduta definitiva.",
        "triagem": tri,
    }


# ============================================================
# PIPELINE COMPLETO (mocks conectados)
# ============================================================
def processar_consulta(relato: str, nome_paciente: str, idade: str, sexo: str, progress=gr.Progress()):
    """Pipeline completo: triagem → RAG → síntese → validação."""
    progress(0.1, desc="🔍 Iniciando triagem...")
    tri = triar_mock(relato)

    progress(0.3, desc="📚 Buscando literatura científica (PMC)...")
    rag_pmc = retrieve_rag_mock(relato, "pmc")

    progress(0.5, desc="🏥 Buscando protocolos internos...")
    rag_interno = retrieve_rag_mock(relato, "interno")

    progress(0.7, desc="🧠 Gerando síntese clínica...")
    sintese = sintetizar_mock(relato, rag_pmc, rag_interno)

    progress(0.9, desc="✅ Validando saída...")
    # Aplicar disclaimer
    sintese["disclaimer"] = (
        "⚕️ ATENÇÃO: Esta resposta foi gerada por IA e constitui APENAS "
        "uma sugestão. A validação do médico assistente é obrigatória "
        "antes de qualquer conduta clínica."
    )

    # Log do evento
    log_event_mock("llm_call", agent="sintese", input_data=relato, output_data=str(sintese))

    progress(1.0, desc="✅ Concluído!")
    return tri, rag_pmc, rag_interno, sintese


def finalizar_consulta(decisao: str, texto_editado: str, sintese_atual: dict):
    """Processa decisão do HITL e gera documentos."""
    if decisao == "rejeitar":
        log_event_mock("hitl_decision", action="rejeitado")
        return "❌ Consulta rejeitada. Nenhum documento gerado.", None, None, None

    log_event_mock("hitl_decision", action=decisao, texto_editado=texto_editado)

    # Gerar PDFs (mock — em produção usa ReportLab)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prontuario = DOCS_DIR / f"prontuario_{timestamp}.txt"
    atestado = DOCS_DIR / f"atestado_{timestamp}.txt"
    receita = DOCS_DIR / f"receita_{timestamp}.txt"

    texto_final = texto_editado if (decisao == "editar" and texto_editado) else str(sintese_atual)

    prontuario.write_text(f"PRONTUÁRIO MÉDICO\n{'='*40}\n{texto_final}", encoding="utf-8")
    atestado.write_text(f"ATESTADO MÉDICO\n{'='*40}\nAtesto para os devidos fins...", encoding="utf-8")
    receita.write_text(f"RECEITA MÉDICA\n{'='*40}\n{texto_final}", encoding="utf-8")

    log_event_mock("document_generated", document_type="prontuario+atestado+receita")

    return (
        f"✅ Consulta {decisao.upper()}! Documentos gerados:",
        str(prontuario),
        str(atestado),
        str(receita),
    )


# ============================================================
# LOGGING MOCK (usa o sistema de auditoria real)
# ============================================================
def log_event_mock(event_type: str, **kwargs):
    """Grava evento no SQLite de auditoria."""
    try:
        from src.logging.audit import init_db, log_event
        from src.logging.schemas import LLMCallEvent, HITLDecisionEvent, DocumentGeneratedEvent

        init_db()

        if event_type == "llm_call":
            event = LLMCallEvent(
                event_type=event_type,
                session_id=SESSION_ID,
                user_id=USER_ID,
                agent=kwargs.get("agent", ""),
                input_preview=str(kwargs.get("input_data", ""))[:500],
                output_preview=str(kwargs.get("output_data", ""))[:500],
                tokens_in=len(str(kwargs.get("input_data", ""))) // 4,
                tokens_out=len(str(kwargs.get("output_data", ""))) // 4,
            )
        elif event_type == "hitl_decision":
            event = HITLDecisionEvent(
                event_type=event_type,
                session_id=SESSION_ID,
                user_id=USER_ID,
                action=kwargs.get("action", ""),
                texto_original=str(kwargs.get("texto_editado", "")),
            )
        elif event_type == "document_generated":
            event = DocumentGeneratedEvent(
                event_type=event_type,
                session_id=SESSION_ID,
                user_id=USER_ID,
                document_type=kwargs.get("document_type", ""),
                file_path="mock",
            )
        else:
            return

        log_event(event)
    except Exception as e:
        print(f"Erro ao logar: {e}")


# ============================================================
# DASHBOARD DE AUDITORIA
# ============================================================
def get_audit_dashboard(horas: int = 24) -> tuple[str, str]:
    """Retorna texto do dashboard + métricas."""
    if not DB_PATH.exists():
        return "📊 Banco de auditoria ainda não inicializado. Faça uma consulta primeiro.", "0"

    try:
        from src.logging.dashboard import dashboard_resumo
        import io
        from contextlib import redirect_stdout

        # Capturar output do dashboard
        buf = io.StringIO()
        with redirect_stdout(buf):
            try:
                dashboard_resumo(horas=horas)
            except Exception as e:
                buf.write(f"Erro: {e}")

        # Métricas rápidas
        conn = sqlite3.connect(str(DB_PATH))
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        sessoes = conn.execute("SELECT COUNT(DISTINCT session_id) FROM events").fetchone()[0]
        conn.close()

        return buf.getvalue(), f"**{total:,}** eventos | **{sessoes}** sessões"
    except Exception as e:
        return f"Erro ao carregar dashboard: {e}", "0"


# ============================================================
# LISTA DE DOCUMENTOS
# ============================================================
def listar_documentos() -> str:
    """Lista PDFs gerados."""
    if not DOCS_DIR.exists():
        return "Nenhum documento gerado ainda."

    docs = sorted(DOCS_DIR.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not docs:
        return "Nenhum documento gerado ainda."

    md = "📄 **Documentos gerados (mais recentes primeiro):**\n\n"
    for doc in docs[:20]:  # últimos 20
        mtime = datetime.fromtimestamp(doc.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        size_kb = doc.stat().st_size / 1024
        md += f"- `{doc.name}` ({size_kb:.1f} KB) — {mtime}\n"
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

    Pipeline: Relato → Triagem → RAG (PMC + Interno) → Síntese → HITL → Documentos
    """)

    # Estado compartilhado entre componentes
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
                label="📝 Relato Clínico (queixa + sintomas)",
                placeholder="Ex: Paciente relata dor torácica em aperto há 3 horas, irradiando para braço esquerdo, com sudorese fria...",
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

            gr.Markdown("### 3️⃣ Decisão do Médico (HITL — Obrigatório)")

            with gr.Row():
                aprovar_btn = gr.Button("✅ Aprovar como está", variant="primary")
                editar_btn = gr.Button("✏️ Editar texto", variant="secondary")
                rejeitar_btn = gr.Button("❌ Rejeitar", variant="stop")

            texto_editado = gr.Textbox(
                label="📝 Texto editado (preencha se clicou em 'Editar')",
                lines=5,
                visible=False,
            )

            status_hitl = gr.Markdown("")

            gr.Markdown("### 4️⃣ Documentos Gerados")
            with gr.Row():
                prontuario_file = gr.File(label="📄 Prontuário", visible=False)
                atestado_file = gr.File(label="📄 Atestado", visible=False)
                receita_file = gr.File(label="📄 Receita", visible=False)

            # Eventos
            def iniciar_e_mostrar(relato_txt, nome_pac, idade_pac, sexo_pac):
                tri, rag_pmc, rag_int, sintese = processar_consulta(relato_txt, nome_pac, idade_pac, sexo_pac)
                return tri, rag_pmc, rag_int, sintese, sintese

            iniciar_btn.click(
                iniciar_e_mostrar,
                inputs=[relato, nome, idade, sexo],
                outputs=[triagem_out, rag_pmc_out, rag_interno_out, sintese_out, sintese_state],
            )

            def on_editar():
                return gr.update(visible=True)

            editar_btn.click(on_editar, outputs=[texto_editado])

            def on_finalizar(decisao, texto_ed, sintese_atual):
                if decisao == "aprovar":
                    status, p, a, r = finalizar_consulta("aprovado", "", sintese_atual)
                elif decisao == "editar":
                    status, p, a, r = finalizar_consulta("editar", texto_ed, sintese_atual)
                else:
                    status, p, a, r = finalizar_consulta("rejeitar", "", sintese_atual)

                # Mostrar arquivos
                return (
                    status,
                    gr.update(value=p, visible=True) if p else gr.update(visible=False),
                    gr.update(value=a, visible=True) if a else gr.update(visible=False),
                    gr.update(value=r, visible=True) if r else gr.update(visible=False),
                )

            aprovar_btn.click(
                lambda s: on_finalizar("aprovar", "", s),
                inputs=[sintese_state],
                outputs=[status_hitl, prontuario_file, atestado_file, receita_file],
            )

            editar_btn.click(
                lambda texto, s: on_finalizar("editar", texto, s),
                inputs=[texto_editado, sintese_state],
                outputs=[status_hitl, prontuario_file, atestado_file, receita_file],
            )

            rejeitar_btn.click(
                lambda s: on_finalizar("rejeitar", "", s),
                inputs=[sintese_state],
                outputs=[status_hitl, prontuario_file, atestado_file, receita_file],
            )

        # ============================================================
        # ABA 2: AUDITORIA
        # ============================================================
        with gr.Tab("📊 Auditoria"):
            gr.Markdown("### Logs de Auditoria do Sistema")
            gr.Markdown("Todas as chamadas LLM, RAG e decisões humanas são registradas em SQLite.")

            with gr.Row():
                horas_slider = gr.Slider(1, 168, value=24, step=1, label="Período (horas)")
                refresh_audit_btn = gr.Button("🔄 Atualizar")

            metricas_out = gr.Markdown("Carregando...")
            dashboard_out = gr.Textbox(label="Dashboard completo", lines=20, max_lines=30)

            def refresh_dashboard(horas):
                dash, metricas = get_audit_dashboard(int(horas))
                return metricas, dash

            refresh_audit_btn.click(refresh_dashboard, inputs=[horas_slider], outputs=[metricas_out, dashboard_out])
            # Carregar inicial
            demo.load(refresh_dashboard, inputs=[horas_slider], outputs=[metricas_out, dashboard_out])

        # ============================================================
        # ABA 3: DOCUMENTOS
        # ============================================================
        with gr.Tab("📁 Documentos"):
            gr.Markdown("### Documentos Gerados")
            docs_list_out = gr.Markdown(listar_documentos())
            refresh_docs_btn = gr.Button("🔄 Atualizar Lista")
            refresh_docs_btn.click(lambda: listar_documentos(), outputs=[docs_list_out])

        # ============================================================
        # ABA 4: CONFIGURAÇÕES
        # ============================================================
        with gr.Tab("⚙️ Config"):
            gr.Markdown("### Informações do Sistema")
            gr.Markdown(f"""
            | Item | Valor |
            |------|-------|
            | Modelo | BioMistral-7B (QLoRA) |
            | RAG #1 | ANVISA + CID-10 + Synthetic (ChromaDB) |
            | RAG #2 | Literatura PMC (a indexar) |
            | Sessão | {SESSION_ID} |
            | Diretório docs | {DOCS_DIR} |
            | Banco auditoria | {DB_PATH} |
            | GitHub | https://github.com/Flamers-Team/Techchalleng3 |

            **Stack**: LangChain + LangGraph + ChromaDB + SQLite + Gradio
            """)

# ============================================================
# LAUNCH
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("🏥 INICIANDO ASSISTENTE MÉDICO INTELIGENTE")
    print("="*60)
    print(f"📂 Documentos: {DOCS_DIR.absolute()}")
    print(f"📊 Auditoria:  {DB_PATH.absolute()}")
    print()
    demo.launch(
        share=False,  # Mude pra True pra gerar URL pública
        server_name="0.0.0.0",  # Acessível pela rede local (celular na mesma WiFi)
        server_port=7860,
        auth=("medico", "demo123"),  # Auth básica (remova em produção)
        show_error=True,
    )
    print("\n✅ Servidor rodando!")
    print("   Local:    http://127.0.0.1:7860")
    print("   Mobile:   http://<seu-IP>:7860")
    print("   Público:  ative share=True pra URL gradio.live")