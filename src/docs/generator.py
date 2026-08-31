"""
Gerador de PDFs médicos com ReportLab.

Cria:
- Prontuário (formato SOAP)
- Atestado
- Receita médica
- Laudo (estrutura genérica)

Uso:
    from src.docs.generator import DocumentGenerator
    gen = DocumentGenerator()
    path = gen.gerar_prontuario({
        "paciente": "Maria Silva",
        "idade": 45,
        "sexo": "F",
        "queixa": "...",
        "hipoteses": [...],
        "exames": [...],
        "conduta": "...",
        "medico": "Dr. João",
        "crm": "12345-DF",
    })

Autor: Michelle Nogueira (Tech Challenge FIAP - Fase 3)
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.colors import HexColor, black, grey, white
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether,
)
from reportlab.pdfgen import canvas
from datetime import datetime
from pathlib import Path
import hashlib


# ============================================================
# ESTILOS
# ============================================================
def get_styles():
    """Retorna estilos customizados para documentos médicos."""
    styles = getSampleStyleSheet()

    # Título principal
    styles.add(ParagraphStyle(
        name="DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=HexColor("#1F4E79"),
        alignment=TA_CENTER,
        spaceAfter=12,
        spaceBefore=0,
    ))

    # Subtítulo
    styles.add(ParagraphStyle(
        name="DocSubtitle",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=HexColor("#2E74B5"),
        alignment=TA_LEFT,
        spaceAfter=8,
        spaceBefore=12,
    ))

    # Disclaimer (caixa amarela)
    styles.add(ParagraphStyle(
        name="Disclaimer",
        parent=styles["Normal"],
        fontSize=9,
        textColor=HexColor("#8B0000"),
        backColor=HexColor("#FFFCE4"),
        borderColor=HexColor("#FFA500"),
        borderWidth=1,
        borderPadding=6,
        leftIndent=10,
        rightIndent=10,
        spaceBefore=6,
        spaceAfter=12,
        alignment=TA_CENTER,
    ))

    # Texto normal
    styles.add(ParagraphStyle(
        name="DocBody",
        parent=styles["Normal"],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    ))

    # Lista
    styles.add(ParagraphStyle(
        name="DocList",
        parent=styles["Normal"],
        fontSize=10,
        leftIndent=20,
        bulletIndent=10,
        spaceAfter=3,
    ))

    return styles


# ============================================================
# GERADOR
# ============================================================
class DocumentGenerator:
    """Gera PDFs médicos: prontuário, atestado, receita, laudo."""

    def __init__(self, output_dir: str = "data/documents"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.styles = get_styles()

    def _criar_doc(self, filename: str, title: str) -> SimpleDocTemplate:
        """Cria documento base com cabeçalho e rodapé customizados."""
        path = self.output_dir / filename

        doc = SimpleDocTemplate(
            str(path),
            pagesize=A4,
            leftMargin=2*cm,
            rightMargin=2*cm,
            topMargin=2.5*cm,
            bottomMargin=2.5*cm,
            title=title,
            author="Assistente Médico - Tech Challenge FIAP",
        )

        # Adicionar header/footer
        def header_footer(canvas_obj, doc):
            canvas_obj.saveState()
            # Header
            canvas_obj.setFont("Helvetica-Bold", 9)
            canvas_obj.setFillColor(HexColor("#1F4E79"))
            canvas_obj.drawString(2*cm, A4[1] - 1.5*cm, "🏥 Assistente Médico - Tech Challenge FIAP")
            canvas_obj.line(2*cm, A4[1] - 1.7*cm, A4[0] - 2*cm, A4[1] - 1.7*cm)
            # Footer
            canvas_obj.setFont("Helvetica", 8)
            canvas_obj.setFillColor(grey)
            canvas_obj.drawString(
                2*cm, 1.5*cm,
                f"Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | Doc ID: {filename[:8]}"
            )
            canvas_obj.drawRightString(
                A4[0] - 2*cm, 1.5*cm,
                f"Página {doc.page}"
            )
            canvas_obj.restoreState()

        doc.header_footer = header_footer
        # Hack: SimpleDocTemplate não tem header_footer nativamente,
        # então vamos usar onLaterPages + onFirstPage
        doc.onFirstPage = header_footer
        doc.onLaterPages = header_footer

        return doc

    def _add_disclaimer(self, story, texto=None):
        """Adiciona caixa de disclaimer amarelo."""
        if texto is None:
            texto = (
                "⚕️ ATENÇÃO: Este documento foi gerado por IA e constitui "
                "uma SUGESTÃO. A validação e assinatura do médico assistente "
                "são obrigatórias antes de qualquer uso clínico."
            )
        story.append(Paragraph(texto, self.styles["Disclaimer"]))
        story.append(Spacer(1, 0.3*cm))

    def _add_metadata(self, story, dados):
        """Adiciona tabela de metadados do paciente."""
        rows = [
            ["Paciente:", dados.get("paciente", ""), "Idade:", str(dados.get("idade", ""))],
            ["Sexo:", dados.get("sexo", ""), "Data:", datetime.now().strftime("%d/%m/%Y")],
            ["Médico:", dados.get("medico", ""), "CRM:", dados.get("crm", "")],
        ]

        tbl = Table(rows, colWidths=[3*cm, 6*cm, 3*cm, 6*cm])
        tbl.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
            ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 10),
            ("BACKGROUND", (0, 0), (0, -1), HexColor("#F0F0F0")),
            ("BACKGROUND", (2, 0), (2, -1), HexColor("#F0F0F0")),
            ("GRID", (0, 0), (-1, -1), 0.5, grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.5*cm))

    def _add_signature_line(self, story):
        """Adiciona linha de assinatura no final do documento."""
        story.append(Spacer(1, 1.5*cm))
        story.append(Paragraph(
            "_____________________________________________",
            self.styles["DocBody"]
        ))
        story.append(Paragraph(
            "<b>Assinatura do Médico</b>",
            self.styles["DocBody"]
        ))
        story.append(Spacer(1, 0.5*cm))

    def _calcular_hash(self, texto: str) -> str:
        """Calcula SHA256 do conteúdo."""
        return hashlib.sha256(texto.encode()).hexdigest()

    # ============================================================
    # PRONTUÁRIO (formato SOAP)
    # ============================================================
    def gerar_prontuario(self, dados: dict) -> str:
        """Gera prontuário médico em PDF."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prontuario_{timestamp}.pdf"
        doc = self._criar_doc(filename, "Prontuário Médico")

        story = []

        # Título
        story.append(Paragraph("PRONTUÁRIO MÉDICO", self.styles["DocTitle"]))
        story.append(Paragraph(
            f"Documento gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}",
            self.styles["DocBody"]
        ))

        # Disclaimer
        self._add_disclaimer(story)

        # Metadados
        self._add_metadata(story, dados)

        # S - Subjetivo (queixa + história)
        story.append(Paragraph("S — SUBJETIVO (Queixa e História)", self.styles["DocSubtitle"]))
        story.append(Paragraph(
            dados.get("queixa_principal", "Não informado").replace("\n", "<br/>"),
            self.styles["DocBody"]
        ))

        # O - Objetivo (exame físico)
        story.append(Paragraph("O — OBJETIVO (Exame Físico)", self.styles["DocSubtitle"]))
        story.append(Paragraph(
            dados.get("exame_fisico", "Não informado").replace("\n", "<br/>"),
            self.styles["DocBody"]
        ))

        # A - Avaliação (hipóteses diagnósticas)
        story.append(Paragraph("A — AVALIAÇÃO (Hipóteses Diagnósticas)", self.styles["DocSubtitle"]))
        hipoteses = dados.get("hipoteses", [])
        if hipoteses:
            for h in hipoteses:
                cid = h.get("cid10", "—")
                nome = h.get("nome", "—")
                prob = h.get("probabilidade", "—")
                fonte = h.get("fonte", "—")
                story.append(Paragraph(
                    f"• <b>{nome}</b> (CID-10: {cid}) — Probabilidade: {prob} | Fonte: {fonte}",
                    self.styles["DocList"]
                ))
        else:
            story.append(Paragraph("Nenhuma hipótese gerada.", self.styles["DocBody"]))

        # P - Plano (exames + conduta)
        story.append(Paragraph("P — PLANO (Conduta e Exames)", self.styles["DocSubtitle"]))

        story.append(Paragraph("<b>Exames complementares sugeridos:</b>", self.styles["DocBody"]))
        exames = dados.get("exames_sugeridos", [])
        if exames:
            for e in exames:
                story.append(Paragraph(
                    f"• {e.get('nome', '')} — {e.get('justificativa', '')}",
                    self.styles["DocList"]
                ))
        else:
            story.append(Paragraph("Nenhum exame sugerido.", self.styles["DocBody"]))

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("<b>Medicações sugeridas (VALIDAÇÃO OBRIGATÓRIA):</b>", self.styles["DocBody"]))
        meds = dados.get("medicacoes_sugeridas", [])
        if meds:
            for m in meds:
                story.append(Paragraph(
                    f"• {m.get('nome', '')} {m.get('dose', '')} — {m.get('frequencia', '')} <i>({m.get('NOTA', '')})</i>",
                    self.styles["DocList"]
                ))
        else:
            story.append(Paragraph("Nenhuma medicação sugerida.", self.styles["DocBody"]))

        # Observações
        if dados.get("observacoes"):
            story.append(Paragraph("OBSERVAÇÕES", self.styles["DocSubtitle"]))
            story.append(Paragraph(dados["observacoes"], self.styles["DocBody"]))

        # Assinatura
        self._add_signature_line(story)

        # Gerar PDF
        doc.build(story, onFirstPage=doc.header_footer if hasattr(doc, 'header_footer') else None)
        return str(self.output_dir / filename)

    # ============================================================
    # ATESTADO
    # ============================================================
    def gerar_atestado(self, dados: dict) -> str:
        """Gera atestado médico em PDF."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"atestado_{timestamp}.pdf"
        doc = self._criar_doc(filename, "Atestado Médico")

        story = []

        story.append(Paragraph("ATESTADO MÉDICO", self.styles["DocTitle"]))
        story.append(Spacer(1, 0.5*cm))

        self._add_metadata(story, dados)

        story.append(Paragraph(
            "Atesto, para os devidos fins, que o(a) paciente acima identificado(a) "
            f"necessita de afastamento de suas atividades por <b>{dados.get('dias_afastamento', '___')} dias</b>, "
            f"a partir desta data, por motivo de <b>{dados.get('motivo', 'tratamento de saúde')}</b>.",
            self.styles["DocBody"]
        ))

        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            f"Local e data: ____________________, {datetime.now().strftime('%d de %B de %Y')}",
            self.styles["DocBody"]
        ))

        self._add_signature_line(story)

        doc.build(story, onFirstPage=doc.header_footer if hasattr(doc, 'header_footer') else None)
        return str(self.output_dir / filename)

    # ============================================================
    # RECEITA MÉDICA
    # ============================================================
    def gerar_receita(self, dados: dict) -> str:
        """Gera receita médica em PDF."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"receita_{timestamp}.pdf"
        doc = self._criar_doc(filename, "Receita Médica")

        story = []

        story.append(Paragraph("RECEITA MÉDICA", self.styles["DocTitle"]))
        story.append(Spacer(1, 0.5*cm))

        self._add_disclaimer(story, (
            "⚕️ PRESCRIÇÃO: Todas as medicações abaixo foram SUGERIDAS por IA. "
            "Médico deve validar antes de assinar."
        ))

        self._add_metadata(story, dados)

        story.append(Paragraph("<b>Prescrição:</b>", self.styles["DocBody"]))
        story.append(Spacer(1, 0.3*cm))

        meds = dados.get("medicacoes_sugeridas", [])
        if meds:
            for i, m in enumerate(meds, 1):
                story.append(Paragraph(
                    f"<b>{i}. {m.get('nome', '')} {m.get('dose', '')}</b>",
                    self.styles["DocBody"]
                ))
                story.append(Paragraph(
                    f"&nbsp;&nbsp;&nbsp;&nbsp;{m.get('frequencia', '')}",
                    self.styles["DocBody"]
                ))
                if m.get("NOTA"):
                    story.append(Paragraph(
                        f"&nbsp;&nbsp;&nbsp;&nbsp;<i>{m.get('NOTA', '')}</i>",
                        self.styles["DocBody"]
                    ))
                story.append(Spacer(1, 0.4*cm))
        else:
            story.append(Paragraph("Nenhuma medicação prescrita.", self.styles["DocBody"]))

        story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph(
            f"Validade: 30 dias a partir de {datetime.now().strftime('%d/%m/%Y')}",
            self.styles["DocBody"]
        ))

        self._add_signature_line(story)

        doc.build(story, onFirstPage=doc.header_footer if hasattr(doc, 'header_footer') else None)
        return str(self.output_dir / filename)

    # ============================================================
    # LAUDO (genérico)
    # ============================================================
    def gerar_laudo(self, dados: dict, tipo_laudo: str = "Laudo Médico") -> str:
        """Gera laudo médico genérico em PDF."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"laudo_{tipo_laudo.lower().replace(' ', '_')}_{timestamp}.pdf"
        doc = self._criar_doc(filename, tipo_laudo)

        story = []

        story.append(Paragraph(tipo_laudo.upper(), self.styles["DocTitle"]))
        story.append(Spacer(1, 0.5*cm))

        self._add_disclaimer(story)
        self._add_metadata(story, dados)

        story.append(Paragraph("DESCRIÇÃO", self.styles["DocSubtitle"]))
        story.append(Paragraph(
            dados.get("descricao", "Não informado").replace("\n", "<br/>"),
            self.styles["DocBody"]
        ))

        if dados.get("conclusao"):
            story.append(Paragraph("CONCLUSÃO", self.styles["DocSubtitle"]))
            story.append(Paragraph(dados["conclusao"], self.styles["DocBody"]))

        self._add_signature_line(story)

        doc.build(story, onFirstPage=doc.header_footer if hasattr(doc, 'header_footer') else None)
        return str(self.output_dir / filename)


# ============================================================
# TESTE / EXEMPLO
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("📄 TESTANDO GERADOR DE PDFs")
    print("="*60)

    gen = DocumentGenerator(output_dir="data/documents")

    # Dados de exemplo
    dados_exemplo = {
        "paciente": "Maria Silva",
        "idade": 45,
        "sexo": "Feminino",
        "medico": "Dr. João Santos",
        "crm": "12345-DF",
        "queixa_principal": "Paciente relata dor torácica em aperto há 3 horas, com irradiação para braço esquerdo.",
        "exame_fisico": "PA 140/90 mmHg, FC 88 bpm, SatO2 96%, ausculta cardíaca normal.",
        "hipoteses": [
            {"cid10": "I21", "nome": "Infarto Agudo do Miocárdio", "probabilidade": "alta", "fonte": "PMC-12345"},
            {"cid10": "I20", "nome": "Angina Instável", "probabilidade": "média", "fonte": "SOP-CARDIO-007"},
        ],
        "exames_sugeridos": [
            {"nome": "ECG 12 derivações", "justificativa": "Investigar isquemia"},
            {"nome": "Troponina I", "justificativa": "Marcador de necrose miocárdica"},
        ],
        "medicacoes_sugeridas": [
            {"nome": "AAS", "dose": "200mg", "frequencia": "dose única VO", "NOTA": "VALIDAÇÃO MÉDICA OBRIGATÓRIA"},
            {"nome": "Atorvastatina", "dose": "40mg", "frequencia": "1x/dia VO", "NOTA": "VALIDAÇÃO MÉDICA OBRIGATÓRIA"},
        ],
        "observacoes": "Caso requer investigação adicional antes de conduta definitiva.",
        "dias_afastamento": "7",
        "motivo": "investigação cardiológica",
    }

    p1 = gen.gerar_prontuario(dados_exemplo)
    print(f"✅ Prontuário: {p1}")

    p2 = gen.gerar_atestado(dados_exemplo)
    print(f"✅ Atestado:   {p2}")

    p3 = gen.gerar_receita(dados_exemplo)
    print(f"✅ Receita:    {p3}")

    p4 = gen.gerar_laudo({**dados_exemplo, "descricao": "Laudo de exame cardiológico.", "conclusao": "Sugere-se investigação adicional."})
    print(f"✅ Laudo:      {p4}")

    print(f"\n📁 Arquivos em: {gen.output_dir}")
    print("="*60)