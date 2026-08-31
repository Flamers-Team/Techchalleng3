"""Consultas SQL para visualizar auditoria."""
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta


def dashboard_resumo(db_path: Path = Path("audit.db"), horas: int = 24):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    since = (datetime.utcnow() - timedelta(hours=horas)).isoformat()

    print("="*70)
    print(f"📊 DASHBOARD DE AUDITORIA (últimas {horas}h)")
    print("="*70)

    total = conn.execute(
        "SELECT COUNT(*) FROM events WHERE timestamp >= ?", (since,)
    ).fetchone()[0]
    print(f"\n📈 Total de eventos: {total:,}")

    print(f"\n📋 Por tipo de evento:")
    for row in conn.execute("""
        SELECT event_type, COUNT(*) as n FROM events
        WHERE timestamp >= ? GROUP BY event_type ORDER BY n DESC
    """, (since,)):
        print(f"   • {row['event_type']:<25} {row['n']:>5,}")

    print(f"\n🤖 Por agente LLM:")
    for row in conn.execute("""
        SELECT agent, COUNT(*) as n, AVG(latency_ms) as avg_lat
        FROM events WHERE timestamp >= ? AND event_type = 'llm_call'
        GROUP BY agent
    """, (since,)):
        print(f"   • {row['agent'] or '(none)':<25} {row['n']:>5,} chamadas, {row['avg_lat']:.0f}ms")

    sessoes = conn.execute("""
        SELECT COUNT(DISTINCT session_id) FROM events WHERE timestamp >= ?
    """, (since,)).fetchone()[0]
    print(f"\n🔄 Sessões ativas: {sessoes:,}")

    custo = conn.execute("""
        SELECT SUM(cost_usd) FROM events WHERE timestamp >= ?
    """, (since,)).fetchone()[0] or 0
    print(f"💰 Custo estimado: ${custo:.4f}")
    conn.close()


def buscar_por_sessao(session_id: str, db_path: Path = Path("audit.db")):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    print("="*70)
    print(f"🔍 EVENTOS DA SESSÃO: {session_id}")
    print("="*70)

    for row in conn.execute("""
        SELECT * FROM events WHERE session_id = ? ORDER BY timestamp ASC
    """, (session_id,)):
        print(f"\n[{row['timestamp']}] {row['event_type']}")
        if row['agent']:
            print(f"   Agente: {row['agent']}")
        if row['latency_ms']:
            print(f"   Latência: {row['latency_ms']}ms")
        if row['input_preview']:
            print(f"   Input: {row['input_preview'][:200]}")
    conn.close()