"""The review flow and the read-only library queries as MCP tools over stdio.

An MCP client starts this server with the same parameters as the page: the
source library, a data directory for the workpapers and the API cache, and the
model with its budget.  ``check_draft`` runs the LangGraph review flow;
``find_number``, ``search_pages`` and ``page_text`` are the same scoped queries
the tool agent uses, and ``page_text`` refuses a document disclosed after the
cutoff date the caller passes.

    python -m src.service.mcp_server --library <library.sqlite> --data-dir <dir> \
        --model deepseek-flash --budget-usd 0.5

``--model`` on ``--url`` extracts the numbers; ``--agent-model`` investigates
and revises.  When the two differ the agent runs on DeepSeek with its own
``api_cache_deepseek-flash.jsonl``, so the local extractor hands the review to
flash; when they match, every request is served and cached exactly as before.
"""

import argparse
import json
from pathlib import Path
import sqlite3

from mcp.server.fastmcp import FastMCP

from src.agent.graph import review_draft
from src.agent.single import REFUSED
from src.data.library import Library

DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
AGENT_CACHE = 'api_cache_deepseek-flash.jsonl'


def build_server(*, library: Path, data_dir: Path, model: str, url: str,
                 budget_usd: float, agent_model: str = 'deepseek-flash') -> FastMCP:
    """The MCP server of one library, model, agent model and budget."""
    cache = data_dir / 'api_cache.jsonl'
    agent = {} if agent_model == model else {
        'agent_model': agent_model, 'agent_url': DEEPSEEK_URL,
        'agent_cache': data_dir / AGENT_CACHE}
    mcp = FastMCP('finagent-review')

    def company_of(code: str, fiscal_year: int) -> str | None:
        db = sqlite3.connect(f'file:{library}?mode=ro', uri=True)
        try:
            row = db.execute(
                "SELECT publisher FROM documents WHERE kind = 'annual' AND code = ?"
                ' AND fiscal_year = ? ORDER BY published DESC LIMIT 1',
                (code, fiscal_year)).fetchone()
        finally:
            db.close()
        return row[0] if row else None

    @mcp.tool()
    def check_draft(code: str, fiscal_year: int, as_of: str, text: str) -> dict:
        """核查一篇点评稿件，返回逐数字的工作底稿。"""
        company = company_of(code, fiscal_year)
        if company is None:
            return {'error': f'出处库中没有发行人 {code} 的 {fiscal_year} 年年度报告'}
        draft = {'text': text, 'code': code, 'company': company,
                 'fiscal_year': fiscal_year, 'as_of': as_of}
        return review_draft(draft, library=library, model=model, url=url,
                            budget_usd=budget_usd, cache=cache, **agent)

    @mcp.tool()
    def find_number(text: str, code: str, as_of: str) -> str:
        """按书写形式精确查找表格事实（本发行人，截止日前披露），返回 JSON 数组。"""
        facts = Library(library).find_number(text, code=code, as_of=as_of)[:8]
        return json.dumps(facts, ensure_ascii=False)

    @mcp.tool()
    def search_pages(query: str, code: str, as_of: str, limit: int = 5) -> str:
        """按关键词检索页面（本发行人，截止日前披露），返回 JSON 数组。"""
        pages = Library(library).search_pages(query, code=code, as_of=as_of,
                                              limit=limit)
        return json.dumps(pages, ensure_ascii=False)

    @mcp.tool()
    def page_text(document_id: str, page: int, as_of: str) -> dict:
        """读某份文件某一页的原文；截止日后披露的文件拒绝作为依据。"""
        books = Library(library)
        row = books.db.execute('SELECT published FROM documents WHERE id = ?',
                               (document_id,)).fetchone()
        if row is None:
            return {'error': '没有这份文件。'}
        if row['published'] > as_of:
            return {'error': REFUSED}
        try:
            text = books.page_text(document_id, page)
        except KeyError:
            return {'error': '这份文件没有这一页。'}
        return {'document_id': document_id, 'page': page, 'text': text}

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True,
                        help='workpapers and the API cache')
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--url', default=DEEPSEEK_URL)
    parser.add_argument('--agent-model', default='deepseek-flash',
                        help='the model that investigates and revises; the local '
                             'extractor of --model hands the review to it')
    parser.add_argument('--budget-usd', type=float, default=0.5)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    build_server(library=args.library, data_dir=args.data_dir, model=args.model,
                 url=args.url, budget_usd=args.budget_usd,
                 agent_model=args.agent_model).run()


if __name__ == '__main__':
    main()
