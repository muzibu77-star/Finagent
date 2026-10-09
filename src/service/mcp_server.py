"""Three read-only MCP tools, with immutable server-side research scope."""

import argparse
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from src.data.corpus import Corpus
from src.data.report_corpus import ReportCorpus
from src.tools.benchmark_tools import execute_action, extract_numeric_facts


class ResearchTools:
    def __init__(self, corpus, scope: dict):
        self.corpus, self.scope = corpus, dict(scope)
        if not scope.get('company') or not scope.get('period'):
            raise ValueError('issuer and period are required')
        self.allowed = corpus.allowed(scope)
        self.facts = {}
        self.read_ids = set()

    def search(self, query: str) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError('query must contain 1 to 2000 characters')
        return {'hits': self.corpus.search(query, self.allowed), 'scope': self.scope}

    def read(self, evidence_id: str) -> dict:
        document = self.corpus.read(evidence_id, self.allowed)
        facts = {f'{evidence_id}/{key}': value for key, value in
                 extract_numeric_facts(document['evidence']).items()}
        self.facts.update(facts)
        self.read_ids.add(evidence_id)
        return {'document': document, 'facts': facts}

    def calculate(self, steps: list[dict]) -> dict:
        result = execute_action({'steps': steps}, self.facts)
        if any(fact['evidence_id'] not in self.allowed for fact in result['facts'].values()):
            raise ValueError('calculation cites evidence outside scope')
        return {**result, 'scope': self.scope, 'corpus_sha256': self.corpus.fingerprint}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--scope', type=Path, required=True)
    parser.add_argument('--reports', action='store_true')
    args = parser.parse_args()
    scope = json.loads(args.scope.read_text())
    corpus_class = ReportCorpus if args.reports else Corpus
    corpus = corpus_class(str(args.corpus), scope['snapshot_id'])
    research = ResearchTools(corpus, scope)
    server = FastMCP('Finagent read-only research')
    annotations = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)

    @server.tool(annotations=annotations)
    def search_documents(query: str) -> dict:
        """Search the immutable issuer/period/cutoff scope; return evidence IDs."""
        return research.search(query)

    @server.tool(annotations=annotations)
    def get_evidence(evidence_id: str) -> dict:
        """Read authorized original evidence and register its numeric fact IDs."""
        return research.read(evidence_id)

    @server.tool(annotations=annotations)
    def calculate(steps: list[dict]) -> dict:
        """Execute up to eight binary steps on previously read facts and constants."""
        return research.calculate(steps)

    @server.prompt()
    def financial_research() -> str:
        """Load the versioned financial-research Skill and its I/O contract."""
        return (Path(__file__).resolve().parents[2] /
                'skills/financial-research/SKILL.md').read_text()

    try:
        server.run(transport='stdio')
    finally:
        corpus.close()


if __name__ == '__main__':
    main()
