"""Real MCP stdio discovery, skill loading and read-only tool execution."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import re
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def check(output: Path) -> None:
    corpus_path = Path('data/staged/m3_acceptance_v1/corpus.json')
    documents = json.loads(corpus_path.read_text())
    from src.tools.benchmark_tools import extract_numeric_facts
    key = next(key for key, row in documents.items()
               if extract_numeric_facts(row['evidence']))
    document = documents[key]
    from src.data.leakage import content_text
    query = ' '.join(re.findall(r'[A-Za-z]{4,}', content_text(document['evidence']))[:8])
    scope = {'company': document['company'], 'period': document['period'],
             'snapshot_id': 'm3-frozen-v1'}
    scope_path = output / 'scope.json'
    scope_path.write_text(json.dumps(scope))
    params = StdioServerParameters(command=sys.executable, args=[
        '-m', 'src.service.mcp_server', '--corpus', str(corpus_path), '--scope', str(scope_path)])
    trace = []
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
            assert {t.name for t in listed.tools} == {'search_documents', 'get_evidence', 'calculate'}
            assert all(t.annotations.readOnlyHint for t in listed.tools)
            prompt = await session.get_prompt('financial_research')
            loaded = prompt.messages[0].content.text
            expected = Path('skills/financial-research/SKILL.md').read_text()
            assert loaded == expected
            trace.append({'event': 'skill_loaded', 'sha256': hashlib.sha256(loaded.encode()).hexdigest()})
            async def call(name, arguments):
                result = await session.call_tool(name, arguments)
                data = result.structuredContent
                if data is None and not result.isError:
                    data = json.loads(result.content[0].text)
                trace.append({'tool': name, 'arguments': arguments, 'error': result.isError,
                              'response': data if data is not None else [c.model_dump() for c in result.content]})
                (output / 'trace.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2))
                return result, data
            response, search = await call('search_documents', {'query': query})
            assert not response.isError and search['hits']
            key = search['hits'][0]['evidence_id']
            response, data = await call('get_evidence', {'evidence_id': key})
            assert not response.isError
            fact_id = next(iter(data['facts']))
            response, result = await call('calculate', {'steps': [
                {'operation': 'multiply', 'args': [fact_id, 'const_1']}]})
            assert not response.isError and result['fact_ids'] == [fact_id]
            outside = next(k for k, row in documents.items() if row['company'] != scope['company'])
            response, _ = await call('get_evidence', {'evidence_id': outside})
            assert response.isError
            response, _ = await call('calculate', {'steps': [
                {'operation': 'multiply', 'args': ['fabricated', 'const_1']}]})
            assert response.isError
    (output / 'trace.json').write_text(json.dumps(trace, ensure_ascii=False, indent=2))
    (output / 'summary.json').write_text(json.dumps({'passed': True, 'checks': 6,
        'client': 'MCP Python SDK stdio ClientSession; Skill loaded through MCP prompt',
        'scope': scope, 'limits': 'Tool/transport contract validation, not financial answer accuracy.'}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    asyncio.run(check(args.output_dir))


if __name__ == '__main__':
    main()
