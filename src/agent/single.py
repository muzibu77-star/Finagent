"""The single-agent control group: one tool-using agent checks a draft itself.

Where the multi-agent flow extracts claims, checks them deterministically and
proposes revisions, this agent gets the draft and a small read-only toolbox over
the source library and decides every number on its own.  It is the control group
for that comparison and, restricted to the numbers the flow hands to a person,
the second opinion of the hybrid flow; both uses share the metered client, the
tool queries and the scoring with the rest of the project.

``plan.md`` asks for that comparison before the multi-agent design is kept.
"""

import argparse
import ast
from decimal import Decimal, localcontext
import json
from pathlib import Path
import sqlite3

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from src.agent.llm import ApiChatModel
from src.data.library import Library, terms
from src.data.table_facts import UNIT
from src.model.api_client import ApiClient
from src.verify.extract import locate, parse

# Model turns, not tool calls: one turn may batch several tool calls.  A run
# that exceeds the cap is scored on the verdicts it has already given.
MAX_STEPS = 40

SYSTEM = '''你是证券公司的质控人员，正在核对一篇关于{company}{year}年年度报告的点评。\
请逐个检查点评中每一个以“亿元”“万元”“百万元”或“%”表示的数字，判断它与年报是否一致，\
或者能否由年报数据正确算出。你可以反复调用工具：先查证，再下结论，不要凭记忆猜测。

工具（都只能看到发行人 {code} 在写稿截止日 {as_of} 之前披露的文件）：
- find_number：按数字的书写形式精确查找年报表格中的事实，能看到该数字所在的行标签、列表头与单位；
- search_pages：按关键词检索年报页面，只返回文件、页码、表题与命中片段；
- page_text：读某份文件的某一页，带 keyword 时只返回含该关键词的行与上下各 1 行，并附该页表题与单位声明行；
- calculate：用 Decimal 计算只含数字与 + - * / ( ) 的算式。

先按点评里写的数字用 find_number 精确查找，查不到再检索页面。

每个数字都必须给出“正确”“错误”或“无法判断”，不要遗漏；查不到依据就写“无法判断”。\
最后只输出 JSON 数组，每个数字一个对象：\
{{"number": "点评中的数字原样不含单位", "occurrence": 同样写法在点评中第几次出现(从1开始), \
"verdict": "正确"、"错误"或"无法判断"}}。除这个 JSON 数组外不要输出任何文字。'''

TASK = '''请核对下面这篇点评里的数字。

点评：
{review}'''

INVESTIGATE_SYSTEM = '''你是证券公司的质控人员，正在复核一篇关于{company}{year}年年度报告的点评里\
几个需要人工判断的数字。请只检查下面列出的数字，逐个判断它与年报是否一致，\
或者能否由年报数据正确算出。你可以反复调用工具：先查证，再下结论，不要凭记忆猜测。

工具（都只能看到发行人 {code} 在写稿截止日 {as_of} 之前披露的文件）：
- find_number：按数字的书写形式精确查找年报表格中的事实，能看到该数字所在的行标签、列表头与单位；
- search_pages：按关键词检索年报页面，只返回文件、页码、表题与命中片段；
- page_text：读某份文件的某一页，带 keyword 时只返回含该关键词的行与上下各 1 行，并附该页表题与单位声明行；
- calculate：用 Decimal 计算只含数字与 + - * / ( ) 的算式。

先按点评里写的数字用 find_number 精确查找，查不到再检索页面。

列出的每个数字都必须给出“正确”“错误”或“无法判断”，不要遗漏；查不到依据就写“无法判断”。\
最后只输出 JSON 数组，每个数字一个对象：\
{{"number": "点评中的数字原样不含单位", "occurrence": 同样写法在点评中第几次出现(从1开始), \
"verdict": "正确"、"错误"或"无法判断", "expected": 判“错误”时年报中的应有值(含单位，其他情况留空), \
"basis": 依据(文件名、页码与原文片段，其他情况留空)}}。除这个 JSON 数组外不要输出任何文字。'''

INVESTIGATE_TASK = '''请核对下面这篇点评里列出的数字，其余数字不用管。

点评：
{review}

需要核对的数字（原文写法与在点评中第几次出现）：
{numbers}'''

MEANING = {'正确': 'consistent', '错误': 'inconsistent', '无法判断': 'unsupported'}

REFUSED = '该文件在写稿截止日之后才披露，不能作为依据。'
PAGE_CHARS = 8000  # one page of an annual report, long enough for any table
SNIPPET_CHARS = 80  # one search hit: short enough that a whole page stays out of the context


def table_titles(db, document_id: str, page: int) -> list[str]:
    """The table titles bound to one page, so an excerpt says which table it is."""
    rows = db.execute('SELECT DISTINCT title FROM facts WHERE document_id = ? AND page = ?'
                      ' AND title IS NOT NULL ORDER BY title', (document_id, page))
    return [row['title'] for row in rows]


def unit_lines(text: str) -> list[str]:
    """The lines of a page that state a monetary unit, e.g. "（人民币百万元）"."""
    return [line.strip() for line in text.splitlines() if UNIT.search(line)]


def keyword_lines(text: str, keyword: str) -> str:
    """Every line that mentions ``keyword`` with one line of context around it.

    Returns an empty string when the page does not mention the keyword, so the
    caller can report that instead of pretending the page is empty.
    """
    lines = text.splitlines()
    hits = [index for index, line in enumerate(lines) if keyword in line]
    if not hits:
        return ''
    wanted = sorted({index for hit in hits for index in (hit - 1, hit, hit + 1)
                     if 0 <= index < len(lines)})
    return '\n'.join(lines[index] for index in wanted)


def snippet(text: str, query: str) -> str:
    """The line of a page that mentions ``query``, cut to ``SNIPPET_CHARS``.

    The whole query is preferred, so a hit shows the words that were searched
    for; a page that only mentions part of it falls back to the query's terms,
    because pages and the index tokenise Chinese differently.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for wanted in [query.strip(), *terms(query)]:
        if not wanted:
            continue
        for line in lines:
            at = line.lower().find(wanted.lower())
            if at != -1:
                start = max(0, at - SNIPPET_CHARS // 4)
                room = SNIPPET_CHARS - (2 if start else 0)
                return ('……' if start else '') + line[start:start + room]
    return ''


def build_tools(library: Path, code: str, as_of: str) -> list:
    """The read-only toolbox of one draft: scoped to its issuer and cutoff date.

    Each call opens its own read-only connection: the agent runs tools on a
    worker thread, and a connection may only be used where it was opened.
    """

    @tool
    def find_number(text: str) -> str:
        """按书写形式在年报表格事实中精确查找一个数字（本发行人，截止日前披露）。"""
        facts = Library(library).find_number(text, code=code, as_of=as_of)[:8]
        return json.dumps([{'document': fact['document'], 'page': fact['page'],
                            'text': fact['text'], 'row_label': fact['row_label'],
                            'column_header': fact['column_header'], 'unit': fact['unit'],
                            'scale': fact['scale'], 'scope': fact['scope'],
                            'title': fact['title'], 'published': fact['published']}
                           for fact in facts], ensure_ascii=False) or '没有找到这个数字'

    @tool
    def search_pages(query: str) -> str:
        """按关键词检索年报页面（本发行人，截止日前披露），返回文件、页码、表题与命中片段。"""
        books = Library(library)
        found = []
        for row in books.search_pages(query, code=code, as_of=as_of)[:5]:
            try:
                text = books.page_text(row['document_id'], row['page'])
            except KeyError:
                continue  # a term hit without page text cannot be read
            found.append({'document_id': row['document_id'], 'document': row['title'],
                          'page': row['page'], 'snippet': snippet(text, query),
                          'table': '；'.join(table_titles(books.db, row['document_id'],
                                                         row['page']))})
        return json.dumps(found, ensure_ascii=False) or '没有检索到页面'

    @tool
    def page_text(document_id: str, page: int, keyword: str = '') -> str:
        """读某份文件某一页；请带 keyword，只返回含该关键词的行与上下各 1 行，并附该页表题与单位声明行，不带 keyword 才返回整页。只能读截止日前已披露的文件。"""
        books = Library(library)
        row = books.db.execute('SELECT published FROM documents WHERE id = ?',
                               (document_id,)).fetchone()
        if row is None:
            return '没有这份文件。'
        if row['published'] > as_of:
            return REFUSED
        try:
            text = books.page_text(document_id, page)
        except KeyError:
            return '这份文件没有这一页。'
        if not keyword:
            return text[:PAGE_CHARS] + ('……' if len(text) > PAGE_CHARS else '')
        body = keyword_lines(text, keyword)
        if not body:
            return '未找到“{}”。该页前 5 行：\n{}'.format(keyword,
                                                          '\n'.join(text.splitlines()[:5]))
        notes = [f'表题：{title}' for title in table_titles(books.db, document_id, page)]
        notes += [f'单位声明：{line}' for line in unit_lines(text)]
        return '\n'.join(notes + [body])

    @tool
    def calculate(expression: str) -> str:
        """用 Decimal 计算只含数字与 + - * / ( ) 的算式，例如 (1741.44-1556.03)/1556.03*100。"""
        try:
            with localcontext() as context:
                context.prec = 28
                return str(arithmetic(ast.parse(expression, mode='eval')))
        except (SyntaxError, ValueError, ArithmeticError) as error:
            return f'算不了：{error}'

    return [find_number, search_pages, page_text, calculate]


def arithmetic(node: ast.AST) -> Decimal:
    """One node of an arithmetic expression, evaluated in Decimal.

    Only numbers, unary signs and the four operations are accepted; nothing is
    ever executed, so an expression can never reach anything but a Decimal.
    """
    if isinstance(node, ast.Expression):
        return arithmetic(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = arithmetic(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left, right = arithmetic(node.left), arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError('只允许数字与 + - * / ( )')


def run_agent(llm, tools: list, prompt: str, task: str, max_steps: int) -> list:
    """The messages of one ReAct run, cut off after ``max_steps`` model turns."""
    agent = create_react_agent(llm, tools, prompt=prompt)
    messages: list = []
    try:
        for state in agent.stream({'messages': [HumanMessage(content=task)]},
                                   config={'recursion_limit': 2 * max_steps + 2},
                                   stream_mode='values'):
            messages = state['messages']
    except GraphRecursionError:
        pass  # the verdicts given so far still count; the rest is scored as missed
    return messages


def final_answer(messages: list) -> str:
    """The last assistant message that is an answer and not a tool call."""
    for message in reversed(messages):
        if isinstance(message, AIMessage) and message.content and not message.tool_calls:
            return message.content
    return ''


def review_draft_single(draft: dict, *, library: Path, model: str, url: str, budget_usd: float,
                        cache: Path, max_steps: int = MAX_STEPS) -> dict:
    """Run the agent on one draft and return its verdicts and its trace.

    ``draft`` holds ``text``, ``code``, ``company``, ``fiscal_year`` and
    ``as_of``.  The result carries, per number start offset, the verdict the
    agent gave (``consistent``, ``inconsistent`` or ``unsupported``), the
    messages of the run, the number of tool calls and the token use.  A run cut
    off at ``max_steps`` turns keeps the verdicts already given.
    """
    client = ApiClient(model, budget_usd, Path(cache), url=url)
    llm = ApiChatModel(client=client)
    messages = run_agent(llm, build_tools(Path(library), draft['code'], draft['as_of']),
                         SYSTEM.format(company=draft['company'], year=draft['fiscal_year'],
                                       code=draft['code'], as_of=draft['as_of']),
                         TASK.format(review=draft['text']), max_steps)
    answer = final_answer(messages)
    verdicts = {}
    for item in parse(answer):
        start = locate(draft['text'], str(item['number']), item.get('occurrence'))
        if start is not None and start not in verdicts:
            verdicts[start] = {'verdict': MEANING.get(item.get('verdict'), 'unsupported')}
    return {'verdicts': verdicts, 'answer': answer,
            'tool_calls': sum(1 for message in messages if isinstance(message, ToolMessage)),
            'usage': dict(llm.used),
            'messages': [trace(message) for message in messages]}


def investigate_draft(draft: dict, numbers: list[dict], *, library: Path, model: str, url: str,
                      budget_usd: float, cache: Path, max_steps: int = MAX_STEPS) -> dict:
    """Run the tool agent over only the numbers a person was asked to judge.

    ``draft`` holds the draft's ``text``, ``code``, ``company``, ``fiscal_year``
    and ``as_of``; ``numbers`` are the entries to check, each with its written
    ``text`` and its ``occurrence`` in the draft.  The answer comes back as
    parsed items together with the token use and the tool call count; what to do
    with a verdict stays with the caller.
    """
    client = ApiClient(model, budget_usd, Path(cache), url=url)
    llm = ApiChatModel(client=client)
    listed = '\n'.join(f'{index}. {item["text"]}（第{item["occurrence"]}次出现）'
                       for index, item in enumerate(numbers, 1))
    messages = run_agent(
        llm, build_tools(Path(library), draft['code'], draft['as_of']),
        INVESTIGATE_SYSTEM.format(company=draft['company'], year=draft['fiscal_year'],
                                  code=draft['code'], as_of=draft['as_of']),
        INVESTIGATE_TASK.format(review=draft['text'], numbers=listed), max_steps)
    answer = final_answer(messages)
    return {'items': parse(answer), 'answer': answer,
            'tool_calls': sum(1 for message in messages if isinstance(message, ToolMessage)),
            'usage': dict(llm.used), 'messages': [trace(message) for message in messages]}


def trace(message) -> dict:
    """One message of the run in a shape that can be written to a trace file."""
    row = {'type': message.type, 'content': message.content}
    if isinstance(message, AIMessage) and message.tool_calls:
        row['tool_calls'] = [{'name': call['name'], 'args': call['args']} for call in
                             message.tool_calls]
    if isinstance(message, ToolMessage):
        row['tool_call_id'] = message.tool_call_id
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--reviews', type=Path, required=True)
    parser.add_argument('--review-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='deepseek-flash')
    parser.add_argument('--url', default='https://api.deepseek.com/chat/completions')
    parser.add_argument('--budget-usd', type=float, required=True)
    args = parser.parse_args()
    reviews = [json.loads(line) for line in args.reviews.read_text().splitlines()]
    review = next((row for row in reviews if row['id'] == args.review_id), None)
    if review is None:
        raise SystemExit(f"review {args.review_id} is not in {args.reviews}")
    db = sqlite3.connect(f'file:{args.library}?mode=ro', uri=True)
    published = db.execute('SELECT published FROM documents WHERE id = ?',
                           (review['document_id'],)).fetchone()[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = review_draft_single({'text': review['text'], 'code': review['code'],
                                  'company': review['company'],
                                  'fiscal_year': review['fiscal_year'], 'as_of': published},
                                 library=args.library, model=args.model, url=args.url,
                                 budget_usd=args.budget_usd,
                                 cache=args.output.parent / 'api_cache.jsonl')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({key: result[key] for key in ('tool_calls', 'usage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
