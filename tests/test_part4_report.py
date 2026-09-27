"""Part 4. 보고서 — 네 질문에 모두 답했는지만 자동으로 봅니다 (내용은 교수자가 채점)"""
import os, re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def test_p4_report__all_answered():
    with open(os.path.join(ROOT, 'REPORT.md'), encoding='utf-8') as f:
        text = f.read()
    parts = re.split(r'^## (Q\d)', text, flags=re.M)
    answers = dict(zip(parts[1::2], parts[2::2]))
    short = []
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        body = answers.get(q, '')
        lines = [l for l in body.splitlines() if l.strip() and not l.lstrip().startswith('>') and '(여기에 답을 쓰세요)' not in l]
        n = len(re.sub(r'\s', '', ''.join(lines)))
        if n < 80:
            short.append(f'{q}({n}자)')
    assert not short, f'답이 비었거나 너무 짧습니다(80자 이상): {short}'
