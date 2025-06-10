# == main.py ==
"""CLI : 키워드 → SVG 저장"""
import sys, pathlib, datetime
from jinja2 import Environment, BaseLoader
from markdown import markdown
from llm import fetch, RES

TEMPLATE = (RES / "template.svg").read_text("utf-8")

def build_markdown(data: dict) -> str:
    meta, s = data['meta'], data['summary']
    md = [f"# {s['title']}", '', f"**기관:** {meta['agency']}  ", f"**날짜:** {meta['date']}  ", '', '## □ 개요', '', *[f"- {b}" for b in s['bullets']], '']
    for i, pg in enumerate(data['body_pages'], 1):
        md += [f"## □ 본문 {i}", '', str(pg), '']
    refs = data.get('references', [])
    if refs:
        md += ['## 참고자료', ''] + [f"{i+1}. {r}" for i, r in enumerate(refs)]
    return "\n".join(md)

def contacts_table_html(contacts):
    return "".join(f"<tr><td>{c['title']}</td><td>{c['name']}</td><td>{c['phone']}</td></tr>" for c in contacts)

def render_svg(md_text: str, data: dict) -> str:
    env = Environment(loader=BaseLoader(), autoescape=False)
    body_html = markdown(md_text, extensions=["nl2br", "tables"])
    return env.from_string(TEMPLATE).render(
        meta=data['meta'], summary=data['summary'], body_html=body_html,
        references=data.get('references', []),
    )

def save_svg(svg_str: str, out_path: pathlib.Path):
    out_path.write_text(svg_str, encoding='utf-8')

def main():
    if len(sys.argv) < 2:
        topic = input('보고서 주제/키워드: ').strip()
        pages = int(input('페이지 수(기본 2): ') or 2)
    else:
        topic = sys.argv[1]
        pages = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    data = fetch(topic, pages)
    md = build_markdown(data)
    svg = render_svg(md, data)
    svg_path = pathlib.Path.cwd() / f"press_release_{datetime.date.today()}.svg"
    save_svg(svg, svg_path)
    print(f"SVG 파일 저장: {svg_path}")

if __name__ == "__main__":
    main()
