import os, sys, json, yaml, openai, math, re
from pathlib import Path
from datetime import datetime
from string import Template
from dateutil import tz
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
# --- 설정 읽기 -------------------------------------------------
cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
prompt_cfg = yaml.safe_load(Path("prompt.yaml").read_text(encoding="utf-8"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# --- 검색 툴 ---------------------------------------------------
# def search_serpapi(q):
#     from serpapi import GoogleSearch
#     res = GoogleSearch({"q": q, "api_key": os.getenv("SERPAPI_API_KEY"), "hl": "ko"}).get_dict()
#     return [{"title":o["title"], "link":o["link"]} for o in res.get("organic_results",[])[:3]]

def search_ddg(q):
    from duckduckgo_search import DDGS
    return list(DDGS().text(q, max_results=3))
# --- 프롬프트 메시지 ------------------------------------------
def build_messages(topic):
    pc = prompt_cfg.copy()
    pc["assistant_plan"] = pc["assistant_plan"].format(
        tool=cfg["tool"], output_format=cfg["output_format"],
        schema_anchor=pc["schema_anchor"],
        html_tpl_anchor=pc["html_tpl_anchor"],
        svg_tpl_anchor=pc["svg_tpl_anchor"]
    )
    return [
        {"role":"system","content":pc["system"]},
        {"role":"user","content":pc["user"].format(topic=topic)},
        {"role":"assistant","content":pc["assistant_plan"]}
    ]
# --- flatten util ---------------------------------------------
def flatten(d, parent_key="", sep="."):
    items={}
    for k,v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v,dict):
            items.update(flatten(v,new_key,sep=sep))
        else:
            items[new_key]=v
    return items
# --- estimate pages (단순) -------------------------------------
def estimate_pages(text, per=1800):
    pure = re.sub(r"<[^>]+>","", text)
    return max(1, math.ceil(len(pure)/per))
# --- HTML 템플릿 치환 -----------------------------------------
def render_html(tpl,data):
    contacts = "".join(
        f"<tr><td>{c['position']}</td><td>{c['name']}</td><td>{c['tel']}</td></tr>"
        for c in data["header"]["contacts"]
    )
    summaries=[]
    for s in data["summaries"]:
        block = "<div class='boxed'>" if s["boxed"] else ""
        block += f"<h2>{s['title']}</h2><ul>"+"".join(f"<li>{b}</li>" for b in s["bullets"])+ "</ul>"
        block += "</div>" if s["boxed"] else ""
        summaries.append(block)
    bodies=[]
    for b in data["body"]:
        bodies.append(f"<p class='b{b['level']}'>{b['text']}</p>")
        if b.get("sub"):
            bodies.append("<ul>"+ "".join(f"<li>{s}</li>" for s in b["sub"]) + "</ul>")
    foots = "".join(f"<li>[{f['id']}] {f['src']}({f['date']})</li>" for f in data["footnotes"])
    photo_mark = "■" if not data["header"]["photo"] else "□"
    fill = {
        **flatten(data),
        "contacts_table":contacts,
        "summary_blocks":"".join(summaries),
        "body_blocks":"".join(bodies),
        "footnote_blocks":foots,
        "photo_mark":photo_mark
    }
    html = Template(tpl).safe_substitute(fill)
    pages = estimate_pages(html)
    html = html.replace("${header.pages}", str(pages))
    return html
# --- 메인 ------------------------------------------------------
def main():
    topic = "대전 부동산 전망"
    # (옵션) 검색 결과 수집 – Collector 단계에서 직접 넣어도 OK
    #if cfg["tool"]=="serpapi": sources=search_serpapi(topic)
    if cfg["tool"]=="duckduckgo": sources=search_ddg(topic)
    else: sources=[]
    # GPT 호출
    rsp = client.chat.completions.create(
        model=cfg["model"],
        messages=build_messages(topic),
        temperature=cfg["params"]["temperature"],
        top_p=cfg["params"]["top_p"],
        max_tokens=cfg["params"]["max_tokens"]
    )
    content = rsp.choices[0].message.content.strip()
    # JSON 또는 바로 HTML/SVG
    if content.lstrip().startswith("{"):
        data=json.loads(content)
        tpl = prompt_cfg["html_tpl_anchor"] if cfg["output_format"]=="html" else prompt_cfg["svg_tpl_anchor"]
        output = render_html(tpl,data) if cfg["output_format"]=="html" else Template(tpl).safe_substitute(flatten(data))
    else:
        output = content   # 이미 렌더된 코드
    ext = "html" if cfg["output_format"]=="html" else "svg"
    filename = f"report_{datetime.now():%Y%m%d_%H%M}.{ext}"
    Path(filename).write_text(output, encoding="utf-8")
    print("✔ 파일 저장:", filename)

if __name__=="__main__":
    main()
