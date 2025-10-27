#!/usr/bin/env python3
# -*- coding: utf-8 -*-

环境变量
------------------
LLM_API_URL  (默认: https://api.example.com/v1/chat/completions)
LLM_API_KEY  (必需)

示例
-------
python extractor.py \
  --input ./papers.jsonl \
  --output ./result.jsonl \
  --ontology ./ontology.xlsx \
  --sheet Sheet1 \
  --workers 6

"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests

# ----------------------------- 日志设置 ---------------------------------

def setup_logger(name: str = "llm_extraction", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log_dir = Path("logs"); log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

logger = setup_logger()

# ----------------------------- API -------------------------------------

DEFAULT_API_URL = "https://api.example.com/v1/chat/completions"


def get_env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or (isinstance(v, str) and not v.strip()):
        raise RuntimeError(f"缺少环境变量: {name}")
    return v


def post_llm(messages: List[Dict[str, str]], stream: bool = True, max_retries: int = 3) -> Dict:
    api_url = os.getenv("LLM_API_URL", DEFAULT_API_URL)
    api_key = get_env("LLM_API_KEY")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "messages": messages,
        "stream": stream,
        "temperature": 0.1,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"}
    }

    for attempt in range(max_retries):
        try:
            with requests.post(api_url, headers=headers, json=payload, stream=stream, timeout=180) as r:
                if r.status_code != 200:
                    logger.warning(f"API HTTP {r.status_code}: {r.text[:200]}")
                    time.sleep(2 ** attempt)
                    continue
                if stream:
                    content = ""
                    for raw in r.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        try:
                            obj = json.loads(line[5:])
                            delta = obj.get("choices", [{}])[0].get("delta", {})
                            piece = delta.get("content")
                            if piece:
                                content += piece
                        except Exception:
                            continue
                    try:
                        return json.loads(content) if content else {"triples": []}
                    except Exception:
                        logger.error("流式JSON解析失败; 返回空三元组")
                        return {"triples": []}
                else:
                    obj = r.json()
                    text = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
                    try:
                        return json.loads(text) if text else {"triples": []}
                    except Exception:
                        logger.error("非流式JSON解析失败; 返回空三元组")
                        return {"triples": []}
        except requests.Timeout:
            logger.warning(f"超时 (尝试 {attempt+1}/{max_retries})")
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"API错误: {e}")
            time.sleep(2 ** attempt)
    logger.error("重试后API仍失败; 返回空三元组")
    return {"triples": []}

# --------------------------- 提示构建 ---------------------------------

def build_prompt(text: str, ontology_rows: List[Tuple], max_onto: int = 150) -> List[Dict[str, str]]:
    # 请在这里输入你的prompt
    # 你可以根据需要自定义提示词，以下是一个通用示例：
    prompt_header = (
        "你是一个信息抽取专家。请从给定文本中抽取出结构化的三元组信息。"
        "三元组格式为：主体-关系-客体。"
        "请输出JSON格式：{\n  \"triples\": [ {\"subject\":..., \"relation\":..., \"object\":...}, ... ]\n}。"
    )
    
    # 本体预览（前N行）
    preview_lines: List[str] = []
    for row in ontology_rows[:max_onto]:
        if len(row) >= 3:
            s = str(row[0] or "").strip(); r = str(row[1] or "").strip(); o = str(row[2] or "").strip()
            if s and r and o:
                preview_lines.append(f"- 主体:{s} | 关系:{r} | 客体:{o}")
    preview = "\n".join(preview_lines)

    user = (
        f"{prompt_header}\n\n[本体示例]\n{preview}\n\n"
        f"[待分析文本]\n{text}\n\n"
        "注意：仅输出JSON对象，不要添加其他解释性文字。"
    )
    return [{"role": "user", "content": user}]

# --------------------------- 数据清理 ----------------------------------

def to_ontology_set(ontology_rows: Iterable[Iterable]) -> set:
    S = set()
    for row in ontology_rows:
        if len(row) >= 3:
            s = str(row[0] or "").strip(); r = str(row[1] or "").strip(); o = str(row[2] or "").strip()
            if s and r and o:
                S.add(f"{s}|{r}|{o}")
    return S


def clean_triples(raw: Dict, ontology_set: set) -> List[Dict[str, str]]:
    triples = raw.get("triples", []) or []
    seen = set(); out: List[Dict[str, str]] = []
    for t in triples:
        try:
            s = str(t.get("subject", "")).strip(); r = str(t.get("relation", "")).strip(); o = str(t.get("object", "")).strip()
            if not (s and r and o):
                continue
            key = f"{s}|{r}|{o}"
            if key in ontology_set:
                continue  # 过滤已知本体三元组
            if key in seen:
                continue
            seen.add(key)
            out.append({"subject": s, "relation": r, "object": o, "triple": key})
        except Exception:
            continue
    return out

# --------------------------- 文件I/O ---------------------------------

def read_jsonl(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def write_outputs(rows: List[Dict], out_jsonl: Path):
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    # JSONL
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # CSV/XLSX
    df = pd.DataFrame(rows)
    df.to_csv(out_jsonl.with_suffix('.csv'), index=False, encoding='utf-8-sig')
    try:
        df.to_excel(out_jsonl.with_suffix('.xlsx'), index=False)
    except Exception as e:
        logger.warning(f"Excel写入失败: {e}")

# --------------------------- 主流程 --------------------------------------

def process(input_jsonl: Path, out_jsonl: Path, ontology_path: Path, sheet: str = "Sheet1", workers: int = 4, stream: bool = True) -> None:
    papers = read_jsonl(input_jsonl)
    logger.info(f"从 {input_jsonl} 加载了 {len(papers)} 个文档")

    onto_df = pd.read_excel(ontology_path, sheet_name=sheet, header=None)
    ontology_rows = onto_df.values.tolist()
    onto_set = to_ontology_set(ontology_rows)
    logger.info(f"加载本体数据: {len(ontology_rows)} 行 (唯一三元组: {len(onto_set)})")

    results: List[Dict] = []

    with ThreadPoolExecutor(max_workers=workers) as ex:
        future_map = {}
        for idx, p in enumerate(papers):
            text = p.get("text", "")
            if not text:
                continue
            messages = build_prompt(text, ontology_rows)
            fut = ex.submit(post_llm, messages, stream)
            future_map[fut] = idx

        for fut in as_completed(future_map):
            idx = future_map[fut]
            p = papers[idx]
            raw = fut.result() or {"triples": []}
            cleaned = clean_triples(raw, onto_set)
            for t in cleaned:
                results.append({
                    "idx": p.get("idx", idx),
                    "subject": t["subject"],
                    "relation": t["relation"],
                    "object": t["object"],
                    "triple": t["triple"],
                    "text": p.get("text", "")
                })
            if len(results) and len(results) % 50 == 0:
                logger.info(f"处理进度: {len(results)} 个三元组…")

    write_outputs(results, out_jsonl)
    logger.info(f"完成。共抽取 {len(results)} 个三元组 -> {out_jsonl}")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="通用LLM三元组抽取工具")
    ap.add_argument("--input", required=True, type=Path, help="输入JSONL文件路径")
    ap.add_argument("--output", required=True, type=Path, help="输出JSONL文件路径")
    ap.add_argument("--ontology", required=True, type=Path, help="本体Excel文件路径")
    ap.add_argument("--sheet", default="Sheet1", help="Excel工作表名称")
    ap.add_argument("--workers", type=int, default=4, help="线程工作数")
    ap.add_argument("--no-stream", action="store_true", help="禁用流式模式")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    process(
        input_jsonl=args.input,
        out_jsonl=args.output,
        ontology_path=args.ontology,
        sheet=args.sheet,
        workers=args.workers,
        stream=not args.no_stream,
    )
