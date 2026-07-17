"""
Runner de lote para o GEO Auditor híbrido.

Responsabilidades:
- Renderizar N páginas com concorrência controlada e browser reciclado
- Rodar heurística (10 critérios) + LLM (5 critérios semânticos)
- Checkpoint incremental em disco: rerun do Streamlit não perde trabalho
- Score híbrido combinando as duas fontes
"""

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup

import geo_core as core
import geo_llm as llm

CHECKPOINT_DIR = Path(".geo_checkpoints")

# Critérios que a heurística mantém (os 5 semânticos vão para o LLM)
HEURISTIC_KEEP = ["A2", "A4", "A5", "B1", "B2", "B3", "B5", "C1", "C2", "D1"]


@dataclass
class HybridResult:
    url: str
    ok: bool
    error: str = ""
    h1: str = ""
    word_count: int = 0
    js_dependency: float = 0.0
    content_type: str = ""
    heuristic: list = field(default_factory=list)   # list[dict]
    semantic: list = field(default_factory=list)    # list[dict]
    target_question: str = ""
    rewritten_lead: str = ""
    coverage_gaps: list = field(default_factory=list)
    suggested_faq: list = field(default_factory=list)
    llm_ok: bool = False
    llm_error: str = ""
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def all_criteria(self):
        return self.heuristic + self.semantic

    @property
    def score(self) -> int:
        scored = [c for c in self.all_criteria if c["status"] != "N/A"]
        if not scored:
            return 0
        pts = {"PASSA": 2, "PARCIAL": 1, "FALHA": 0}
        got = sum(pts[c["status"]] for c in scored)
        return round(got / (len(scored) * 2) * 100)

    @property
    def score_heuristic(self) -> int:
        scored = [c for c in self.heuristic if c["status"] != "N/A"]
        if not scored:
            return 0
        pts = {"PASSA": 2, "PARCIAL": 1, "FALHA": 0}
        return round(sum(pts[c["status"]] for c in scored) / (len(scored) * 2) * 100)

    @property
    def score_semantic(self):
        scored = [c for c in self.semantic if c["status"] != "N/A"]
        if not scored:
            return None
        pts = {"PASSA": 2, "PARCIAL": 1, "FALHA": 0}
        return round(sum(pts[c["status"]] for c in scored) / (len(scored) * 2) * 100)


# ---------------------------------------------------------------- render


async def _render_batch(urls, ua, timeout, wait, concurrency, recycle_every, on_done=None):
    """
    Renderiza urls com concorrência limitada. Recicla o browser a cada
    `recycle_every` páginas — Chromium vaza memória e o Community Cloud tem 1GB.
    Retorna dict url -> (raw_html, rendered_html, error)
    """
    from playwright.async_api import async_playwright

    out = {}
    sem = asyncio.Semaphore(concurrency)

    async with async_playwright() as p:
        chunks = [urls[i:i + recycle_every] for i in range(0, len(urls), recycle_every)]

        for chunk in chunks:
            browser = await p.chromium.launch(
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                      "--single-process", "--no-zygote"]
            )
            ctx = await browser.new_context(user_agent=ua,
                                            viewport={"width": 1366, "height": 900})

            async def one(url):
                async with sem:
                    page = await ctx.new_page()
                    raw = {"html": ""}

                    async def cap(resp):
                        if resp.url.rstrip("/") == url.rstrip("/") and not raw["html"]:
                            try:
                                raw["html"] = await resp.text()
                            except Exception:
                                pass

                    page.on("response", cap)
                    try:
                        await page.goto(url, timeout=timeout * 1000,
                                        wait_until="domcontentloaded")
                        try:
                            await page.wait_for_load_state("networkidle",
                                                           timeout=wait * 1000)
                        except Exception:
                            await page.wait_for_timeout(wait * 1000)
                        rendered = await page.content()
                        out[url] = (raw["html"], rendered, None)
                    except Exception as e:
                        out[url] = (None, None, str(e)[:200])
                    finally:
                        try:
                            await page.close()
                        except Exception:
                            pass
                        if on_done:
                            on_done(url)

            await asyncio.gather(*[one(u) for u in chunk])
            await ctx.close()
            await browser.close()

    return out


def render_batch(urls, ua, timeout, wait, concurrency, recycle_every, on_done=None):
    try:
        return asyncio.run(_render_batch(urls, ua, timeout, wait, concurrency,
                                         recycle_every, on_done))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                _render_batch(urls, ua, timeout, wait, concurrency, recycle_every, on_done))
        finally:
            loop.close()


# ---------------------------------------------------------------- heuristics


def run_heuristics(rendered_html, raw_html, url):
    """Roda os 10 critérios determinísticos. Retorna (result_dict, body_text) ou (None, erro)."""
    soup = BeautifulSoup(rendered_html, "lxml")
    main, text = core.extract_main(rendered_html)
    wc = core.word_count(text)

    if wc < 50:
        return None, (f"corpo com apenas {wc} palavras após limpeza — "
                      "possível bloqueio, paywall ou renderização falha")

    raw_wc = 0
    if raw_html:
        _, raw_text = core.extract_main(raw_html)
        raw_wc = core.word_count(raw_text)
    js_dep = max(round((1 - raw_wc / wc) * 100, 1), 0.0) if wc else 0.0

    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(" ", strip=True) if h1_tag else (
        soup.title.get_text(strip=True) if soup.title else "")

    cs = [
        core.a2_headings(main),
        core.a4_density(main),
        core.a5_extractable(main, wc),
        core.b1_original_data(text, wc),
        core.b2_verifiable(main, text),
        core.b3_authority(soup, text),
        core.b5_definitional(text, wc),
        core.c1_entities(text, wc),
        core.c2_consistency(text, h1),
        core.d1_freshness(soup, text),
    ]

    return {
        "h1": h1,
        "word_count": wc,
        "js_dependency": js_dep,
        "content_type": core.guess_type(main, text, wc),
        "criteria": [{"code": c.code, "name": c.name, "group": c.group,
                      "status": c.status, "evidence": c.evidence, "fix": c.fix,
                      "source": "heurística"} for c in cs],
        "body_text": text,
    }, None


def signals_for_llm(h) -> str:
    """Resumo compacto dos achados heurísticos, como contexto de apoio ao LLM."""
    lines = [f"- {c['code']} {c['name']}: {c['status']} ({c['evidence'][:70]})"
             for c in h["criteria"]]
    lines.append(f"- Palavras no corpo: {h['word_count']}")
    lines.append(f"- Dependência de JS: {h['js_dependency']}%")
    return "\n".join(lines)


# ---------------------------------------------------------------- checkpoint


def ckpt_path(batch_id):
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    return CHECKPOINT_DIR / f"{batch_id}.jsonl"


def save_ckpt(batch_id, result: HybridResult):
    with open(ckpt_path(batch_id), "a") as f:
        f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def load_ckpt(batch_id):
    p = ckpt_path(batch_id)
    if not p.exists():
        return {}
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out[d["url"]] = HybridResult(**d)
            except Exception:
                continue
    return out


def clear_ckpt(batch_id):
    p = ckpt_path(batch_id)
    if p.exists():
        p.unlink()


# ---------------------------------------------------------------- orchestration


def audit_batch(urls, ua, timeout, wait, concurrency, recycle_every,
                use_llm, provider, api_key, model, max_words,
                batch_id, resume=True, progress=None):
    """
    Gera HybridResult por URL. `progress(done, total, msg)` para UI.
    Retoma do checkpoint se resume=True.
    """
    done = load_ckpt(batch_id) if resume else {}
    todo = [u for u in urls if u not in done]

    results = list(done.values())

    if not todo:
        return results

    if progress:
        progress(len(done), len(urls), f"Retomando: {len(done)} já auditadas")

    # --- fase 1: render (concorrente)
    rendered_count = {"n": len(done)}

    def on_page(url):
        rendered_count["n"] += 1
        if progress:
            progress(rendered_count["n"], len(urls), f"Renderizando ({rendered_count['n']}/{len(urls)})")

    pages = render_batch(todo, ua, timeout, wait, concurrency, recycle_every, on_page)

    # --- fase 2: heurística + LLM (sequencial no LLM, com backoff interno)
    for i, url in enumerate(todo):
        raw, rend, err = pages.get(url, (None, None, "não renderizada"))

        if err or not rend:
            r = HybridResult(url=url, ok=False, error=err or "conteúdo vazio")
            results.append(r)
            save_ckpt(batch_id, r)
            continue

        h, herr = run_heuristics(rend, raw, url)
        if herr:
            r = HybridResult(url=url, ok=False, error=herr)
            results.append(r)
            save_ckpt(batch_id, r)
            continue

        r = HybridResult(
            url=url, ok=True, h1=h["h1"], word_count=h["word_count"],
            js_dependency=h["js_dependency"], content_type=h["content_type"],
            heuristic=h["criteria"],
        )

        if use_llm and api_key:
            if progress:
                progress(len(done) + i, len(urls), f"LLM ({i+1}/{len(todo)}): {url[:45]}")
            lr = llm.judge(provider, api_key, model, url, h["h1"], h["content_type"],
                           signals_for_llm(h), h["body_text"], max_words)
            r.llm_ok = lr.ok
            if lr.ok:
                r.semantic = [{"code": v.code, "name": v.name, "group": "semântico",
                               "status": v.status, "evidence": v.evidence,
                               "fix": v.fix, "rationale": v.rationale,
                               "source": "LLM"} for v in lr.verdicts]
                r.target_question = lr.target_question
                r.rewritten_lead = lr.rewritten_lead
                r.coverage_gaps = lr.coverage_gaps
                r.suggested_faq = lr.suggested_faq
                r.cost_usd = lr.cost_usd
                r.tokens_in = lr.tokens_in
                r.tokens_out = lr.tokens_out
            else:
                r.llm_error = lr.error

        results.append(r)
        save_ckpt(batch_id, r)

    if progress:
        progress(len(urls), len(urls), "Concluído")

    return results
