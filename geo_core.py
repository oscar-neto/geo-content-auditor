"""
GEO Content Auditor — Heuristic Edition
Audita conteúdo renderizado (DOM via Playwright) contra critérios de GEO.
Sem LLM: apenas heurísticas determinísticas.
"""

import geo_bootstrap  # fixa PLAYWRIGHT_BROWSERS_PATH antes de importar playwright

import asyncio
import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
from bs4 import BeautifulSoup

# ---------------------------------------------------------------- config


PT_STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "da", "do", "das", "dos",
    "em", "na", "no", "nas", "nos", "por", "para", "com", "sem", "sobre", "entre",
    "e", "ou", "mas", "que", "se", "como", "quando", "onde", "qual", "quais",
    "ao", "aos", "à", "às", "pelo", "pela", "pelos", "pelas", "é", "ser", "está",
    "the", "of", "and", "to", "in", "for", "on", "with", "is", "are", "a", "an",
}

AMBIGUOUS_PT = [
    r"\bisso\b", r"\bisto\b", r"\baquilo\b", r"\bo mesmo\b", r"\bcomo vimos\b",
    r"\bcomo visto\b", r"\bconforme acima\b", r"\bdito isso\b", r"\bcomo mencionado\b",
    r"\bcomo citado\b", r"\bacima\b", r"\babaixo\b", r"\bele\b", r"\bela\b",
    r"\beles\b", r"\belas\b", r"\bdisso\b", r"\bnisso\b", r"\bdele\b", r"\bdela\b",
]

HEDGE_PT = [
    r"\btalvez\b", r"\bpode ser que\b", r"\bgeralmente\b", r"\bnormalmente\b",
    r"\bem geral\b", r"\bcostuma\b", r"\bpossivelmente\b", r"\bprovavelmente\b",
    r"\bde certa forma\b", r"\bde alguma maneira\b", r"\bvale lembrar\b",
    r"\bvale ressaltar\b", r"\bé importante destacar\b", r"\bnão é segredo\b",
    r"\bnos dias de hoje\b", r"\batualmente, \b", r"\bcada vez mais\b",
    r"\bsem dúvida\b", r"\bcom certeza\b", r"\bafinal de contas\b",
]

FLUFF_OPENERS = [
    r"^no mundo (de hoje|atual|moderno)", r"^nos dias de hoje", r"^atualmente",
    r"^cada vez mais", r"^não é novidade", r"^você já (se perguntou|parou para)",
    r"^quem nunca", r"^com o avanço", r"^em um cenário",
]

DEFINITIONAL = [
    # "X é um/uma..." — aceita texto com ou sem acento (conteúdo real varia)
    r"[\wÀ-ÿ]{3,}(?:\s+[\wÀ-ÿ]+){0,4}\s+(?:é|e|são|sao)\s+(?:um|uma|o|a|os|as)\s+[\wÀ-ÿ]{3,}",
    r"\bsignifica\b", r"\bdefine-se\b", r"\bse define como\b", r"\bconsiste em\b",
    r"\bpode ser definido como\b", r"\b(?:é|e)\s+o\s+processo\b",
    r"\b(?:é|e)\s+a\s+(?:prática|pratica|técnica|tecnica)\b",
    r"\btrata-se de\b", r"\bcaracteriza-se por\b",
    r"\bis a\b", r"\bis the\b", r"\brefers to\b", r"\bmeans\b",
]

DATE_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{4}\b",
    r"\b\d{1,2}\s+de\s+(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4}\b",
    r"\b(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+\d{4}\b",
    r"\b(19|20)\d{2}\b",
]

AUTHOR_HINTS = [
    r"(?:^|[\n>.]|\s)Por\s+[A-ZÀ-Ú][\wà-ÿ]+\s+[A-ZÀ-Ú][\wà-ÿ]+",
    r"\bautor(?:a)?\s*:", r"\bescrito por\b", r"\bwritten by\b",
    r"\bBy\s+[A-Z][a-z]+\s+[A-Z][a-z]+",
    r"\brevisado por\b", r"\breviewed by\b", r"\bpublicado por\b",
]

SOURCE_HINTS = [
    r"\bsegundo\b", r"\bde acordo com\b", r"\bconforme\b", r"\bfonte:", r"\bsource:",
    r"\bpesquisa d[aeo]\b", r"\bestudo d[aeo]\b", r"\brelatório d[aeo]\b",
    r"\baccording to\b", r"\bper the\b",
]

EXPERT_HINTS = [
    r"\bespecialista\b", r"\bdiretor\b", r"\bgerente\b", r"\bCEO\b", r"\bpesquisador\b",
    r"\bprofessor\b", r"\bdoutor\b", r"\bPhD\b", r"\bafirma\b", r"\bexplica\b",
    r"\bdeclarou\b", r"\bsegundo ele\b", r"\bsegundo ela\b",
]

BOILERPLATE_TAGS = ["nav", "header", "footer", "aside", "script", "style", "noscript",
                    "form", "iframe", "svg", "button"]

BOILERPLATE_HINTS = re.compile(
    r"(nav|menu|footer|header|sidebar|cookie|banner|breadcrumb|social|share|"
    r"newsletter|subscribe|related|recommend|comment|advertisement|ads?-|popup|modal)",
    re.I,
)

MAIN_SELECTORS = [
    "article", "main", '[role="main"]', ".post-content", ".entry-content",
    ".article-body", ".content-body", "#content", ".post-body", ".blog-content",
]

# ---------------------------------------------------------------- models


@dataclass
class Criterion:
    code: str
    name: str
    group: str
    status: str  # PASSA | PARCIAL | FALHA | N/A
    points: int
    evidence: str = ""
    fix: str = ""
    proxy: bool = False  # heurística aproxima critério semântico


@dataclass
class AuditResult:
    url: str
    ok: bool
    error: str = ""
    h1: str = ""
    title: str = ""
    word_count: int = 0
    raw_words: int = 0
    rendered_words: int = 0
    js_dependency: float = 0.0
    criteria: list = field(default_factory=list)
    first_para: str = ""
    target_question: str = ""
    content_type: str = ""

    @property
    def score(self) -> int:
        scored = [c for c in self.criteria if c.status != "N/A"]
        if not scored:
            return 0
        got = sum(c.points for c in scored)
        possible = len(scored) * 2
        return round(got / possible * 100)


# ---------------------------------------------------------------- fetching


async def _render(url: str, timeout: int, wait: int, ua: str):
    """Retorna (raw_html, rendered_html)."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        ctx = await browser.new_context(user_agent=ua, viewport={"width": 1366, "height": 900})
        page = await ctx.new_page()

        raw_html = ""

        async def capture(response):
            nonlocal raw_html
            if response.url.rstrip("/") == url.rstrip("/") and not raw_html:
                try:
                    raw_html = await response.text()
                except Exception:
                    pass

        page.on("response", capture)

        try:
            await page.goto(url, timeout=timeout * 1000, wait_until="domcontentloaded")
            try:
                await page.wait_for_load_state("networkidle", timeout=wait * 1000)
            except Exception:
                await page.wait_for_timeout(wait * 1000)
            rendered_html = await page.content()
        finally:
            await browser.close()

    return raw_html, rendered_html


def fetch(url: str, timeout: int, wait: int, ua: str):
    try:
        return asyncio.run(_render(url, timeout, wait, ua)), None
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_render(url, timeout, wait, ua)), None
        finally:
            loop.close()
    except Exception as e:
        return (None, None), str(e)


# ---------------------------------------------------------------- extraction


def strip_boilerplate(soup: BeautifulSoup) -> BeautifulSoup:
    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"class": BOILERPLATE_HINTS}):
        tag.decompose()
    for tag in soup.find_all(attrs={"id": BOILERPLATE_HINTS}):
        tag.decompose()
    return soup


def extract_main(html: str):
    """Retorna (soup_do_corpo, texto)."""
    soup = BeautifulSoup(html, "lxml")
    for t in soup.find_all(["script", "style", "noscript"]):
        t.decompose()

    best, best_len = None, 0
    for sel in MAIN_SELECTORS:
        for node in soup.select(sel):
            length = len(node.get_text(" ", strip=True))
            if length > best_len:
                best, best_len = node, length

    if best is None or best_len < 200:
        best = soup.body or soup

    main = BeautifulSoup(str(best), "lxml")
    main = strip_boilerplate(main)
    text = main.get_text("\n", strip=True)
    return main, text


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\wÀ-ÿ]+\b", text))


def sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ---------------------------------------------------------------- criteria


def mk(code, name, group, status, evidence="", fix="", proxy=False) -> Criterion:
    pts = {"PASSA": 2, "PARCIAL": 1, "FALHA": 0, "N/A": 0}[status]
    return Criterion(code, name, group, status, pts, evidence, fix, proxy)


def a1_answer_first(main, text, h1) -> Criterion:
    paras = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    paras = [p for p in paras if word_count(p) >= 8]
    if not paras:
        return mk("A1", "Answer-first", "Estrutura", "FALHA", "sem parágrafo de corpo",
                  "Abrir com um parágrafo de 40–60 palavras que responda a pergunta central.", proxy=True)

    first = paras[0]
    wc = word_count(first)
    kw = {w.lower() for w in re.findall(r"\b[\wÀ-ÿ]{4,}\b", h1)} - PT_STOPWORDS
    first_low = first.lower()
    overlap = sum(1 for k in kw if k in first_low)
    ratio = overlap / len(kw) if kw else 0

    fluff = any(re.search(p, first_low) for p in FLUFF_OPENERS)

    ev = first[:90] + ("…" if len(first) > 90 else "")
    fix = (f"Reescrever o 1º parágrafo em 40–60 palavras (atual: {wc}), respondendo diretamente "
           f"a pergunta do H1 e retomando os termos: {', '.join(sorted(kw)[:5])}.")

    if fluff:
        return mk("A1", "Answer-first", "Estrutura", "FALHA", ev,
                  "Abertura genérica. Cortar a introdução e começar pela resposta. " + fix, proxy=True)
    if 35 <= wc <= 70 and ratio >= 0.5:
        return mk("A1", "Answer-first", "Estrutura", "PASSA", ev, "", proxy=True)
    if (25 <= wc <= 100) and ratio >= 0.3:
        return mk("A1", "Answer-first", "Estrutura", "PARCIAL", ev, fix, proxy=True)
    return mk("A1", "Answer-first", "Estrutura", "FALHA", ev, fix, proxy=True)


def a2_headings(main) -> Criterion:
    hs = [h.get_text(" ", strip=True) for h in main.find_all(["h2", "h3"])]
    hs = [h for h in hs if h]
    if not hs:
        return mk("A2", "Headings interrogativos", "Estrutura", "FALHA", "nenhum H2/H3",
                  "Estruturar o conteúdo em H2/H3 formulados como perguntas reais de usuário.")
    q = [h for h in hs if h.rstrip().endswith("?")
         or re.match(r"^(como|o que|por que|porque|quando|onde|qual|quais|quanto|quem|vale a pena|"
                     r"how|what|why|when|where|which|who)\b", h.strip(), re.I)]
    ratio = len(q) / len(hs)
    ev = f"{len(q)}/{len(hs)} headings interrogativos"
    non_q = [h for h in hs if h not in q][:3]
    fix = ("Converter headings declarativos em perguntas. Ex: " +
           " | ".join(f'"{h}" → "O que é {h.lower()}?"' for h in non_q)) if non_q else ""
    if ratio >= 0.5:
        return mk("A2", "Headings interrogativos", "Estrutura", "PASSA", ev)
    if ratio >= 0.25:
        return mk("A2", "Headings interrogativos", "Estrutura", "PARCIAL", ev, fix)
    return mk("A2", "Headings interrogativos", "Estrutura", "FALHA", ev, fix)


def a3_self_contained(main) -> Criterion:
    """Proxy: seções com H2 + volume mínimo + poucas referências anafóricas na abertura."""
    h2s = main.find_all(["h2"])
    if not h2s:
        return mk("A3", "Blocos autocontidos", "Estrutura", "FALHA", "sem H2 para segmentar",
                  "Dividir o conteúdo em seções com H2, cada uma resolvendo uma pergunta sozinha.",
                  proxy=True)

    weak = 0
    example = ""
    for h in h2s:
        # varre o documento em ordem a partir do H2 até o próximo H2,
        # coletando apenas nós de texto folha (evita duplicar por aninhamento)
        heading_text = h.get_text(" ", strip=True)
        chunk = []
        for node in h.next_elements:
            if getattr(node, "name", None) == "h2" and node is not h:
                break
            if isinstance(node, str):
                s = node.strip()
                if s:
                    chunk.append(s)
        body = " ".join(chunk).strip()
        # remove o próprio heading do início do corpo da seção
        if body.lower().startswith(heading_text.lower()):
            body = body[len(heading_text):].strip()
        opener = " ".join(body.split()[:25]).lower()
        anaphora = any(re.search(p, opener) for p in AMBIGUOUS_PT[:8])
        if word_count(body) < 25 or anaphora:
            weak += 1
            if not example:
                example = h.get_text(" ", strip=True)[:60]

    ratio = weak / len(h2s)
    ev = f"{weak}/{len(h2s)} seções fracas" + (f' — ex: "{example}"' if example else "")
    fix = ("Cada seção deve abrir renomeando o assunto por extenso e ter corpo suficiente para "
           "ser citada isolada. Trocar aberturas anafóricas ('isso', 'como vimos') pelo nome da entidade.")
    if ratio <= 0.2:
        return mk("A3", "Blocos autocontidos", "Estrutura", "PASSA", ev, proxy=True)
    if ratio <= 0.5:
        return mk("A3", "Blocos autocontidos", "Estrutura", "PARCIAL", ev, fix, proxy=True)
    return mk("A3", "Blocos autocontidos", "Estrutura", "FALHA", ev, fix, proxy=True)


def a4_density(main) -> Criterion:
    paras = [p.get_text(" ", strip=True) for p in main.find_all("p")]
    paras = [p for p in paras if word_count(p) >= 5]
    if not paras:
        return mk("A4", "Densidade de parágrafo", "Estrutura", "N/A", "sem parágrafos identificáveis")
    wcs = [word_count(p) for p in paras]
    long_paras = [w for w in wcs if w > 80]
    ratio = len(long_paras) / len(wcs)
    avg = sum(wcs) / len(wcs)
    ev = f"média {avg:.0f} palavras/parágrafo; {len(long_paras)}/{len(wcs)} acima de 80"
    fix = "Quebrar parágrafos longos em blocos de 2–4 linhas, uma ideia por parágrafo."
    if ratio <= 0.1 and avg <= 60:
        return mk("A4", "Densidade de parágrafo", "Estrutura", "PASSA", ev)
    if ratio <= 0.3:
        return mk("A4", "Densidade de parágrafo", "Estrutura", "PARCIAL", ev, fix)
    return mk("A4", "Densidade de parágrafo", "Estrutura", "FALHA", ev, fix)


def a5_extractable(main, wc) -> Criterion:
    lists = len(main.find_all(["ul", "ol"]))
    tables = len(main.find_all("table"))
    lis = len(main.find_all("li"))
    ev = f"{lists} listas ({lis} itens), {tables} tabelas"
    expected = max(1, wc // 400)
    fix = ("Converter enumerações em prosa para listas <ul>/<ol> e comparações para <table>. "
           f"Para {wc} palavras, esperado ao menos {expected} bloco(s) estruturado(s).")
    total = lists + tables
    if total >= expected and lis >= 3:
        return mk("A5", "Formatos extraíveis", "Estrutura", "PASSA", ev)
    if total >= 1:
        return mk("A5", "Formatos extraíveis", "Estrutura", "PARCIAL", ev, fix)
    return mk("A5", "Formatos extraíveis", "Estrutura", "FALHA", ev, fix)


def b1_original_data(text, wc) -> Criterion:
    nums = re.findall(r"\b\d+(?:[.,]\d+)?\s*(?:%|mil|milhões|bilhões|R\$|US\$|reais|dólares)?", text)
    strong = re.findall(r"\b\d+(?:[.,]\d+)?\s*%|R\$\s*\d|US\$\s*\d|\b\d{4,}\b", text)
    propers = re.findall(r"\b[A-ZÀ-Ú][a-zà-ú]{2,}(?:\s+[A-ZÀ-Ú][a-zà-ú]{2,})?\b", text)
    per1k = len(strong) / max(wc, 1) * 1000
    ev = f"{len(strong)} dados fortes (%/moeda/números grandes), {len(set(propers))} entidades nomeadas, {per1k:.1f}/1k palavras"
    fix = ("Adicionar números específicos, datas e nomes próprios. Substituir afirmações genéricas "
           "('muitas empresas', 'grande parte') por dados quantificados com origem.")
    if per1k >= 5 and len(set(propers)) >= 5:
        return mk("B1", "Dados originais", "Substância", "PASSA", ev)
    if per1k >= 2:
        return mk("B1", "Dados originais", "Substância", "PARCIAL", ev, fix)
    return mk("B1", "Dados originais", "Substância", "FALHA", ev, fix)


def b2_verifiable(main, text) -> Criterion:
    cues = sum(len(re.findall(p, text, re.I)) for p in SOURCE_HINTS)
    ext_links = [a for a in main.find_all("a", href=True)
                 if a["href"].startswith("http")]
    nums = len(re.findall(r"\b\d+(?:[.,]\d+)?\s*%|\b\d{3,}\b", text))
    ev = f"{cues} marcadores de fonte, {len(ext_links)} links externos, {nums} números no texto"
    fix = ("Cada estatística precisa de fonte primária identificável no texto ('Segundo o relatório X, 2025') "
           "e, quando possível, link externo para a fonte original.")
    if nums == 0:
        return mk("B2", "Estatísticas verificáveis", "Substância", "N/A", "sem estatísticas no conteúdo")
    if cues >= 2 and len(ext_links) >= 1:
        return mk("B2", "Estatísticas verificáveis", "Substância", "PASSA", ev)
    if cues >= 1 or ext_links:
        return mk("B2", "Estatísticas verificáveis", "Substância", "PARCIAL", ev, fix)
    return mk("B2", "Estatísticas verificáveis", "Substância", "FALHA", ev, fix)


def b3_authority(soup, text) -> Criterion:
    author = any(re.search(p, text) for p in AUTHOR_HINTS)
    schema_author = bool(soup.find(attrs={"itemprop": "author"})) or \
        bool(re.search(r'"@type"\s*:\s*"(Person|Organization)"', str(soup))) or \
        bool(soup.select('[rel="author"], .author, .byline, [class*="author"]'))
    experts = sum(len(re.findall(p, text, re.I)) for p in EXPERT_HINTS)
    ev = f"autoria no texto: {'sim' if author else 'não'}; marcação/classe de autor: {'sim' if schema_author else 'não'}; {experts} sinais de especialista"
    fix = ("Adicionar assinatura visível com credencial ('Por [Nome], [cargo] há X anos em [área]') "
           "e ao menos uma citação atribuída a especialista nomeado.")
    signals = sum([author, schema_author, experts >= 2])
    if signals >= 2:
        return mk("B3", "Autoridade", "Substância", "PASSA", ev)
    if signals == 1:
        return mk("B3", "Autoridade", "Substância", "PARCIAL", ev, fix)
    return mk("B3", "Autoridade", "Substância", "FALHA", ev, fix)


def b4_coverage(main, wc) -> Criterion:
    """Proxy: volume + amplitude de subtópicos + presença de FAQ."""
    h2 = len(main.find_all("h2"))
    h3 = len(main.find_all("h3"))
    faq = bool(re.search(r"\b(FAQ|perguntas frequentes|dúvidas comuns)\b",
                         main.get_text(" ", strip=True), re.I))
    ev = f"{wc} palavras, {h2} H2 + {h3} H3, FAQ: {'sim' if faq else 'não'}"
    fix = ("Ampliar cobertura: cada subpergunta do tema merece seu próprio H2/H3. "
           "Adicionar bloco de perguntas frequentes cobrindo objeções e casos de borda.")
    if wc >= 900 and h2 >= 4:
        return mk("B4", "Cobertura exaustiva", "Substância", "PASSA", ev, proxy=True)
    if wc >= 450 and h2 >= 2:
        return mk("B4", "Cobertura exaustiva", "Substância", "PARCIAL", ev, fix, proxy=True)
    return mk("B4", "Cobertura exaustiva", "Substância", "FALHA", ev, fix, proxy=True)


def b5_definitional(text, wc) -> Criterion:
    hits = sum(len(re.findall(p, text, re.I)) for p in DEFINITIONAL)
    per1k = hits / max(wc, 1) * 1000
    ev = f"{hits} construções definicionais ({per1k:.1f}/1k palavras)"
    fix = ('Incluir definições explícitas do tipo "X é um/uma..." logo após cada heading conceitual. '
           "LLMs extraem preferencialmente frases definicionais.")
    if hits >= 2 and per1k >= 1.5:
        return mk("B5", "Frases definicionais", "Substância", "PASSA", ev)
    if hits >= 1:
        return mk("B5", "Frases definicionais", "Substância", "PARCIAL", ev, fix)
    return mk("B5", "Frases definicionais", "Substância", "FALHA", ev, fix)


def c1_entities(text, wc) -> Criterion:
    hits = []
    for p in AMBIGUOUS_PT:
        hits += re.findall(p, text, re.I)
    per1k = len(hits) / max(wc, 1) * 1000
    ev = f"{len(hits)} referências ambíguas ({per1k:.1f}/1k palavras)"
    fix = ("Substituir pronomes e anáforas ('isso', 'ele', 'como vimos') pelo nome da entidade "
           "por extenso. Cada frase deve ser compreensível fora de contexto.")
    if per1k <= 8:
        return mk("C1", "Entidades explícitas", "Linguagem", "PASSA", ev)
    if per1k <= 18:
        return mk("C1", "Entidades explícitas", "Linguagem", "PARCIAL", ev, fix)
    return mk("C1", "Entidades explícitas", "Linguagem", "FALHA", ev, fix)


def c2_consistency(text, h1) -> Criterion:
    """Proxy: termos do H1 devem reaparecer com grafia estável ao longo do texto."""
    terms = [w for w in re.findall(r"\b[\wÀ-ÿ]{5,}\b", h1) if w.lower() not in PT_STOPWORDS]
    if not terms:
        return mk("C2", "Consistência terminológica", "Linguagem", "N/A", "H1 sem termos substantivos")
    low = text.lower()
    counts = {t: low.count(t.lower()) for t in terms}
    absent = [t for t, c in counts.items() if c == 0]
    ev = "; ".join(f"{t}: {c}x" for t, c in counts.items())
    fix = (f"Termos do H1 ausentes ou raros no corpo: {', '.join(absent) if absent else '—'}. "
           "Usar sempre o mesmo rótulo para o mesmo conceito, sem sinônimos rotativos.")
    if not absent and min(counts.values()) >= 2:
        return mk("C2", "Consistência terminológica", "Linguagem", "PASSA", ev)
    if len(absent) <= len(terms) / 2:
        return mk("C2", "Consistência terminológica", "Linguagem", "PARCIAL", ev, fix)
    return mk("C2", "Consistência terminológica", "Linguagem", "FALHA", ev, fix)


def c3_fanout(main, text) -> Criterion:
    """Proxy: diversidade de intenções interrogativas cobertas."""
    intents = {
        "o que": r"\bo que é\b|\bwhat is\b",
        "como": r"\bcomo\s+\w+", "por que": r"\bpor que\b|\bporque\b|\bwhy\b",
        "quando": r"\bquando\b|\bwhen\b", "quanto/custo": r"\bquanto custa\b|\bpreço\b|\bcusto\b",
        "qual/melhor": r"\bqual (o|a|é)\b|\bmelhor\b|\bvs\.?\b|\bcomparação\b",
        "vale a pena": r"\bvale a pena\b|\bvantagens\b|\bdesvantagens\b|\bprós\b|\bcontras\b",
        "passo a passo": r"\bpasso a passo\b|\bcomo fazer\b|\btutorial\b",
    }
    covered = [k for k, p in intents.items() if re.search(p, text, re.I)]
    ev = f"{len(covered)}/{len(intents)} intenções cobertas: {', '.join(covered) if covered else '—'}"
    missing = [k for k in intents if k not in covered]
    fix = (f"Ampliar o fan-out. Intenções não cobertas: {', '.join(missing)}. "
           "Adicionar seções que respondam essas variações da pergunta principal.")
    if len(covered) >= 5:
        return mk("C3", "Fan-out", "Linguagem", "PASSA", ev, proxy=True)
    if len(covered) >= 3:
        return mk("C3", "Fan-out", "Linguagem", "PARCIAL", ev, fix, proxy=True)
    return mk("C3", "Fan-out", "Linguagem", "FALHA", ev, fix, proxy=True)


def c4_declarative(text, wc) -> Criterion:
    hedges = []
    for p in HEDGE_PT:
        hedges += re.findall(p, text, re.I)
    sents = sentences(text)
    long_s = [s for s in sents if word_count(s) > 35]
    per1k = len(hedges) / max(wc, 1) * 1000
    ev = (f"{len(hedges)} hedges/floreios ({per1k:.1f}/1k), "
          f"{len(long_s)}/{len(sents)} frases acima de 35 palavras")
    sample = hedges[:4]
    fix = (f"Remover hedges{': ' + ', '.join(set(sample)) if sample else ''}. "
           "Trocar por afirmações diretas. Quebrar frases longas — trecho citável precisa ser curto e assertivo.")
    long_ratio = len(long_s) / max(len(sents), 1)
    if per1k <= 3 and long_ratio <= 0.15:
        return mk("C4", "Tom declarativo", "Linguagem", "PASSA", ev, proxy=True)
    if per1k <= 8 and long_ratio <= 0.3:
        return mk("C4", "Tom declarativo", "Linguagem", "PARCIAL", ev, fix, proxy=True)
    return mk("C4", "Tom declarativo", "Linguagem", "FALHA", ev, fix, proxy=True)


def d1_freshness(soup, text) -> Criterion:
    now = datetime.now().year
    years = [int(y) for y in re.findall(r"\b(19|20)\d{2}\b", text) for y in [re.search(r"\b(?:19|20)\d{2}\b", text).group()]] if False else []
    years = [int(m) for m in re.findall(r"\b((?:19|20)\d{2})\b", text)]
    years = [y for y in years if 1990 <= y <= now + 1]
    meta_dates = []
    for sel in ['meta[property="article:modified_time"]', 'meta[property="article:published_time"]',
                'meta[name="date"]', "time[datetime]"]:
        for n in soup.select(sel):
            v = n.get("content") or n.get("datetime")
            if v:
                meta_dates.append(v[:10])
    explicit = any(re.search(p, text, re.I) for p in DATE_PATTERNS[:3])
    newest = max(years) if years else None
    ev = (f"ano mais recente citado: {newest or '—'}; datas em meta/time: "
          f"{', '.join(meta_dates[:2]) if meta_dates else 'nenhuma'}")
    fix = ("Expor data de publicação e de última atualização visíveis no corpo e em "
           "<time datetime> / meta article:modified_time. Atualizar dados datados.")
    fresh = newest is not None and newest >= now - 1
    if fresh and (meta_dates or explicit):
        return mk("D1", "Frescor", "Manutenção", "PASSA", ev)
    if fresh or meta_dates:
        return mk("D1", "Frescor", "Manutenção", "PARCIAL", ev, fix)
    return mk("D1", "Frescor", "Manutenção", "FALHA", ev, fix)


# ---------------------------------------------------------------- orchestration


def guess_type(main, text, wc) -> str:
    if re.search(r"\b(FAQ|perguntas frequentes)\b", text, re.I) and wc < 900:
        return "FAQ"
    if main.find_all("table") and re.search(r"\b(comprar|preço|R\$|adicionar ao carrinho)\b", text, re.I):
        return "PDP / produto"
    if re.search(r"\bpasso a passo\b|\bcomo fazer\b|\btutorial\b", text, re.I):
        return "guia / how-to"
    if wc > 1200:
        return "artigo longo"
    if wc < 300:
        return "página curta / categoria"
    return "artigo"


def detect_question(main, h1) -> str:
    hs = [h.get_text(" ", strip=True) for h in main.find_all(["h1", "h2"])]
    for h in hs:
        if h.rstrip().endswith("?"):
            return h
    if h1:
        return f"(inferido do H1) {h1}"
    return "não detectável — falha em si"


def audit(url: str, timeout: int, wait: int, ua: str) -> AuditResult:
    (raw_html, rendered_html), err = fetch(url, timeout, wait, ua)
    if err or not rendered_html:
        return AuditResult(url=url, ok=False, error=err or "conteúdo vazio")

    soup = BeautifulSoup(rendered_html, "lxml")
    main, text = extract_main(rendered_html)
    wc = word_count(text)

    if wc < 50:
        return AuditResult(url=url, ok=False,
                           error=f"corpo com apenas {wc} palavras após limpeza — possível bloqueio, paywall ou conteúdo vazio")

    raw_wc = 0
    if raw_html:
        _, raw_text = extract_main(raw_html)
        raw_wc = word_count(raw_text)
    js_dep = round((1 - raw_wc / wc) * 100, 1) if wc else 0.0

    h1_tag = soup.find("h1")
    h1 = h1_tag.get_text(" ", strip=True) if h1_tag else ""
    title = soup.title.get_text(strip=True) if soup.title else ""
    if not h1:
        h1 = title

    paras = [p.get_text(" ", strip=True) for p in main.find_all("p") if word_count(p.get_text()) >= 8]

    criteria = [
        a1_answer_first(main, text, h1),
        a2_headings(main),
        a3_self_contained(main),
        a4_density(main),
        a5_extractable(main, wc),
        b1_original_data(text, wc),
        b2_verifiable(main, text),
        b3_authority(soup, text),
        b4_coverage(main, wc),
        b5_definitional(text, wc),
        c1_entities(text, wc),
        c2_consistency(text, h1),
        c3_fanout(main, text),
        c4_declarative(text, wc),
        d1_freshness(soup, text),
    ]

    return AuditResult(
        url=url, ok=True, h1=h1, title=title, word_count=wc,
        raw_words=raw_wc, rendered_words=wc, js_dependency=max(js_dep, 0.0),
        criteria=criteria,
        first_para=paras[0] if paras else "",
        target_question=detect_question(main, h1),
        content_type=guess_type(main, text, wc),
    )


