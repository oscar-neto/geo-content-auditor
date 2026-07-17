"""
Camada LLM do GEO Auditor — avalia apenas os 5 critérios semânticos.
Heurística cuida dos 10 determinísticos (ver geo_core.py).

Providers: Anthropic (Claude) e OpenAI. Sem SDK: HTTP puro via requests,
para não arrastar dependências e para tratar rate limit de forma explícita.
"""

import hashlib
import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

# ---------------------------------------------------------------- models

MODELS = {
    "anthropic": {
        "Claude Opus 4.8 — máxima qualidade": "claude-opus-4-8",
        "Claude Sonnet 5 — recomendado (custo/qualidade)": "claude-sonnet-5",
        "Claude Haiku 4.5 — mais barato": "claude-haiku-4-5-20251001",
    },
    "openai": {
        "GPT-5.6 Sol — máxima qualidade": "gpt-5.6-sol",
        "GPT-5.6 Terra — recomendado (custo/qualidade)": "gpt-5.6-terra",
        "GPT-5.6 Luna — mais barato": "gpt-5.6-luna",
    },
}

ENDPOINTS = {
    "anthropic": "https://api.anthropic.com/v1/messages",
    "openai": "https://api.openai.com/v1/chat/completions",
}

# US$ por 1M tokens (input, output). Verificado em 17/07/2026 nas páginas
# oficiais de pricing. Preços mudam — confira antes de estimar orçamento grande.
# Sonnet 5: preço promocional de introdução ($2/$10) vale até 31/08/2026;
# usamos o preço cheio ($3/$15) para a estimativa não subestimar o custo.
PRICING = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "gpt-5.6-sol": (5.00, 30.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-luna": (1.00, 6.00),
}

SEMANTIC_CRITERIA = ["A1", "A3", "B4", "C3", "C4"]


@dataclass
class LLMVerdict:
    code: str
    name: str
    status: str          # PASSA | PARCIAL | FALHA | N/A
    evidence: str = ""
    fix: str = ""
    rationale: str = ""


@dataclass
class LLMResult:
    ok: bool
    error: str = ""
    verdicts: list = field(default_factory=list)
    target_question: str = ""
    rewritten_lead: str = ""
    coverage_gaps: list = field(default_factory=list)
    suggested_faq: list = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    raw: str = ""


# ---------------------------------------------------------------- prompt

SYSTEM = """Você é um auditor sênior de conteúdo especializado em GEO (Generative Engine Optimization) — a prática de otimizar conteúdo para ser citado por LLMs (ChatGPT, Claude, Perplexity, AI Overviews).

Você avalia APENAS 5 critérios semânticos. Os critérios mecânicos (contagem de listas, densidade de parágrafo, datas) já foram medidos por heurística e NÃO são sua responsabilidade.

Você é rigoroso. Conteúdo de marca brasileiro tipicamente falha em substância e tom — não seja generoso. "PASSA" significa que o conteúdo realmente serve como fonte citável por uma IA, não que ele é "aceitável".

Responda EXCLUSIVAMENTE com um objeto json válido. Sem markdown, sem crases, sem preâmbulo. A saída inteira deve ser json parseável."""

RUBRIC = """
## CRITÉRIOS

**A1 — Answer-first**
A pergunta central da página é respondida de forma completa e autossuficiente nas primeiras ~40-60 palavras do corpo, ANTES de qualquer contextualização, história ou introdução.
- PASSA: quem lê só o primeiro parágrafo já tem a resposta útil.
- PARCIAL: responde, mas enterrado depois de contexto, ou incompleto.
- FALHA: abre com introdução genérica, história, ou não responde nada.
IMPORTANTE: tamanho correto NÃO basta. Um parágrafo de 50 palavras cheio de keywords que não responde a pergunta é FALHA.

**A3 — Blocos autocontidos**
Cada seção faz sentido lida isoladamente, sem depender do que veio antes. LLMs citam trechos, não páginas.
- PASSA: qualquer seção pode ser extraída e citada sozinha.
- PARCIAL: algumas seções dependem de contexto anterior.
- FALHA: seções abrem com anáforas ("isso", "como vimos") ou só fazem sentido em sequência.

**B4 — Cobertura exaustiva**
O tema é resolvido por completo. Identifique o que um leitor ainda perguntaria depois de ler.
- PASSA: sem lacunas óbvias; o leitor não precisa de outra fonte.
- PARCIAL: cobre o básico, faltam casos de borda/objeções.
- FALHA: superficial; lacunas centrais.
Julgue pela profundidade real, não pelo tamanho.

**C3 — Fan-out**
Cobre variações de linguagem natural e perguntas adjacentes que usuários realmente fazem, não só a keyword-cabeça.
- PASSA: antecipa o leque de perguntas relacionadas.
- PARCIAL: cobre algumas variações.
- FALHA: monotemático na keyword principal.

**C4 — Tom declarativo**
Objetivo e assertivo. Sem hedge, floreio, enrolação ou "encheção de linguiça" que dilua o trecho citável.
- PASSA: afirmações diretas, extraíveis como citação.
- PARCIAL: assertivo em partes, com trechos diluídos.
- FALHA: hedge constante, frases longas e vagas, texto que não afirma nada citável.

## SAÍDA (JSON estrito)

{
  "target_question": "a pergunta que esta página tenta responder; se indetectável, escreva exatamente: NÃO DETECTÁVEL",
  "content_type": "artigo | guia | FAQ | PDP | categoria | institucional",
  "verdicts": [
    {
      "code": "A1",
      "status": "PASSA | PARCIAL | FALHA | N/A",
      "evidence": "trecho literal do conteúdo que prova o veredito, máximo 15 palavras",
      "rationale": "por que este status, em 1 frase direta",
      "fix": "correção CONCRETA E PRONTA PARA COLAR. Não escreva conselho abstrato como 'melhore a introdução'. Escreva o texto sugerido."
    }
    // ... um objeto para cada: A1, A3, B4, C3, C4
  ],
  "rewritten_lead": "reescreva o primeiro parágrafo em formato answer-first, 40-60 palavras, pronto para publicar. Use os fatos que existem no conteúdo; não invente dados. Se não houver fatos suficientes, escreva: DADOS INSUFICIENTES",
  "coverage_gaps": ["lacuna 1", "lacuna 2", "lacuna 3"],
  "suggested_faq": ["pergunta que deveria estar na página e não está", "..."]
}

REGRAS:
- Se usar N/A, justifique no rationale.
- evidence deve ser trecho LITERAL do conteúdo, máximo 15 palavras. Se não houver trecho aplicável (ex: ausência), descreva a ausência.
- NUNCA invente dados, números ou fontes em rewritten_lead. Use apenas o que está no conteúdo.
- fix deve ser texto colável, não instrução.
"""


def build_user_prompt(url, h1, content_type_guess, heuristic_signals, body_text):
    return f"""Audite o conteúdo abaixo contra os 5 critérios semânticos.

URL: {url}
H1: {h1 or "(ausente)"}
Tipo estimado por heurística: {content_type_guess}

Sinais já medidos por heurística (contexto de apoio — não reavalie, apenas use como informação):
{heuristic_signals}

{RUBRIC}

## CONTEÚDO

<content>
{body_text}
</content>"""


# ---------------------------------------------------------------- truncation


def smart_truncate(text: str, max_words: int = 6000) -> tuple:
    """
    Truncagem inteligente: preserva início (onde vive answer-first) e o resto
    amostrado, para não destruir julgamento de A1/B4.
    Retorna (texto, foi_truncado).
    """
    words = text.split()
    if len(words) <= max_words:
        return text, False

    head_n = int(max_words * 0.55)
    tail_n = int(max_words * 0.20)
    mid_n = max_words - head_n - tail_n

    head = " ".join(words[:head_n])
    mid_pool = words[head_n:-tail_n] if tail_n else words[head_n:]
    step = max(1, len(mid_pool) // max(mid_n, 1))
    mid = " ".join(mid_pool[::step][:mid_n])
    tail = " ".join(words[-tail_n:]) if tail_n else ""

    out = (f"{head}\n\n[...trecho intermediário amostrado — página truncada para caber no contexto...]\n\n"
           f"{mid}\n\n[...]\n\n{tail}")
    return out, True


# ---------------------------------------------------------------- api calls


class RateLimitError(Exception):
    pass


def _call_anthropic(api_key, model, system, user, timeout=120):
    r = requests.post(
        ENDPOINTS["anthropic"],
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 3000,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=timeout,
    )
    if r.status_code == 429:
        raise RateLimitError(r.text[:200])
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
    u = d.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


def _call_openai(api_key, model, system, user, timeout=120):
    """
    Modelos novos da OpenAI (GPT-5.x, o-series) trocaram `max_tokens` por
    `max_completion_tokens` e rejeitam o antigo com HTTP 400. Modelos antigos
    (gpt-4o, gpt-4.1) só aceitam o antigo. Tentamos o novo e, se o servidor
    reclamar do parâmetro, refazemos com o legado — assim funciona nos dois.
    """
    payload = {
        "model": model,
        "max_completion_tokens": 3000,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
    }

    # A OpenAI exige a palavra "json" literal nas mensagens para aceitar
    # response_format=json_object, e a checagem é sensível a caixa: "JSON" no
    # prompt não satisfaz. Em vez de depender do fraseado (que qualquer edição
    # futura do prompt pode quebrar), garantimos aqui.
    joined = (system + " " + user).lower()
    if "json" not in joined:
        payload["messages"][0]["content"] = (
            system + "\n\nFormato de saída: responda somente com um objeto json válido."
        )

    def _post(body):
        return requests.post(
            ENDPOINTS["openai"],
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )

    r = _post(payload)

    # fallback para modelos legados que só conhecem max_tokens
    if r.status_code == 400 and "max_completion_tokens" in r.text and "nsupported" in r.text:
        legacy = dict(payload)
        legacy.pop("max_completion_tokens")
        legacy["max_tokens"] = 3000
        r = _post(legacy)

    if r.status_code == 429:
        raise RateLimitError(r.text[:200])
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

    d = r.json()
    choice = d["choices"][0]
    text = choice["message"].get("content") or ""

    # modelos com reasoning podem estourar o teto só com tokens de raciocínio,
    # devolvendo content vazio — falha silenciosa que vira "JSON inválido" lá na frente
    if not text.strip():
        raise RuntimeError(
            f"Modelo retornou conteúdo vazio (finish_reason={choice.get('finish_reason')}). "
            "Provável estouro do limite de tokens com raciocínio interno."
        )

    u = d.get("usage", {})
    return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def call_with_retry(provider, api_key, model, system, user, max_retries=4):
    fn = _call_anthropic if provider == "anthropic" else _call_openai
    last = None
    for attempt in range(max_retries):
        try:
            return fn(api_key, model, system, user)
        except RateLimitError as e:
            last = e
            # backoff exponencial com jitter — 429 em lote é esperado
            sleep = (2 ** attempt) + random.uniform(0, 1.5)
            time.sleep(min(sleep, 30))
        except requests.Timeout as e:
            last = e
            time.sleep(2 ** attempt)
        except Exception as e:
            raise
    raise RuntimeError(f"Falhou após {max_retries} tentativas: {last}")


# ---------------------------------------------------------------- parsing


def parse_json_loose(text: str) -> dict:
    """LLM às vezes envolve em crases apesar da instrução. Tolerar."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # última tentativa: extrair o maior bloco {...}
        m = re.search(r"\{.*\}", t, re.S)
        if not m:
            raise
        return json.loads(m.group())


CRIT_NAMES = {
    "A1": "Answer-first", "A3": "Blocos autocontidos", "B4": "Cobertura exaustiva",
    "C3": "Fan-out", "C4": "Tom declarativo",
}

VALID_STATUS = {"PASSA", "PARCIAL", "FALHA", "N/A"}


def estimate_cost(model, tin, tout) -> float:
    pin, pout = PRICING.get(model, (0, 0))
    return tin / 1e6 * pin + tout / 1e6 * pout


def judge(provider, api_key, model, url, h1, type_guess, signals, body_text,
          max_words=6000) -> LLMResult:
    body, truncated = smart_truncate(body_text, max_words)
    if truncated:
        signals += "\n[AVISO: conteúdo truncado para caber no contexto; julgue com o que há]"

    user = build_user_prompt(url, h1, type_guess, signals, body)

    try:
        text, tin, tout = call_with_retry(provider, api_key, model, SYSTEM, user)
    except Exception as e:
        return LLMResult(ok=False, error=str(e)[:300])

    try:
        d = parse_json_loose(text)
    except Exception as e:
        return LLMResult(ok=False, error=f"JSON inválido do modelo: {str(e)[:150]}",
                         raw=text[:500], tokens_in=tin, tokens_out=tout)

    verdicts = []
    seen = set()
    for v in d.get("verdicts", []):
        code = str(v.get("code", "")).upper().strip()
        if code not in CRIT_NAMES or code in seen:
            continue
        seen.add(code)
        status = str(v.get("status", "")).upper().strip()
        if status not in VALID_STATUS:
            status = "N/A"
        verdicts.append(LLMVerdict(
            code=code, name=CRIT_NAMES[code], status=status,
            evidence=str(v.get("evidence", ""))[:200],
            fix=str(v.get("fix", ""))[:800],
            rationale=str(v.get("rationale", ""))[:300],
        ))

    # critério ausente na resposta vira N/A explícito, não some
    for code in SEMANTIC_CRITERIA:
        if code not in seen:
            verdicts.append(LLMVerdict(code=code, name=CRIT_NAMES[code], status="N/A",
                                       rationale="modelo não retornou veredito para este critério"))
    verdicts.sort(key=lambda v: SEMANTIC_CRITERIA.index(v.code))

    return LLMResult(
        ok=True,
        verdicts=verdicts,
        target_question=str(d.get("target_question", ""))[:300],
        rewritten_lead=str(d.get("rewritten_lead", ""))[:1200],
        coverage_gaps=[str(x)[:200] for x in (d.get("coverage_gaps") or [])][:6],
        suggested_faq=[str(x)[:200] for x in (d.get("suggested_faq") or [])][:8],
        tokens_in=tin, tokens_out=tout,
        cost_usd=estimate_cost(model, tin, tout),
        raw=text[:2000],
    )


def content_hash(url, model, text) -> str:
    return hashlib.sha256(f"{url}|{model}|{text[:5000]}".encode()).hexdigest()[:16]


def test_key(provider, api_key, model) -> tuple:
    """Valida a chave com uma chamada mínima. Retorna (ok, mensagem)."""
    try:
        fn = _call_anthropic if provider == "anthropic" else _call_openai
        txt, tin, tout = fn(api_key, model, "Responda apenas: OK", "Diga OK")
        return True, f"Chave válida ({tin + tout} tokens no teste)"
    except RateLimitError:
        return True, "Chave válida (rate limit atingido no teste, mas autenticou)"
    except Exception as e:
        return False, str(e)[:200]
