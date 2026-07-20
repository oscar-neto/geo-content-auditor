"""
GEO Content Auditor — Hybrid Edition
Heurística (10 critérios determinísticos) + LLM (5 critérios semânticos).
"""

# geo_bootstrap fixa PLAYWRIGHT_BROWSERS_PATH e PRECISA vir antes de qualquer
# import que arraste o Playwright (geo_batch, geo_core).
import geo_bootstrap

import io
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import geo_batch as B
import geo_llm as L

st.set_page_config(page_title="GEO Auditor Híbrido", page_icon="◐", layout="wide")


# ---------------------------------------------------------------- bootstrap


@st.cache_resource(show_spinner=False)
def ensure_chromium():
    """
    Garante que o Chromium sobe ANTES de rodar o lote.

    Estratégia (ver geo_bootstrap): o browser é instalado em
    PLAYWRIGHT_BROWSERS_PATH, uma pasta dentro do projeto, para que o mesmo
    caminho valha no build e no runtime do Streamlit Cloud. Sem isso, o binário
    cai num cache que o processo de runtime não enxerga.

    Instala com `--with-deps` quando possível (traz libs de sistema) e valida
    subindo o browser de verdade — com a MESMA config do lote.

    Retorna (ok, log).
    """
    def _probe():
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                b = p.chromium.launch(
                    args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                          "--single-process", "--no-zygote"],
                )
                ver = b.version
                b.close()
            return True, f"Chromium {ver} operacional em {geo_bootstrap.BROWSERS_PATH}"
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:400]}"

    logs = [f"PLAYWRIGHT_BROWSERS_PATH = {geo_bootstrap.BROWSERS_PATH}"]

    if geo_bootstrap.chromium_binary_exists():
        ok, msg = _probe()
        if ok:
            return True, logs[0] + f"\n{msg}"
        logs.append(f"Binário presente mas probe falhou → {msg}")

    # instala no caminho fixado. O launch headless usa o chromium-headless-shell,
    # que é um binário SEPARADO — instalá-lo explicitamente é o que resolve o erro
    # "Executable doesn't exist at .../chromium_headless_shell-XXXX/...".
    # --with-deps precisa de root; se falhar, cai para versão sem deps (as libs
    # de sistema vêm do packages.txt no Streamlit Cloud).
    for cmd in (
        [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium", "chromium-headless-shell"],
        [sys.executable, "-m", "playwright", "install", "chromium", "chromium-headless-shell"],
        [sys.executable, "-m", "playwright", "install", "chromium"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                               env={**os.environ,
                                    "PLAYWRIGHT_BROWSERS_PATH": geo_bootstrap.BROWSERS_PATH})
            tail = (r.stderr or r.stdout or "").strip()[-500:]
            logs.append(f"$ {' '.join(cmd[3:])} → exit={r.returncode}\n  {tail}")
            if r.returncode == 0 and geo_bootstrap.chromium_binary_exists():
                break
        except subprocess.TimeoutExpired:
            logs.append(f"$ {' '.join(cmd[3:])} → timeout (600s)")
        except Exception as e:
            logs.append(f"$ {' '.join(cmd[3:])} → {type(e).__name__}: {e}")

    ok, msg = _probe()
    logs.append(("✓ " if ok else "✗ ") + msg)
    return ok, "\n".join(logs)


_boot_ok, _boot_log = ensure_chromium()

if not _boot_ok:
    st.error("**O Chromium não pôde ser instalado.** Sem ele, nenhuma URL pode ser "
             "renderizada — o app não tem como funcionar.")
    with st.expander("Ver log do diagnóstico", expanded=True):
        st.code(_boot_log, language="bash")
    st.markdown("""
**O que costuma resolver, em ordem:**

1. **Reboot do app.** No Streamlit Cloud: `Manage app` → `⋮` → `Reboot app`.
   O download às vezes falha por rede e passa na segunda.
2. **Confirme o `packages.txt`** na raiz do repo, com as libs de sistema do
   Chromium. Sem elas o binário baixa mas não sobe.
3. **Memória.** O Community Cloud dá 1GB. Se outro processo já consumiu, o
   launch falha. Reboot limpa.
4. Se persistir, rode local — o Playwright no Cloud é notoriamente frágil.
""")
    st.stop()


STATUS_ICON = {"PASSA": "🟢", "PARCIAL": "🟡", "FALHA": "🔴", "N/A": "⚪"}

UAS = {
    "Chrome (padrão)": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "ClaudeBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko) compatible; ClaudeBot/1.0; +claudebot@anthropic.com)",
    "GPTBot": "Mozilla/5.0 AppleWebKit/537.36 (KHTML, like Gecko) compatible; GPTBot/1.1; +https://openai.com/gptbot",
    "PerplexityBot": "Mozilla/5.0 (compatible; PerplexityBot/1.0; +https://perplexity.ai/perplexitybot)",
}

CRIT_ORDER = ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5",
              "C1", "C2", "C3", "C4", "D1"]


def band(s):
    return "🟢" if s >= 75 else ("🟡" if s >= 50 else "🔴")


# ---------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("⚙ Configuração")

    st.subheader("Análise semântica (LLM)")
    use_llm = st.toggle("Ativar LLM", value=True,
                        help="Sem LLM: 10 critérios heurísticos. Com LLM: +5 semânticos "
                             "(A1, A3, B4, C3, C4), trecho reescrito, gaps e FAQ sugerida.")

    provider = model = api_key = None
    if use_llm:
        provider_label = st.selectbox("Provedor", ["Anthropic (Claude)", "OpenAI"])
        provider = "anthropic" if provider_label.startswith("Anthropic") else "openai"

        model_label = st.selectbox("Modelo", list(L.MODELS[provider].keys()), index=1)
        model = L.MODELS[provider][model_label]

        api_key = st.text_input(
            f"API key ({'ANTHROPIC' if provider == 'anthropic' else 'OPENAI'})",
            type="password",
            help="A chave fica apenas na sessão do navegador. Não é gravada em disco "
                 "nem incluída nos checkpoints.",
        )

        pin, pout = L.PRICING.get(model, (0, 0))
        st.caption(f"💲 ${pin}/M in · ${pout}/M out")

        c1, c2 = st.columns(2)
        if c1.button("Testar chave", use_container_width=True):
            if not api_key:
                st.error("Informe a chave.")
            else:
                with st.spinner("Testando…"):
                    ok, msg = L.test_key(provider, api_key, model)
                (st.success if ok else st.error)(msg)
        if c2.button("Limpar cache", use_container_width=True):
            bid = st.session_state.get("batch_id")
            if bid:
                B.clear_ckpt(bid)
            st.session_state.pop("results", None)
            st.success("Cache limpo.")

    st.divider()
    st.subheader("Renderização")
    ua = UAS[st.selectbox("User-Agent", list(UAS.keys()))]
    timeout = st.slider("Timeout (s)", 10, 90, 35)
    wait = st.slider("Espera pós-load (s)", 0, 15, 3)

    st.subheader("Lote")
    concurrency = st.slider("Concorrência", 1, 8, 3,
                            help="Streamlit Cloud tem 1GB de RAM. Acima de 3 arrisca OOM.")
    recycle = st.slider("Reciclar browser a cada N páginas", 3, 25, 8,
                        help="Chromium vaza memória. Reciclar evita OOM em lotes longos.")
    max_words = st.slider("Máx. palavras enviadas ao LLM", 2000, 12000, 6000, step=1000)

    st.divider()
    st.caption("**Streamlit Cloud:** recomendado até 25 URLs por lote, "
               "concorrência 3. Para 100+, rode local.")

# ---------------------------------------------------------------- main

st.title("◐ GEO Content Auditor — Híbrido")
st.caption("Renderiza o DOM real (Playwright) e audita 15 critérios de GEO. "
           "Heurística mede o mecânico; LLM julga o semântico.")

urls_raw = st.text_area("URLs (uma por linha)", height=150,
                        placeholder="https://exemplo.com/artigo-1\nhttps://exemplo.com/artigo-2")
urls = [u.strip() for u in urls_raw.splitlines() if u.strip().startswith("http")]
urls = list(dict.fromkeys(urls))  # dedup preservando ordem

if urls:
    c1, c2, c3 = st.columns(3)
    c1.metric("URLs", len(urls))
    if use_llm and model:
        est = L.estimate_cost(model, 4500, 1500) * len(urls)
        c2.metric("Custo estimado", f"${est:.2f}",
                  help="Estimativa em ~4.5k tokens in / 1.5k out por URL. "
                       "Páginas longas custam mais.")
    mins = len(urls) * (6 + wait) / max(concurrency, 1) / 60 + (len(urls) * 6 / 60 if use_llm else 0)
    c3.metric("Tempo estimado", f"~{mins:.0f} min")

    if len(urls) > 25:
        st.warning(f"⚠ {len(urls)} URLs. No Streamlit Cloud (1GB RAM) o risco de OOM é real "
                   "acima de ~25. O checkpoint retoma de onde parou se cair — "
                   "basta clicar em Auditar de novo.")

col_a, col_b = st.columns([1, 4])
run = col_a.button("▶ Auditar", type="primary", use_container_width=True,
                   disabled=not urls)
resume = col_b.checkbox("Retomar do checkpoint (não re-audita URLs já feitas)", value=True)

if run:
    if use_llm and not api_key:
        st.error("LLM ativado mas sem API key. Informe a chave ou desative o LLM.")
        st.stop()

    batch_id = f"b_{abs(hash('|'.join(urls))) % 10**10}_{model or 'heur'}"
    st.session_state["batch_id"] = batch_id

    bar = st.progress(0.0)
    status = st.empty()
    t0 = time.time()

    def prog(done, total, msg):
        bar.progress(min(done / max(total, 1), 1.0))
        status.caption(f"{msg} · {time.time() - t0:.0f}s")

    try:
        results = B.audit_batch(
            urls, ua, timeout, wait, concurrency, recycle,
            use_llm, provider, api_key, model, max_words,
            batch_id, resume, prog,
        )
        st.session_state["results"] = results
        bar.empty()
        status.empty()
    except Exception as e:
        bar.empty()
        st.error(f"Falha no lote: {e}")
        st.info("O checkpoint preservou o que já foi auditado. Clique em Auditar "
                "novamente para retomar de onde parou.")
        st.stop()

results = st.session_state.get("results")

if results:
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    llm_failed = [r for r in ok if r.semantic and not r.llm_ok] + \
                 [r for r in ok if r.llm_error]

    if ok:
        st.header("Consolidado")
        m = st.columns(5)
        m[0].metric("Auditadas", len(ok))
        m[1].metric("Score médio", sum(r.score for r in ok) // len(ok))
        m[2].metric("Falhas de leitura", len(bad))
        m[3].metric("JS dep. média", f"{sum(r.js_dependency for r in ok)/len(ok):.0f}%")
        m[4].metric("Custo real", f"${sum(r.cost_usd for r in ok):.3f}")

        rows = []
        for r in ok:
            fails = [c["code"] for c in r.all_criteria if c["status"] == "FALHA"][:3]
            rows.append({
                "URL": r.url, "Score": r.score,
                "Heur.": r.score_heuristic,
                "Sem.": r.score_semantic if r.score_semantic is not None else "—",
                "Tipo": r.content_type, "Palavras": r.word_count,
                "JS": f"{r.js_dependency:.0f}%",
                "Top falhas": ", ".join(fails) or "—",
            })
        df = pd.DataFrame(rows).sort_values("Score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # falhas sistêmicas
        st.subheader("Falhas sistêmicas")
        tally = {}
        for r in ok:
            for c in r.all_criteria:
                if c["status"] in ("FALHA", "PARCIAL"):
                    tally.setdefault((c["code"], c["name"], c.get("source", "")), []).append(c["status"])
        sysf = []
        for (code, name, src), sts in tally.items():
            rate = len(sts) / len(ok)
            if rate > 0.5:
                sysf.append({"Critério": f"{code} {name}", "Fonte": src,
                             "% com problema": round(rate * 100),
                             "% falha total": round(sts.count("FALHA") / len(ok) * 100)})
        if sysf:
            st.dataframe(pd.DataFrame(sysf).sort_values("% com problema", ascending=False),
                         use_container_width=True, hide_index=True)
            st.info("Acima de 50% = problema de **template ou processo editorial**, "
                    "não de peça isolada. Corrigir na origem rende mais que URL a URL.")
        else:
            st.success("Nenhum critério falha em mais de 50% das URLs.")

        # canibalização
        qs = {}
        for r in ok:
            q = (r.target_question or "").strip().lower()
            if q and "não detectável" not in q:
                qs.setdefault(re.sub(r"[^\wà-ÿ ]", "", q), []).append(r.url)
        dupes = {k: v for k, v in qs.items() if len(v) > 1}
        if dupes:
            st.subheader("Sobreposição de pergunta-alvo")
            for k, v in dupes.items():
                st.warning(f"**{k}** → {len(v)} URLs competindo:\n" +
                           "\n".join(f"- {u}" for u in v))

        # export
        det = []
        for r in ok:
            for c in r.all_criteria:
                det.append({"URL": r.url, "Score": r.score, "Tipo": r.content_type,
                            "Critério": c["code"], "Nome": c["name"],
                            "Fonte": c.get("source", ""), "Status": c["status"],
                            "Evidência": c["evidence"],
                            "Justificativa": c.get("rationale", ""),
                            "Correção": c["fix"]})
        acts = [{"URL": r.url, "Pergunta-alvo": r.target_question,
                 "Lead reescrito": r.rewritten_lead,
                 "Gaps": " | ".join(r.coverage_gaps),
                 "FAQ sugerida": " | ".join(r.suggested_faq)} for r in ok if r.llm_ok]

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="Resumo", index=False)
            pd.DataFrame(det).to_excel(w, sheet_name="Detalhado", index=False)
            if sysf:
                pd.DataFrame(sysf).to_excel(w, sheet_name="Sistêmicas", index=False)
            if acts:
                pd.DataFrame(acts).to_excel(w, sheet_name="Acionáveis", index=False)
            if bad:
                pd.DataFrame([{"URL": r.url, "Erro": r.error} for r in bad]).to_excel(
                    w, sheet_name="Falhas", index=False)
        st.download_button("⬇ Exportar auditoria (.xlsx)", buf.getvalue(),
                           f"geo_audit_{datetime.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if llm_failed:
        st.warning(f"⚠ {len(llm_failed)} URL(s) auditadas só pela heurística — o LLM falhou. "
                   "Score dessas linhas cobre 10 critérios, não 15, e não é comparável "
                   "diretamente com as demais.")
        for r in llm_failed[:5]:
            st.caption(f"- {r.url}: {r.llm_error[:120]}")

    if bad:
        st.header("Falhas de leitura")
        for r in bad:
            st.error(f"**{r.url}** — {r.error}")
        st.caption("Página que um browser headless não lê dificilmente é lida por crawler "
                   "de IA. Este é o achado mais grave possível — investigue antes dos demais.")

    # ---- por URL
    st.header("Diagnóstico por URL")
    for r in sorted(ok, key=lambda x: x.score):
        with st.expander(f"{band(r.score)} **{r.score}/100** — {r.url}"):
            m = st.columns(5)
            m[0].metric("Híbrido", f"{r.score}/100")
            m[1].metric("Heurístico", f"{r.score_heuristic}/100")
            m[2].metric("Semântico", f"{r.score_semantic}/100" if r.score_semantic is not None else "—")
            m[3].metric("Palavras", r.word_count)
            m[4].metric("JS dep.", f"{r.js_dependency:.0f}%")

            if r.js_dependency > 40:
                st.error(f"⚠ {r.js_dependency:.0f}% do conteúdo só existe após JS. "
                         "Crawlers que não renderizam veem uma página quase vazia. "
                         "**Este problema anula todos os outros critérios** — resolva primeiro.")

            st.markdown(f"**H1:** {r.h1 or '—'}")
            if r.target_question:
                st.markdown(f"**Pergunta-alvo:** {r.target_question}")

            crit = sorted(r.all_criteria, key=lambda c: CRIT_ORDER.index(c["code"])
                          if c["code"] in CRIT_ORDER else 99)
            st.dataframe(pd.DataFrame([{
                "": STATUS_ICON[c["status"]],
                "Critério": f"{c['code']} {c['name']}",
                "Fonte": c.get("source", ""),
                "Status": c["status"],
                "Evidência": c["evidence"],
                "Correção": c["fix"] or "—",
            } for c in crit]), use_container_width=True, hide_index=True)

            prio = sorted([c for c in crit if c["status"] in ("FALHA", "PARCIAL")],
                          key=lambda c: (0 if c["status"] == "FALHA" else 1))[:3]
            if prio:
                st.markdown("**Top 3 ações prioritárias**")
                for i, c in enumerate(prio, 1):
                    st.markdown(f"{i}. **{c['code']} {c['name']}** ({c['status']}) — {c['fix']}")

            if r.rewritten_lead:
                st.markdown("**Lead reescrito (answer-first, pronto para publicar)**")
                st.code(r.rewritten_lead, language=None)

            cc = st.columns(2)
            if r.coverage_gaps:
                cc[0].markdown("**Lacunas de cobertura**\n" +
                               "\n".join(f"- {g}" for g in r.coverage_gaps))
            if r.suggested_faq:
                cc[1].markdown("**FAQ sugerida**\n" +
                               "\n".join(f"- {q}" for q in r.suggested_faq))

            if r.llm_error:
                st.warning(f"LLM falhou nesta URL: {r.llm_error}")

st.divider()
st.caption("Heurística: A2, A4, A5, B1, B2, B3, B5, C1, C2, D1 · "
           "LLM: A1, A3, B4, C3, C4 · "
           "Score híbrido = média ponderada de todos os critérios avaliáveis.")
