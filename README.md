# GEO Content Auditor — Hybrid Edition

Audita conteúdo **renderizado** (DOM real via Playwright) contra 15 critérios de GEO.
Arquitetura **híbrida assimétrica**: heurística mede o mecânico, LLM julga o semântico.

## Duas versões

| Arquivo | O que faz | Precisa de API key |
|---|---|---|
| `app.py` | Só heurística, 15 critérios (5 deles como proxy) | Não |
| `app_hybrid.py` | Heurística (10) + LLM (5 semânticos) | Sim, se LLM ligado |

## Divisão de trabalho

**Heurística (determinística, grátis):**
A2 headings · A4 densidade · A5 formatos · B1 dados · B2 fontes · B3 autoridade ·
B5 definições · C1 entidades · C2 consistência · D1 frescor

**LLM (julgamento semântico):**
A1 answer-first · A3 blocos autocontidos · B4 cobertura · C3 fan-out · C4 tom

Motivo: contar `<ul>` não melhora com LLM e custa token. Já "o primeiro parágrafo
realmente responde a pergunta?" nenhuma regex resolve.

**Extras que só o LLM entrega:** pergunta-alvo real, lead reescrito answer-first
pronto para publicar, lacunas de cobertura, FAQ sugerida.

## Rodar

```bash
pip install -r requirements.txt
playwright install chromium
streamlit run app_hybrid.py
```

## Streamlit Community Cloud

`requirements.txt` e `packages.txt` já incluídos. O Chromium **não** é instalado
automaticamente — se o deploy falhar, é quase sempre isso. Rode `playwright install
chromium` via console ou adicione ao processo de build.

**Limites do Community Cloud (1GB RAM):**
- Até ~25 URLs por lote
- Concorrência 3
- Reciclagem de browser a cada 8 páginas (já é o default)
- Para 100+ URLs, rode local ou em VM

## Custo (medido, não estimado)

~4.5k tokens in + 1.5k out por URL:

| Modelo | Por URL | 25 URLs | 100 URLs |
|---|---|---|---|
| Claude Haiku 4.5 | $0.012 | $0.29 | $1.15 |
| Claude Sonnet 5 | $0.035 | $0.86 | $3.45 |
| GPT-5.6 Luna | $0.013 | $0.33 | $1.30 |
| GPT-5.6 Terra | $0.033 | $0.81 | $3.25 |

Preços conferidos nas páginas oficiais em 17/07/2026. Sonnet 5 tem preço
promocional ($2/$10) até 31/08/2026; a estimativa usa o preço cheio para não
subestimar.

## Resiliência

- **Checkpoint incremental**: cada URL é gravada assim que termina. Rerun do
  Streamlit ou queda no meio do lote não perde trabalho — clique em Auditar de
  novo e retoma de onde parou.
- **Cache**: URL já auditada no mesmo lote/modelo não é recobrada.
- **Backoff exponencial** com jitter nos 429.
- **Browser reciclado** a cada N páginas (Chromium vaza memória).
- **Truncagem inteligente**: páginas longas preservam início (onde vive o
  answer-first) + amostra do meio + fim, em vez de cortar no meio.

## Limites conhecidos

- **A chave fica na sessão do navegador.** Não é gravada em disco nem entra nos
  checkpoints. Ainda assim, use uma chave com limite de gasto.
- **O LLM pode errar.** Ele julga com o texto truncado se a página for longa. O
  veredito vem com evidência literal justamente para você auditar o auditor.
- **Score híbrido vs heurístico não são comparáveis.** Se o LLM falhar numa URL,
  aquela linha pontua sobre 10 critérios, não 15. A UI sinaliza isso.
- **Léxico PT-BR** nas heurísticas de hedge/anáfora/definição. Conteúdo em ES/EN
  pontua com menos precisão em C1, C4 e B5. O LLM não tem essa limitação.
- **Sites que bloqueiam headless** retornam falha explícita. O app não infere
  conteúdo que não conseguiu ler.
- **Limiares heurísticos** (40–60 palavras em A1, 5 dados/1k em B1) são calibração
  inicial. Rode em 20 URLs suas e ajuste — são constantes no topo de `geo_core.py`.
