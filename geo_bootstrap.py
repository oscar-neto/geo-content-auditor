"""
geo_bootstrap — DEVE ser importado ANTES de qualquer import do Playwright.

O problema no Streamlit Community Cloud (e Render, Vercel, etc.): o browser é
baixado num cache que o processo de runtime não enxerga, ou não é baixado. O
erro clássico é:

    BrowserType.launch: Executable doesn't exist at
    /home/appuser/.cache/ms-playwright/chromium_headless_shell-XXXX/...

A solução robusta (não depende do cache default): fixar PLAYWRIGHT_BROWSERS_PATH
para uma pasta DENTRO do projeto e instalar ali. Assim o mesmo caminho vale no
build e no runtime, e o Playwright sempre acha o binário.

Como o Playwright lê essa variável no momento do import, este módulo precisa ser
o PRIMEIRO import de qualquer arquivo que use Playwright — antes de
`playwright`, `geo_core` ou `geo_batch`.
"""

import os
from pathlib import Path

# pasta dentro do projeto (não em ~/.cache, que some entre build e runtime)
_BROWSERS_DIR = Path(__file__).parent / ".pw-browsers"
_BROWSERS_DIR.mkdir(exist_ok=True)

# só define se ainda não veio do ambiente — respeita override manual
os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(_BROWSERS_DIR.resolve()))

BROWSERS_PATH = os.environ["PLAYWRIGHT_BROWSERS_PATH"]


def chromium_binary_exists() -> bool:
    """True se há um executável de chromium (completo OU headless shell) instalado."""
    root = Path(BROWSERS_PATH)
    if not root.exists():
        return False
    patterns = [
        "chromium-*/chrome-linux/chrome",
        "chromium_headless_shell-*/chrome-linux/headless_shell",
        "chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell",
    ]
    return any(any(root.glob(p)) for p in patterns)
