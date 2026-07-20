#!/bin/bash
# Instala o Chromium E o headless shell no caminho do projeto.
# O launch headless do Playwright usa o chromium-headless-shell, que é um
# binário SEPARADO — instalar só "chromium" pode não trazê-lo, e aí o lote
# falha com "Executable doesn't exist at .../chromium_headless_shell-XXXX/...".
export PLAYWRIGHT_BROWSERS_PATH="$(dirname "$0")/.pw-browsers"
playwright install --with-deps chromium chromium-headless-shell || \
  playwright install chromium chromium-headless-shell
