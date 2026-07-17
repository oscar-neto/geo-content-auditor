#!/bin/bash
# Streamlit Community Cloud não roda `playwright install` sozinho.
# Este script é chamado pelo app na primeira execução (ver geo_batch bootstrap)
# ou pode ser rodado manualmente.
playwright install chromium
