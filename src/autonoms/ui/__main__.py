"""Start the web interface:  uv run python -m autonoms.ui   (the API must be running too)."""

from nicegui import ui

from autonoms.config import get_settings
from autonoms.ui.api import Api
from autonoms.ui.app import register

settings = get_settings()
register(Api(settings.api_url))
# 127.0.0.1: this page can publish pull requests and has no login, so it must not be reachable from your network.
ui.run(host="127.0.0.1", port=settings.ui_port, title="Autonoms", reload=False, show=False)
