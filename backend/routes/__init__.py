"""HTTP route registration by product area."""

from typing import TYPE_CHECKING

from flask import Flask

from routes.chat import register_chat_routes
from routes.files import register_file_routes
from routes.settings import register_settings_routes

if TYPE_CHECKING:
    from composition import Services


def register_routes(app: Flask, services: "Services") -> None:
    """Register every HTTP route over the injected application graph."""
    register_file_routes(app, services)
    register_chat_routes(app, services)
    register_settings_routes(app, services)
