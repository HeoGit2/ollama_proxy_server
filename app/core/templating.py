# app/core/templating.py
"""Shared helpers for rendering the admin UI templates and flashing messages."""
from typing import Any, Dict, Optional

from fastapi import Request, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.core.security import get_or_create_csrf_token

templates = Jinja2Templates(directory="app/templates")


def get_template_context(request: Request) -> Dict[str, Any]:
    """Context shared by every template."""
    return {
        "request": request,
        "is_redis_connected": request.app.state.redis is not None,
        "bootstrap_settings": settings,
    }


def render_template(request: Request, template_name: str, **extra_context: Any) -> HTMLResponse:
    """Renders a template with the shared context, a CSRF token and any extra context."""
    context = get_template_context(request)
    context["csrf_token"] = get_or_create_csrf_token(request.session)
    context.update(extra_context)
    return templates.TemplateResponse(template_name, context)


def redirect_to(request: Request, route_name: str, **path_params: Any) -> RedirectResponse:
    """Redirects to a named route using the POST/redirect/GET status code."""
    return RedirectResponse(
        url=request.url_for(route_name, **path_params),
        status_code=status.HTTP_303_SEE_OTHER,
    )


def flash(request: Request, message: str, category: str = "info") -> None:
    """
    FIX: Re-assign list to session to avoid mutation issues with modern SessionMiddleware.
    """
    messages = request.session.get("_messages", [])
    messages.append({"message": message, "category": category})
    request.session["_messages"] = messages


def flash_result(request: Request, result: Dict[str, Any], message: Optional[str] = None) -> None:
    """Flashes a backend operation result, choosing the category from its 'success' flag."""
    flash(request, message or result["message"], "success" if result["success"] else "error")


def get_flashed_messages(request: Request):
    return request.session.pop("_messages", [])


templates.env.globals["get_flashed_messages"] = get_flashed_messages
