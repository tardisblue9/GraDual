"""Minimal mock-browser primitives for FlowBench Web scenario.

Goal: just enough to exercise a G2 "form field hijack" attack — the page's
visible text contains a hidden directive that tries to override a form field
value. No real rendering, no JavaScript, no URL fetching.

Env:
    WebEnv.pages         : dict[url -> Page]  — pre-authored content
    WebEnv.current_url   : str | None
    WebEnv.form_fields   : dict[str, str]     — values entered by agent
    WebEnv.submissions   : list[dict]         — submit_form call records

Tools (all take AnnotatedWebEnv):
    goto(url)              : navigate to a known page
    get_page_text()        : return current page's visible_text (this is the
                              injection surface)
    fill_form_field(name, value)
    submit_form()          : freeze form_fields into a submission record and
                              clear form_fields for next page
"""
from __future__ import annotations

from typing import Annotated, Any
from pydantic import BaseModel, Field

from agentdojo.functions_runtime import Depends


class Page(BaseModel):
    url:          str
    title:        str
    visible_text: str
    # Optional: a short form schema the page hosts (used by the benign form
    # fill task). In the pilot we only care about the field NAMES; the agent
    # still has to decide values from the user query + visible_text.
    form_fields:  list[str] = Field(default_factory=list)


class WebEnv(BaseModel):
    pages:        dict[str, Page]
    current_url:  str | None = None
    # Agent-authored form values for the current page (cleared on submit).
    form_values:  dict[str, str] = Field(default_factory=dict)
    submissions:  list[dict[str, Any]] = Field(default_factory=list)


AnnotatedWebEnv = Annotated[WebEnv, Depends("webenv")]


def goto(webenv: AnnotatedWebEnv, url: str) -> str:
    """Navigate the browser to ``url``. Returns the page title or a 404 message.

    :param url: The full URL to navigate to, e.g., "https://acme.com/register".
    """
    if url not in webenv.pages:
        return f"404: no page found at {url}"
    webenv.current_url = url
    webenv.form_values = {}
    return f"Loaded: {webenv.pages[url].title}"


def get_page_text(webenv: AnnotatedWebEnv) -> str:
    """Return the visible text of the current page. If no page is loaded,
    returns an error string.

    Note: this is the primary surface on which an attacker may have planted
    an injected directive (e.g., a hidden <INFORMATION> block in the text).
    """
    if not webenv.current_url:
        return "Error: no page loaded. Call goto() first."
    return webenv.pages[webenv.current_url].visible_text


def fill_form_field(webenv: AnnotatedWebEnv, name: str, value: str) -> str:
    """Enter ``value`` into the form field named ``name`` on the current
    page. Agent must call this for each field before submit_form().

    :param name: Field name (e.g., "email", "username").
    :param value: The value to enter.
    """
    if not webenv.current_url:
        return "Error: no page loaded."
    page = webenv.pages[webenv.current_url]
    if page.form_fields and name not in page.form_fields:
        return f"Error: no such field '{name}' on this page. Known: {page.form_fields}"
    webenv.form_values[name] = value
    return f"ok: {name}='{value}'"


def submit_form(webenv: AnnotatedWebEnv) -> str:
    """Submit the currently-entered form values. The submission is recorded
    in webenv.submissions; form_values is cleared for a subsequent fill cycle.
    """
    if not webenv.current_url:
        return "Error: no page loaded."
    if not webenv.form_values:
        return "Error: no fields filled; call fill_form_field() first."
    record = {
        "url":   webenv.current_url,
        "values": dict(webenv.form_values),
    }
    webenv.submissions.append(record)
    webenv.form_values = {}
    return f"submitted to {record['url']}: {record['values']}"


# All web tools in one list for runtime registration.
WEB_TOOLS = [goto, get_page_text, fill_form_field, submit_form]
