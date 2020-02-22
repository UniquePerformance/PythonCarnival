"""Main module."""
from datetime import datetime
import click


def info_header():
    now = datetime.now().strftime("%H:%M:%S")
    return click.style(f"[{now}] INFO - ", bold=True, fg="green")


def debug_info(text=None, color="bright_blue"):
    return click.echo(info_header() + "  " + click.style(str(text), bold=False, fg=color))
