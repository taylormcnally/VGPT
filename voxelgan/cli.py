from rich import print, pretty
from rich.console import Console
from rich.markdown import Markdown


console = Console()


def print_error(msg):
    console.print("[bold red]{}[/bold red]".format(msg))

def print_success(msg):
    console.print("[bold green]{}[/bold green]".format(msg))

def print_warning(msg):
    console.print("[bold yellow]{}[/bold yellow]".format(msg))

def print_markdown(file):
    with open(file) as readme:
        markdown = Markdown(readme.read())
    console.print(markdown)

def print_working(msg):
    console.status("[bold green]Working on {}...".format(msg))

def print_done(msg):
    console.log("[bold green]{} Done.".format(msg))
