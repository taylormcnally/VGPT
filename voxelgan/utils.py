import torch
from rich import print, pretty
from rich.console import Console
from rich.markdown import Markdown
import sys

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



def get_gpu_info():
    """
    Get GPU Hardware info. GPUs are preferred, warn if not available.
    """
    #get pytorch gpu info
    if torch.cuda.is_available():
        print_success(f'Found {torch.cuda.device_count()} GPUs')
    else:
        print_warning('No hardware accelerator detected! Are you sure you want to continue?')
        user_input = input('Continue? (y/n) ')
        if user_input == 'y':
            print_warning('Continuing without hardware acceleration.')
        else:
            print_error('Exiting.')
            sys.exit(1)


#test locally
if __name__ == '__main__':
    get_gpu_info()



