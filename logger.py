# logger.py
from rich.console import Console
from rich.table   import Table
from rich.theme   import Theme

custom_theme = Theme({
    "ok"     : "bold green",
    "fail"   : "bold red",
    "title"  : "bold cyan",
    "number" : "bold yellow"
})
console = Console(theme=custom_theme)

def log_scenario(idx, score, elapsed_s, ε, failure):
    style = "fail" if failure else "ok"
    emoji = ":x:" if failure else ":white_check_mark:"
    console.print(
        f"{emoji}  Scenario [number]{idx}"
        f" | score: [number]{score:+.1f}"
        f" | steps: [number]{elapsed_s}"
        f" | ε: {ε:.3f}",
        style=style
    )

def log_loss(idx, losses, epsilon):
    console.print(
        f"Scenario [number]{idx}"
        f" | [LEFT] loss: [number]{losses[0]:.4f}"
        f" | [STRAIGHT] loss: [number]{losses[1]:.4f}"
        f" | [RIGHT] loss: [number]{losses[2]:.4f}",
        f" | [number]ε: {epsilon:.3f}",
        style="title"
    )