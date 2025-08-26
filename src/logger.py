import sys
from rich.console import Console
from rich.theme   import Theme

custom_theme = Theme({
    "ok": "bold green",
    "fail": "bold red",
    "title": "bold cyan",
    "number": "bold yellow"
})

console = Console(
            theme=custom_theme,
            force_terminal=False,
            no_color=True,
            highlight=False
        )


def log_scenario(idx, tot_reward, rate, avg_step, ε, time):
    console.print(
        f"Scen: [number]{idx}"
        f" | total reward: [number]{tot_reward:.5f}"
        f" | success: [number]{rate:.1f}%"
        f" | avg_step: [number]{avg_step}"
        f" | ε: [number]{ε:.3f}",
        f" | time: [number]{time:.3f}s",
        style="ok"
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