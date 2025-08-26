import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import seaborn as sns
from agent import AgentType

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (15, 12)

def plot_agent_performance(data, save_path=None):
    """
    Args:
        data: dict
        {
            'episodes': [100, 200, 300, ...],  # scen_idx
            'total_reward': [...],
            'success_rate': [...], 
            'agent_1': {
                'Q-mean': [...],
                'Q-std': [...],
                'TD-error': [...],
                'step': int
            },
            'agent_2': {...},
            'agent_3': {...}
        }
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Multi-Agent Performance Comparison', fontsize=16, fontweight='bold')
    
    episodes = data['episodes']
    agents = [i.value for i in AgentType]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    agent_names = ['Left Agent', 'Right Agent', 'Straight Agent']
    
    individual_metrics = [
        ('Q-mean', 'Q-Value Mean', 'Q-Value'),
        ('Q-std', 'Q-Value Variance', 'Variance'),
        ('TD-error', 'TD Errors', 'TD Error'),
        ('total_reward', 'Total Reward', 'Reward'),
        ('success_rate', 'Success Rate', 'Success Rate (%)'),
    ]

    plot_positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    
    for idx, (metric, title, ylabel) in enumerate(individual_metrics):
        if idx >= len(plot_positions):
            break
            
        row, col = plot_positions[idx]
        ax = axes[row, col]

        if metric == 'total_reward' or metric == 'success_rate':
            values = data[metric]
            ax.plot(episodes, values, 
                color="black", 
                marker='o', 
                markersize=4,
                linewidth=2,
                label="simulation",
                alpha=0.8)
        else:
            for agent_idx, agent in enumerate(agents):
                if agent in data and metric in data[agent]:
                    values = data[agent][metric]

                    ax.plot(episodes[-len(values):], values, 
                        color=colors[agent_idx], 
                        marker='o', 
                        markersize=4,
                        linewidth=2,
                        label=agent_names[agent_idx],
                        alpha=0.8)
            
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Episodes')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
        

        if metric == 'success_rate':
            ax.set_ylim(0, 100)

    axes[1, 2].axis('off')
        
    total_episodes = episodes[-1] if episodes else 0
    eval_points = len(episodes)

    final_total_reward = data.get('total_reward', [0])[-1] if data.get('total_reward') else 0
    final_success_rate = data.get('success_rate', [0])[-1] if data.get('success_rate') else 0    


    summary_lines = [
        f"[TRAINING SUMMARY]",
        f"Episodes Completed: {total_episodes:,}",
        f"Evaluation Points: {eval_points}",
        f"Active Agents: {len(agents)}",
        f"",
        f"[OVERALL PERFORMANCE]",
        f"Total Reward: {final_total_reward:.1f}",
        f"Success Rate: {final_success_rate:.1f}%",
        f"",
        f"[AGENT STATUS]"
    ]
    
    for agent_idx, agent in enumerate(agents):
        if agent in data and 'Q-mean' in data[agent] and data[agent]['Q-mean']:
            final_q_mean = data[agent]['Q-mean'][-1]
            final_td_error = data[agent].get('TD-error', [0])[-1] if data[agent].get('TD-error') else 0
            step = data[agent]['step']
            summary_lines.append(f"{agent_names[agent_idx]}:")
            summary_lines.append(f"  Q-Value: {final_q_mean:.3f}")
            summary_lines.append(f"  TD-Error: {final_td_error:.3f}")
            summary_lines.append(f"  Step: {step}")
            summary_lines.append(f"")
    
    summary_text = "\n".join(summary_lines)
    
    axes[1, 2].text(0.00, 0.95, summary_text, 
                transform=axes[1, 2].transAxes,
                fontsize=12.5,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(
                    boxstyle="round,pad=0.5", 
                    facecolor="#f8f9fa",
                    edgecolor="#dee2e6",
                    linewidth=1.5,
                    alpha=0.9
                ))
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.close()
    
if __name__ == "__main__":
    sample_data = {
        'episodes': [100, 200, 300, 400, 500],
        'total_reward': [300, 420, 550, 680, 790],
        'success_rate': [60, 70, 75, 82, 87],
        
        'left': {
            'Q-mean': [2.1, 2.3, 2.5, 2.6, 2.8],
            'Q-std': [0.5, 0.4, 0.35, 0.3, 0.25],
            'TD-error': [1.2, 1.0, 0.8, 0.7, 0.6],
            'step': 252,
        },
        'straight': {
            'Q-mean': [1.9, 2.1, 2.3, 2.4, 2.6],
            'Q-std': [0.6, 0.5, 0.4, 0.35, 0.3],
            'TD-error': [1.3, 1.1, 0.9, 0.8, 0.7],
            'step': 523,
        },
        'right': {
            'Q-mean': [1.7, 1.9, 2.1, 2.2, 2.4],
            'Q-std': [0.7, 0.6, 0.5, 0.4, 0.35],
            'TD-error': [1.5, 1.3, 1.1, 1.0, 0.9],
            'step': 352,
        }
    }
    
    plot_agent_performance(sample_data, save_path='agent_performance.png')
