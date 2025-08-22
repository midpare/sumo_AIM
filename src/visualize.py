import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import seaborn as sns
from agent import AgentType

# 한글 폰트 설정 (필요시)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (15, 12)

def plot_agent_performance(data, save_path=None):
    """
    3개 에이전트의 성능 지표를 시각화
    
    Args:
        data: dict 형태
        {
            'episodes': [100, 200, 300, ...],  # 시나리오 번호
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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 파란색, 주황색, 초록색
    agent_names = ['Left Agent', 'Right Agent', 'Straight Agent']
    
    # 지표별 설정
    individual_metrics = [
        ('Q-mean', 'Q-Value Mean', 'Q-Value'),
        ('Q-std', 'Q-Value Variance', 'Variance'),
        ('TD-error', 'TD Errors', 'TD Error'),
        ('total_reward', 'Total Reward', 'Reward'),
        ('success_rate', 'Success Rate', 'Success Rate (%)'),
    ]
    # 5개 지표를 2x3 그리드에 배치
    plot_positions = [(0,0), (0,1), (0,2), (1,0), (1,1)]
    
    for idx, (metric, title, ylabel) in enumerate(individual_metrics):
        if idx >= len(plot_positions):
            break
            
        row, col = plot_positions[idx]
        ax = axes[row, col]
        
        # 각 에이전트별로 선 그래프 그리기
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
        
        # Success rate는 0-100% 범위로 설정
        if metric == 'success_rate':
            ax.set_ylim(0, 100)
    
    # 마지막 subplot은 비우거나 전체 요약 정보 표시
    axes[1, 2].axis('off')
        
    total_episodes = episodes[-1] if episodes else 0
    eval_points = len(episodes)
    
    # 통합 성과
    final_total_reward = data.get('total_reward', [0])[-1] if data.get('total_reward') else 0
    final_success_rate = data.get('success_rate', [0])[-1] if data.get('success_rate') else 0
    
    # 요약 텍스트 구성 (이모지 대신 텍스트 심볼 사용)
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
    
    # 각 에이전트별 Q-value 정보 (더 깔끔하게)
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
    
    # 텍스트 합치기
    summary_text = "\n".join(summary_lines)
    
    # 예쁜 박스 스타일로 표시
    axes[1, 2].text(0.00, 0.95, summary_text, 
                transform=axes[1, 2].transAxes,
                fontsize=12.5,
                verticalalignment='top',
                fontfamily='monospace',  # 고정폭 폰트로 정렬
                bbox=dict(
                    boxstyle="round,pad=0.5", 
                    facecolor="#f8f9fa",    # 연한 회색
                    edgecolor="#dee2e6",    # 테두리
                    linewidth=1.5,
                    alpha=0.9
                ))
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.close()
    
# 사용 예시
if __name__ == "__main__":
    # 수정된 예시 데이터 구조
    sample_data = {
        'episodes': [100, 200, 300, 400, 500],
        # 통합 지표들
        'total_reward': [300, 420, 550, 680, 790],      # 전체 에이전트 합계
        'success_rate': [60, 70, 75, 82, 87],           # 전체 성공률
        
        # 에이전트별 개별 지표들
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
    
    # 그래프 생성
    plot_agent_performance(sample_data, save_path='agent_performance.png')
