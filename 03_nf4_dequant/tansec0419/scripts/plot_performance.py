import matplotlib.pyplot as plt
import numpy as np

# 性能数据 (GB/s)
versions = ['V1', 'V2', 'V3', 'V4', 'V5']
bandwidth_4096 = [33.7, 38.5, 20.5, np.nan, np.nan]
bandwidth_8192 = [np.nan, np.nan, 60.0, 103, 92]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 4096x4096
valid_v1 = [(v, b) for v, b in zip(versions, bandwidth_4096) if not np.isnan(b)]
ax1.bar([v[0] for v in valid_v1], [v[1] for v in valid_v1],
        color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax1.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax1.set_title('Performance on 4096×4096', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 50)

# 8192x8192
valid_v2 = [(v, b) for v, b in zip(versions, bandwidth_8192) if not np.isnan(b)]
bars = ax2.bar([v[0] for v in valid_v2], [v[1] for v in valid_v2],
               color=['#2ca02c', '#d62728', '#9467bd'])
ax2.set_ylabel('Bandwidth (GB/s)', fontsize=12)
ax2.set_title('Performance on 8192×8192', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim(0, 120)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 图表已保存: performance_comparison.png")
