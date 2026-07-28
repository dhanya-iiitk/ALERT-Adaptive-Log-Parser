import matplotlib.pyplot as plt

labels = ['AEL','LenMa','Spell','Logram','LogMine',
          'Drain','LogPPT','Brain','Lilac','ALERT']
GA  = [0.778,0.720,0.706,0.505,0.662,0.803,0.594,0.841,0.660,0.755]
PA  = [0.554,0.508,0.321,0.345,0.407,0.612,0.349,0.743,0.471,0.655]
FGA = [0.695,0.602,0.623,0.431,0.482,0.741,0.760,0.882,0.605,0.665]
FTA = [0.289,0.171,0.177,0.124,0.151,0.309,0.558,0.389,0.374,0.521]

x = range(len(labels))
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(x, GA,  color='#378ADD', marker='s', linewidth=1.8, markersize=7, label='GA')
ax.plot(x, PA,  color='#E24B4A', marker='o', linewidth=1.8, markersize=7, label='PA')
ax.plot(x, FGA, color='#1D9E75', marker='^', linewidth=1.8, markersize=7, label='FGA')
ax.plot(x, FTA, color='#EF9F27', marker='v', linewidth=1.8, markersize=7, label='FTA')
ax.axvline(x=9, color='#7F77DD', linestyle='--', alpha=0.4, linewidth=1.2)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticks([i/10 for i in range(0,11)])
ax.set_yticklabels([f'{i/10:.1f}' for i in range(0,11)], fontsize=10)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.get_xticklabels()[-1].set_color('#7F77DD')
ax.get_xticklabels()[-1].set_fontweight('bold')
plt.tight_layout()
plt.savefig('metrics_line_chart.pdf', dpi=300, bbox_inches='tight')
plt.savefig('metrics_line_chart.png', dpi=300, bbox_inches='tight')
print('Saved metrics_line_chart.pdf and metrics_line_chart.png')
