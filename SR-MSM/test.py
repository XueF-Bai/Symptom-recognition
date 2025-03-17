import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axes
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')  # Hide axes

# Define positions for boxes and arrows
positions = [
    (4, 8, "医疗对话", (4.5, 8.8)),        # Medical Dialogue
    (4, 6, "信息提取", (4.5, 6.8)),        # Information Extraction
    (4, 4, "结构化数据", (4.5, 4.8)),      # Structured Data
    (4, 2, "医疗建议生成", (4.5, 2.8))    # Medical Recommendation
]

# Draw boxes and add text
for x, y, text, text_pos in positions:
    rect = patches.Rectangle((x, y), 2, 1, linewidth=2, edgecolor='black', facecolor='white')
    ax.add_patch(rect)
    ax.text(text_pos[0], text_pos[1], text, fontsize=12, ha='center', va='center')

# Draw arrows
arrow_positions = [
    (5, 8, 5, 7.2),  # From "医疗对话" to "信息提取"
    (5, 6, 5, 5.2),  # From "信息提取" to "结构化数据"
    (5, 4, 5, 3.2)   # From "结构化数据" to "医疗建议生成"
]

for x1, y1, x2, y2 in arrow_positions:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', arrowstyle='->'))

# Add a title
plt.title("医疗建议生成流程图", fontsize=16, pad=20)

# Show the plot
plt.show()
