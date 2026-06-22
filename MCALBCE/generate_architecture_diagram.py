"""
生成 MCALBCE 模型架构流程图 (纯 matplotlib 绘制，无需 graphviz)
运行: python generate_architecture_diagram.py
输出: figures/architecture.png
"""
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties, fontManager

# Register Noto CJK fonts from system OTF/TTC files
_cjk_font_paths = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
]
for _p in _cjk_font_paths:
    if os.path.exists(_p):
        fontManager.addfont(_p)

# Try to find a working CJK font
_cjk_font = None
for _candidate in ["Noto Sans CJK SC", "Noto Serif CJK SC", "Noto Sans CJK JP"]:
    try:
        _test = FontProperties(family=_candidate)
        _path = matplotlib.font_manager.findfont(_test, fallback_to_default=False)
        if _path and "last" not in _path.lower():
            _cjk_font = _candidate
            break
    except Exception:
        continue

if _cjk_font:
    plt.rcParams["font.family"] = [_cjk_font, "DejaVu Sans", "sans-serif"]
    print(f"Using CJK font: {_cjk_font}")
else:
    plt.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]
    print("Warning: No CJK font found, Chinese characters may not render")


def draw_box(ax, x, y, w, h, text, color, fontsize=9, text_color="white", bold=False):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="white", linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=weight, zorder=4)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color="#555555", style="-|>"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        color=color, linewidth=1.8, zorder=2,
    )
    ax.add_patch(arrow)


def draw_dashed_arrow(ax, x1, y1, x2, y2, color="#888888"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=12,
        color=color, linewidth=1.2, linestyle="dashed", zorder=2,
    )
    ax.add_patch(arrow)


def main():
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 20)
    ax.set_aspect("equal")
    ax.axis("off")

    # Color palette
    C_INPUT = "#2c3e50"
    C_EMBED = "#8e44ad"
    C_LSTM = "#2980b9"
    C_GRU = "#27ae60"
    C_TRANS = "#e67e22"
    C_CROSS = "#c0392b"
    C_FC = "#34495e"
    C_OUTPUT = "#16a085"
    C_LABEL = "#7f8c8d"

    # ===== Title =====
    ax.text(7, 19.3, "MCALBCE 模型架构", ha="center", va="center",
            fontsize=18, fontweight="bold", color="#2c3e50")
    ax.text(7, 18.8,
            "Multi-head Cross-Attention for Linear B-Cell Epitope Prediction",
            ha="center", va="center", fontsize=10, color="#7f8c8d", style="italic")

    # ===== Layer 1: Input =====
    draw_box(ax, 7, 17.8, 5, 0.55, "输入氨基酸序列  (长度 5~25)", C_INPUT, fontsize=10, bold=True)

    draw_arrow(ax, 7, 17.52, 7, 17.0)

    # ===== Layer 2: Embedding =====
    draw_box(ax, 7, 16.6, 5.5, 0.7,
             "词嵌入模块 (Embedding)\n23种氨基酸 → 128维向量  输出: (25, 128)",
             C_EMBED, fontsize=9, bold=True)

    # Three branches
    draw_arrow(ax, 4.5, 16.25, 2.8, 15.3)
    draw_arrow(ax, 7, 16.25, 7, 15.3)
    draw_arrow(ax, 9.5, 16.25, 11.2, 15.3)

    # ===== Layer 3: Three parallel branches =====
    # BiLSTM
    draw_box(ax, 2.8, 14.7, 3.5, 1.1,
             "双向长短期记忆\n(BiLSTM)\n隐藏维度: 64×2=128",
             C_LSTM, fontsize=9, bold=True)
    ax.text(2.8, 13.95, "遗忘门 / 输入门 / 输出门", ha="center", fontsize=7, color=C_LABEL)

    # BiGRU
    draw_box(ax, 7, 14.7, 3.5, 1.1,
             "双向门控循环单元\n(BiGRU)\n隐藏维度: 64×2=128",
             C_GRU, fontsize=9, bold=True)
    ax.text(7, 13.95, "更新门 / 重置门", ha="center", fontsize=7, color=C_LABEL)

    # Transformer
    draw_box(ax, 11.2, 14.7, 3.5, 1.1,
             "Transformer 编码器\n2层 / 8头自注意力\nFFN维度: 256",
             C_TRANS, fontsize=9, bold=True)
    ax.text(11.2, 13.95, "MSA + MLP + LN + 残差", ha="center", fontsize=7, color=C_LABEL)

    # ===== Role labels =====
    draw_arrow(ax, 2.8, 14.14, 2.8, 12.95)
    draw_arrow(ax, 7, 14.14, 7, 12.95)
    draw_arrow(ax, 11.2, 14.14, 11.2, 12.95)

    # Q, K, V labels
    ax.text(2.8, 13.55, "Q (查询)", ha="center", fontsize=10, fontweight="bold", color=C_LSTM)
    ax.text(7, 13.55, "K (键)", ha="center", fontsize=10, fontweight="bold", color=C_GRU)
    ax.text(11.2, 13.55, "V (值)", ha="center", fontsize=10, fontweight="bold", color=C_TRANS)

    # ===== Converge arrows to cross-attention =====
    draw_arrow(ax, 2.8, 12.95, 5.5, 12.1)
    draw_arrow(ax, 7, 12.95, 7, 12.1)
    draw_arrow(ax, 11.2, 12.95, 8.5, 12.1)

    # ===== Layer 4: Multi-head Cross-Attention =====
    draw_box(ax, 7, 11.5, 6.5, 1.1,
             "多头交叉注意力机制 (Multi-Head Cross-Attention)\n"
             "8个注意力头  |  Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V",
             C_CROSS, fontsize=9.5, bold=True)

    draw_arrow(ax, 7, 10.94, 7, 10.3)

    # ===== Layer 5: LayerNorm + Pooling =====
    draw_box(ax, 7, 9.9, 5, 0.6,
             "层归一化 (LayerNorm) + 全局平均池化",
             "#7f8c8d", fontsize=9.5, bold=True)

    draw_arrow(ax, 7, 9.6, 7, 8.9)

    # ===== Layer 6: FC Classifier =====
    draw_box(ax, 7, 8.4, 5.5, 0.9,
             "全连接分类模块\n128 → 64 → 32 → 1  (ReLU + Dropout)",
             C_FC, fontsize=9.5, bold=True)

    draw_arrow(ax, 7, 7.95, 7, 7.3)

    # ===== Layer 7: Output =====
    draw_box(ax, 7, 6.9, 4.5, 0.6,
             "Sigmoid → 二分类输出",
             C_OUTPUT, fontsize=10, bold=True)

    draw_arrow(ax, 5.5, 6.6, 4, 5.95)
    draw_arrow(ax, 8.5, 6.6, 10, 5.95)

    draw_box(ax, 4, 5.6, 2.5, 0.55, "线性 BCE", "#2980b9", fontsize=10, bold=True)
    draw_box(ax, 10, 5.6, 2.5, 0.55, "非线性 BCE", "#e74c3c", fontsize=10, bold=True)

    # ===== Training info box =====
    info_text = (
        "训练参数\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "优化器: Adam  |  学习率: 0.0001\n"
        "批大小: 32    |  训练轮数: 100\n"
        "损失函数: BCEWithLogitsLoss\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "数据集 (IEDB)\n"
        "训练: 4,440正 + 5,485负\n"
        "测试: 1,110正 + 1,408负\n"
        "CD-HIT 阈值: 0.7"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="#bdc3c7", alpha=0.95)
    ax.text(7, 3.5, info_text, ha="center", va="center", fontsize=8.5,
            fontfamily="monospace", bbox=props, color="#2c3e50", linespacing=1.5)

    # ===== Performance box =====
    perf_text = (
        "独立测试集性能\n"
        "━━━━━━━━━━━━━━━━━━━\n"
        "ACC: 0.767  SN: 0.814\n"
        "SP:  0.729  AUC: 0.820\n"
        "MCC: 0.540"
    )
    props2 = dict(boxstyle="round,pad=0.5", facecolor="#eaf7ea", edgecolor="#27ae60", alpha=0.95)
    ax.text(12, 8.4, perf_text, ha="center", va="center", fontsize=8.5,
            fontfamily="monospace", bbox=props2, color="#2c3e50", linespacing=1.5)

    plt.tight_layout()
    plt.savefig("figures/architecture.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print("架构图已保存到 figures/architecture.png")


if __name__ == "__main__":
    main()
