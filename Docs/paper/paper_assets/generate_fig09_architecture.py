import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_model_architecture_diagram():
    fig, ax = plt.subplots(figsize=(20, 11), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 11)
    ax.axis('off')

    # Color Palette - Professional Medical AI Theme
    c_input = '#EBF5FB'      # Soft Blue
    c_backbone = '#E8F8F5'   # Soft Mint Green
    c_feature = '#FEF9E7'    # Soft Amber
    c_head = '#F4ECF7'       # Soft Purple
    c_loss = '#FDEDEC'       # Soft Crimson
    c_infer = '#EAFAF1'      # Soft Emerald
    
    c_stroke_dark = '#2C3E50'
    c_accent_blue = '#2980B9'
    c_accent_green = '#27AE60'
    c_accent_purple = '#8E44AD'
    c_accent_red = '#C0392B'
    c_accent_orange = '#D35400'

    # Helper function for drawing rounded boxes
    def draw_box(x, y, w, h, text, subtitle='', bg_color='#FFFFFF', border_color=c_stroke_dark, lw=1.8, title_fontsize=11, sub_fontsize=9, text_color='#1C2833'):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12,rounding_size=0.15",
                                    linewidth=lw, edgecolor=border_color, facecolor=bg_color, zorder=3)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, text, ha='center', va='center', fontsize=title_fontsize, fontweight='bold', color=text_color, zorder=4)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=sub_fontsize, color='#566573', zorder=4)
        else:
            ax.text(x + w/2, y + h*0.50, text, ha='center', va='center', fontsize=title_fontsize, fontweight='bold', color=text_color, zorder=4)

    # 1. INPUT SECTION
    draw_box(0.5, 4.5, 2.4, 2.0, "Cystoscopy Image", "224 × 224 × 3 (WLC / NBI)", bg_color=c_input, border_color=c_accent_blue)

    # 2. SHARED BACKBONE CONTAINER (Swin Transformer Tiny)
    backbone_container = patches.FancyBboxPatch((3.4, 2.0), 4.3, 6.7, boxstyle="round,pad=0.15,rounding_size=0.2",
                                               linewidth=2.0, edgecolor=c_accent_green, facecolor=c_backbone, zorder=1)
    ax.add_patch(backbone_container)
    ax.text(5.55, 8.40, "Shared Vision Backbone: Swin-Tiny (28.23M)", ha='center', va='center', fontsize=11, fontweight='bold', color='#196F3D', zorder=4)

    draw_box(3.7, 7.1, 3.7, 0.9, "Patch Partition & Linear Embed", "Patch size: 4 × 4 | C = 96", bg_color='#FFFFFF', border_color='#7DCEA0')
    draw_box(3.7, 5.9, 3.7, 0.9, "Stage 1: Swin Blocks (×2)", "Window: 7 × 7 | H/4 × W/4", bg_color='#FFFFFF', border_color='#7DCEA0')
    draw_box(3.7, 4.7, 3.7, 0.9, "Stage 2: Patch Merge + Swin (×2)", "C = 192 | H/8 × W/8", bg_color='#FFFFFF', border_color='#7DCEA0')
    draw_box(3.7, 3.5, 3.7, 0.9, "Stage 3: Patch Merge + Swin (×6)", "C = 384 | H/16 × W/16", bg_color='#FFFFFF', border_color='#7DCEA0')
    draw_box(3.7, 2.3, 3.7, 0.9, "Stage 4: Patch Merge + Swin (×2)", "C = 768 | H/32 × W/32", bg_color='#D4EFDF', border_color='#27AE60')

    # Backbone internal vertical connections (Straight down)
    for y_top, y_bot in [(7.1, 6.8), (5.9, 5.6), (4.7, 4.4), (3.5, 3.2)]:
        ax.annotate('', xy=(5.55, y_bot), xytext=(5.55, y_top),
                    arrowprops=dict(arrowstyle="-|>", color=c_accent_green, lw=1.5), zorder=4)

    # 3. SHARED EMBEDDING (Late-Stage Feature z)
    draw_box(8.3, 4.5, 2.1, 2.0, "Shared Late-Stage\nEmbedding (Stage 4)", r"$z \in \mathbb{R}^{768}$" + "\n(LayerNorm Pooling)", bg_color=c_feature, border_color=c_accent_orange, lw=2.0)

    # 4. MULTI-TASK & CONTRASTIVE HEADS (Right Column)
    # Head 0: Contrastive Projection Head
    draw_box(11.2, 8.4, 2.8, 1.2, "SupCon Projection Head", r"MLP: $768 \to 128$ (L2 norm)", bg_color=c_head, border_color=c_accent_purple)

    # Head 1: Binary Head
    draw_box(11.2, 6.4, 2.8, 1.2, "Layer 1: Binary Head", r"Linear: $768 \to 2$ (ROI / Non-ROI)", bg_color=c_head, border_color=c_accent_blue)

    # Head 2: Coarse Head
    draw_box(11.2, 4.4, 2.8, 1.2, "Layer 2: Coarse Head", r"Linear: $768 \to 5$ Clinical Groups", bg_color=c_head, border_color=c_accent_purple)

    # Head 3: Fine Head
    draw_box(11.2, 2.4, 2.8, 1.2, "Layer 3: Fine Head", r"Linear: $768 \to 22$ Diagnostic Classes", bg_color=c_head, border_color=c_accent_purple)

    # 5. OBJECTIVE FUNCTIONS & LOSSES (Far Right Column)
    draw_box(14.7, 8.4, 4.8, 1.2, r"$\mathcal{L}_{\text{supcon}}$ (Supervised Contrastive)", r"$\tau=0.10$ | Pulls intra-class, pushes inter-class", bg_color=c_loss, border_color=c_accent_red)
    draw_box(14.7, 6.4, 4.8, 1.2, r"$\mathcal{L}_{\text{bin}}$ (Binary Cross-Entropy)", r"Stage 1 joint representation pretraining", bg_color=c_loss, border_color=c_accent_red)
    draw_box(14.7, 4.4, 4.8, 1.2, r"$\mathcal{L}_{\text{SBS}}(\text{Coarse})$ (Smoothed Balanced Softmax)", r"Patient prior: $\pi_{\text{coarse}} = (\text{pts}_k + \epsilon)^{0.5}$", bg_color=c_loss, border_color=c_accent_red)
    draw_box(14.7, 2.4, 4.8, 1.2, r"$\mathcal{L}_{\text{SBS}}(\text{Fine})$ (Smoothed Balanced Softmax)", r"Patient prior: $\pi_{\text{fine}} = (\text{pts}_j + \epsilon)^{0.5}$", bg_color=c_loss, border_color=c_accent_red)

    # 6. INFERENCE & HIERARCHICAL MARGINALIZATION (Bottom Container)
    draw_box(8.3, 0.4, 11.2, 1.4, "Hierarchical Marginalization & Multi-Head Blending (Inference)",
             r"$P_{\text{ens}}(C) = \lambda P_{\text{coarse}}(C) + (1-\lambda)\sum_{f \in \text{Children}(C)} P_{\text{fine}}(f) \quad (\lambda = 0.25 \to +5.24\%\text{ Coarse Accuracy})$",
             bg_color=c_infer, border_color=c_accent_green, lw=2.0, title_fontsize=11, sub_fontsize=9.5, text_color='#145A32')

    # ==================== ORTHOGONAL WIRING (NO DIAGONALS / NO CROSSING BOXES) ====================
    # Wiring 1: Input -> Backbone
    ax.annotate('', xy=(3.4, 5.5), xytext=(2.9, 5.5),
                arrowprops=dict(arrowstyle="-|>", color=c_stroke_dark, lw=2.0), zorder=5)

    # Wiring 2: Backbone Stage 4 -> Shared Feature z (Orthogonal from bottom of stage 4 to z)
    # Stage 4 right edge: (7.4, 2.75) -> right to 7.8 -> up to 5.5 -> right to 8.3
    ax.plot([7.4, 7.85, 7.85, 8.3], [2.75, 2.75, 5.5, 5.5], color=c_stroke_dark, lw=2.0, zorder=5)
    ax.annotate('', xy=(8.3, 5.5), xytext=(8.0, 5.5),
                arrowprops=dict(arrowstyle="-|>", color=c_stroke_dark, lw=2.0), zorder=5)

    # Wiring 3: Shared Feature z -> 4 Heads (Clean Orthogonal Bus Line at x = 10.8)
    # Bus trunk: (10.4, 5.5) -> (10.8, 5.5) -> branches to y = 9.0, 7.0, 5.0, 3.0
    ax.plot([10.4, 10.8], [5.5, 5.5], color=c_stroke_dark, lw=2.0, zorder=5)
    ax.plot([10.8, 10.8], [3.0, 9.0], color=c_stroke_dark, lw=2.0, zorder=5) # vertical bus

    # 4 Branch connections from bus to heads
    for y_head in [9.0, 7.0, 5.0, 3.0]:
        ax.annotate('', xy=(11.2, y_head), xytext=(10.8, y_head),
                    arrowprops=dict(arrowstyle="-|>", color=c_stroke_dark, lw=2.0), zorder=5)

    # Wiring 4: Heads -> Loss Functions (Straight horizontal lines)
    for y_loss in [9.0, 7.0, 5.0, 3.0]:
        ax.annotate('', xy=(14.7, y_loss), xytext=(14.0, y_loss),
                    arrowprops=dict(arrowstyle="-|>", color=c_accent_red, lw=1.8), zorder=5)

    # Wiring 5: Consistency Constraint between Fine Head and Coarse Head (Orthogonal loop on the right)
    # Fine (y=3.0) -> go right to x=14.35 -> go up to y=5.0 -> Coarse (y=5.0)
    ax.plot([14.0, 14.35, 14.35, 14.0], [3.0, 3.0, 5.0, 5.0], color=c_accent_purple, lw=2.2, linestyle='--', zorder=5)
    ax.annotate('', xy=(14.0, 5.0), xytext=(14.2, 5.0),
                arrowprops=dict(arrowstyle="-|>", color=c_accent_purple, lw=2.2), zorder=5)
    
    # Consistency Label Badge (Neatly centered on vertical segment without overlapping)
    cons_badge = patches.FancyBboxPatch((13.4, 3.65), 1.9, 0.70, boxstyle="round,pad=0.08",
                                       facecolor='#FADBD8', edgecolor=c_accent_red, lw=1.5, zorder=6)
    ax.add_patch(cons_badge)
    ax.text(14.35, 4.07, r"$\mathcal{L}_{\text{cf}} = D_{\text{KL}}(P_{\text{c}} \parallel P_{\text{fine}})$", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_accent_red, zorder=7)
    ax.text(14.35, 3.80, r"Warmup $w_{\text{hrc}}(t)$", ha='center', va='center', fontsize=8.0, color='#78281F', zorder=7)

    # Wiring 6: Orthogonal connection from Heads to Hierarchical Marginalization (Inference block)
    # From Coarse (12.6, 4.4) down to (12.6, 1.8) -> Inference
    # From Fine (12.6, 2.4) down to (12.6, 1.8) -> Inference
    ax.plot([12.6, 12.6], [2.4, 1.8], color=c_accent_green, lw=2.0, linestyle=':', zorder=5)
    ax.annotate('', xy=(12.6, 1.8), xytext=(12.6, 2.0),
                arrowprops=dict(arrowstyle="-|>", color=c_accent_green, lw=2.0), zorder=5)

    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/output/pdf/paper_assets/fig09_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/output/pdf/paper_assets/fig09_model_architecture.pdf', bbox_inches='tight')
    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/paper/paper_assets/fig09_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/paper/paper_assets/fig09_model_architecture.pdf', bbox_inches='tight')
    print("Successfully generated high-resolution orthogonal architecture diagrams!")

if __name__ == '__main__':
    create_model_architecture_diagram()
