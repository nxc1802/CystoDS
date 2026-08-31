import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_perfect_planar_architecture():
    # Canvas size: 21.5 x 9.2 inches
    fig, ax = plt.subplots(figsize=(21.5, 9.2), dpi=300)
    ax.set_xlim(0, 21.5)
    ax.set_ylim(0, 9.2)
    ax.axis('off')

    # Color Palette: Modern Academic LNCS
    c_bg_card = '#FFFFFF'
    c_input_bg = '#F0F7FD'
    c_input_stroke = '#2980B9'
    
    c_backbone_bg = '#F4FBF7'
    c_backbone_stroke = '#27AE60'
    c_stage_bg = '#E8F8F5'
    
    c_feat_bg = '#FEF9E7'
    c_feat_stroke = '#D35400'
    
    c_head_bg = '#FBF8FE'
    c_head_stroke = '#8E44AD'
    
    c_loss_bg = '#FDEDEC'
    c_loss_stroke = '#C0392B'
    
    c_infer_bg = '#EAFDF5'
    c_infer_stroke = '#1E8449'
    
    c_dark = '#2C3E50'
    c_gray = '#5D6D7E'

    def draw_card(x, y, w, h, title, subtitle='', bg=c_bg_card, stroke=c_dark, lw=1.6, t_size=11, s_size=9, t_color=c_dark):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                                    linewidth=lw, edgecolor=stroke, facecolor=bg, zorder=3)
        ax.add_patch(box)
        if subtitle:
            ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=t_size, fontweight='bold', color=t_color, zorder=4)
            ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=s_size, color=c_gray, zorder=4)
        else:
            ax.text(x + w/2, y + h*0.50, title, ha='center', va='center', fontsize=t_size, fontweight='bold', color=t_color, zorder=4)

    # ---------------- COLUMN 1: INPUT IMAGE (x = 0.5 to 2.5) ----------------
    draw_card(0.5, 3.8, 2.0, 2.2, "Endoscopy Input", "224 × 224 × 3 (WLC / NBI)", bg=c_input_bg, stroke=c_input_stroke, lw=2.0)

    # ---------------- COLUMN 2: SHARED BACKBONE (x = 3.0 to 6.6) ----------------
    backbone_box = patches.FancyBboxPatch((3.0, 1.9), 3.6, 6.0, boxstyle="round,pad=0.12,rounding_size=0.20",
                                         linewidth=2.0, edgecolor=c_backbone_stroke, facecolor=c_backbone_bg, zorder=2)
    ax.add_patch(backbone_box)
    ax.text(4.8, 7.45, "Shared Swin-Tiny Backbone", ha='center', va='center', fontsize=11.5, fontweight='bold', color='#196F3D', zorder=4)
    ax.text(4.8, 7.15, "Hierarchical Vision Transformer (28.2M)", ha='center', va='center', fontsize=8.5, color='#27AE60', zorder=4)

    draw_card(3.3, 5.7, 3.0, 0.9, "Stages 1 & 2", "Patch Merge + Shifted-Window Attention", bg='#FFFFFF', stroke='#A9DFBF', t_size=10, s_size=8.5)
    draw_card(3.3, 4.3, 3.0, 0.9, "Stage 3 (6 Blocks)", "Deep Multi-scale Feature Hierarchy", bg='#FFFFFF', stroke='#A9DFBF', t_size=10, s_size=8.5)
    draw_card(3.3, 2.3, 3.0, 1.4, "Stage 4 (Late-Stage)", r"$z \in \mathbb{R}^{768}$ | Global Context", bg=c_stage_bg, stroke=c_backbone_stroke, lw=1.8, t_size=10.5, s_size=8.5, t_color='#145A32')

    ax.annotate('', xy=(4.8, 5.2), xytext=(4.8, 5.7), arrowprops=dict(arrowstyle="-|>", color=c_backbone_stroke, lw=1.5), zorder=4)
    ax.annotate('', xy=(4.8, 3.7), xytext=(4.8, 4.3), arrowprops=dict(arrowstyle="-|>", color=c_backbone_stroke, lw=1.5), zorder=4)

    # ---------------- COLUMN 3: SHARED EMBEDDING z (x = 7.1 to 8.8) ----------------
    draw_card(7.1, 3.8, 1.7, 2.2, r"$z \in \mathbb{R}^{768}$", "Shared Semantic\nRepresentation", bg=c_feat_bg, stroke=c_feat_stroke, lw=2.0, t_size=12, s_size=9, t_color='#B9770E')

    # ---------------- COLUMN 4: MULTI-HEADS (x = 9.4 to 12.0) ----------------
    draw_card(9.4, 7.0, 2.6, 1.15, "SupCon Head", r"MLP: $768 \to 128$ (L2 norm)", bg=c_head_bg, stroke=c_head_stroke)
    draw_card(9.4, 5.3, 2.6, 1.15, "Layer 1: Binary Head", r"Linear: $768 \to 2$ (ROI / Non-ROI)", bg=c_head_bg, stroke=c_input_stroke)
    draw_card(9.4, 3.6, 2.6, 1.15, "Layer 2: Coarse Head", r"Linear: $768 \to 5$ Clinical Groups", bg=c_head_bg, stroke=c_head_stroke)
    draw_card(9.4, 1.9, 2.6, 1.15, "Layer 3: Fine Head", r"Linear: $768 \to 22$ Diagnostic Classes", bg=c_head_bg, stroke=c_head_stroke)

    # ---------------- COLUMN 5: LOSS OBJECTIVES (x = 12.6 to 17.0) ----------------
    draw_card(12.6, 7.0, 4.4, 1.15, r"$\mathcal{L}_{\text{supcon}}$ (Contrastive)", r"$\tau=0.10$ | Intra-class pull, inter-class push", bg=c_loss_bg, stroke=c_loss_stroke)
    draw_card(12.6, 5.3, 4.4, 1.15, r"$\mathcal{L}_{\text{bin}}$ (Binary CE)", "Initial joint representation pretraining", bg=c_loss_bg, stroke=c_loss_stroke)
    draw_card(12.6, 3.6, 4.4, 1.15, r"$\mathcal{L}_{\text{SBS}}(\text{Coarse})$ (Balanced Softmax)", r"Patient prior: $\pi_k = (\text{pts}_k + \epsilon)^{0.5}$", bg=c_loss_bg, stroke=c_loss_stroke)
    draw_card(12.6, 1.9, 4.4, 1.15, r"$\mathcal{L}_{\text{SBS}}(\text{Fine})$ (Balanced Softmax)", r"Patient prior: $\pi_j = (\text{pts}_j + \epsilon)^{0.5}$", bg=c_loss_bg, stroke=c_loss_stroke)

    # ---------------- COLUMN 6: CONSISTENCY & WARMUP (x = 17.6 to 20.8) - ZERO OVERLAPS ----------------
    cons_box = patches.FancyBboxPatch((17.6, 2.2), 3.2, 2.3, boxstyle="round,pad=0.08,rounding_size=0.15",
                                     linewidth=1.8, edgecolor='#8E44AD', facecolor='#FDF7FF', zorder=3)
    ax.add_patch(cons_box)
    ax.text(19.2, 4.05, "Coarse-Fine Consistency", ha='center', va='center', fontsize=10.0, fontweight='bold', color='#6C3483', zorder=4)
    ax.text(19.2, 3.65, r"$\mathcal{L}_{\text{cf}} = D_{\text{KL}}\left(P_{\text{c}} \parallel \sum_{f} P_{\text{fine}}(f)\right)$", ha='center', va='center', fontsize=9.0, fontweight='bold', color=c_loss_stroke, zorder=4)
    ax.text(19.2, 3.15, r"Warmup $w_{\text{hrc}}(t) = 0.25 \cdot \min(1.0, t/12)$", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#196F3D', zorder=4)
    ax.text(19.2, 2.65, "Resolves early representation bottleneck", ha='center', va='center', fontsize=7.5, color=c_gray, zorder=4)

    # ---------------- BOTTOM ROW: INFERENCE MARGINALIZATION (x = 7.1 to 20.8) ----------------
    draw_card(7.1, 0.35, 13.7, 1.15, "Hierarchical Marginalization & Multi-Head Blending (Inference-Only)",
              r"$P_{\text{ens}}(C) = \lambda P_{\text{coarse}}(C) + (1-\lambda)\sum_{f \in \text{Children}(C)} P_{\text{fine}}(f) \quad (\lambda = 0.25 \to +0.0524\text{ Coarse Accuracy Gain})$",
              bg=c_infer_bg, stroke=c_infer_stroke, lw=1.8, t_size=10.5, s_size=9.0, t_color='#145A32')

    # ==================== 100% PLANAR ORTHOGONAL WIRING (NO CROSSING / NO OVERLAPS) ====================
    # 1. Input -> Backbone
    ax.annotate('', xy=(3.0, 4.9), xytext=(2.5, 4.9),
                arrowprops=dict(arrowstyle="-|>", color=c_dark, lw=2.0), zorder=5)

    # 2. Stage 4 -> Shared Feature z (Orthogonal from bottom of stage 4)
    ax.plot([6.3, 6.7, 6.7, 7.1], [3.0, 3.0, 4.9, 4.9], color=c_dark, lw=2.0, zorder=5)
    ax.annotate('', xy=(7.1, 4.9), xytext=(6.8, 4.9),
                arrowprops=dict(arrowstyle="-|>", color=c_dark, lw=2.0), zorder=5)

    # 3. Shared Feature z -> Vertical Bus -> 4 Heads
    ax.plot([8.8, 9.1], [4.9, 4.9], color=c_dark, lw=2.0, zorder=5)
    ax.plot([9.1, 9.1], [2.47, 7.57], color=c_dark, lw=2.0, zorder=5)

    for y_h in [7.57, 5.87, 4.17, 2.47]:
        ax.annotate('', xy=(9.4, y_h), xytext=(9.1, y_h),
                    arrowprops=dict(arrowstyle="-|>", color=c_dark, lw=2.0), zorder=5)

    # 4. Heads -> Losses (Straight Horizontal arrows)
    for y_l in [7.57, 5.87, 4.17, 2.47]:
        ax.annotate('', xy=(12.6, y_l), xytext=(12.0, y_l),
                    arrowprops=dict(arrowstyle="-|>", color=c_loss_stroke, lw=1.8), zorder=5)

    # 5. Coarse-Fine Consistency Loop (On Far Right, completely free of obstructions)
    # Fine Loss (17.0, 2.47) -> goes right to (17.3, 2.47) -> goes up to (17.3, 3.35) -> enters Consistency Box at (17.6, 3.35)
    ax.plot([17.0, 17.3, 17.3, 17.6], [2.47, 2.47, 3.35, 3.35], color='#8E44AD', lw=1.8, linestyle='--', zorder=5)
    ax.annotate('', xy=(17.6, 3.35), xytext=(17.3, 3.35),
                arrowprops=dict(arrowstyle="-|>", color='#8E44AD', lw=1.8, linestyle='--'), zorder=5)

    # Consistency Box (17.6, 4.17) -> goes left directly to Coarse Loss (17.0, 4.17)
    ax.plot([17.6, 17.0], [4.17, 4.17], color='#8E44AD', lw=1.8, linestyle='--', zorder=5)
    ax.annotate('', xy=(17.0, 4.17), xytext=(17.4, 4.17),
                arrowprops=dict(arrowstyle="-|>", color='#8E44AD', lw=1.8, linestyle='--'), zorder=5)

    # 6. Heads -> Inference Marginalization (Dotted downward connection)
    ax.plot([10.7, 10.7], [1.9, 1.5], color=c_infer_stroke, lw=1.8, linestyle=':', zorder=5)
    ax.annotate('', xy=(10.7, 1.5), xytext=(10.7, 1.7),
                arrowprops=dict(arrowstyle="-|>", color=c_infer_stroke, lw=1.8), zorder=5)

    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/output/pdf/paper_assets/fig09_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/output/pdf/paper_assets/fig09_model_architecture.pdf', bbox_inches='tight')
    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/paper/paper_assets/fig09_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/paper/paper_assets/fig09_model_architecture.pdf', bbox_inches='tight')
    print("Generated 100% planar, non-overlapping Fig 2 architecture diagram!")

if __name__ == '__main__':
    create_perfect_planar_architecture()
