import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_clean_minimal_architecture():
    # Dimensions: 18 x 8.8 inches, clean proportions
    fig, ax = plt.subplots(figsize=(18, 8.8), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8.8)
    ax.axis('off')

    # Color Palette: Modern, Academic, High-Contrast
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

    # ---------------- 1. INPUT IMAGE ----------------
    draw_card(0.6, 3.6, 2.2, 2.2, "Endoscopy Input", "224 × 224 × 3 (WLC / NBI)", bg=c_input_bg, stroke=c_input_stroke, lw=2.0)

    # ---------------- 2. SHARED BACKBONE CONTAINER ----------------
    backbone_box = patches.FancyBboxPatch((3.4, 1.8), 3.8, 5.8, boxstyle="round,pad=0.12,rounding_size=0.20",
                                         linewidth=2.0, edgecolor=c_backbone_stroke, facecolor=c_backbone_bg, zorder=2)
    ax.add_patch(backbone_box)
    ax.text(5.3, 7.25, "Shared Swin-Tiny Backbone", ha='center', va='center', fontsize=11.5, fontweight='bold', color='#196F3D', zorder=4)
    ax.text(5.3, 6.95, "Hierarchical Vision Transformer (28.2M)", ha='center', va='center', fontsize=8.5, color='#27AE60', zorder=4)

    # Clean, uncluttered stages
    draw_card(3.7, 5.6, 3.2, 0.9, "Stages 1 & 2", "Patch Merge + Shifted-Window Attention", bg='#FFFFFF', stroke='#A9DFBF', t_size=10, s_size=8.5)
    draw_card(3.7, 4.2, 3.2, 0.9, "Stage 3 (6 Blocks)", "Deep Multi-scale Feature Hierarchy", bg='#FFFFFF', stroke='#A9DFBF', t_size=10, s_size=8.5)
    draw_card(3.7, 2.3, 3.2, 1.3, "Stage 4 (Late-Stage)", r"$z \in \mathbb{R}^{768}$ | Global Receptive Field", bg=c_stage_bg, stroke=c_backbone_stroke, lw=1.8, t_size=10.5, s_size=8.5, t_color='#145A32')

    # Internal backbone downward flow
    ax.annotate('', xy=(5.3, 5.1), xytext=(5.3, 5.6), arrowprops=dict(arrowstyle="-|>", color=c_backbone_stroke, lw=1.5), zorder=4)
    ax.annotate('', xy=(5.3, 3.6), xytext=(5.3, 4.2), arrowprops=dict(arrowstyle="-|>", color=c_backbone_stroke, lw=1.5), zorder=4)

    # ---------------- 3. SHARED EMBEDDING (z) ----------------
    draw_card(7.8, 3.7, 1.8, 2.0, r"$z \in \mathbb{R}^{768}$", "Shared Semantic\nRepresentation", bg=c_feat_bg, stroke=c_feat_stroke, lw=2.0, t_size=12, s_size=9, t_color='#B9770E')

    # ---------------- 4. MULTI-HEADS ----------------
    draw_card(10.3, 6.7, 2.6, 1.1, "SupCon Head", r"MLP: $768 \to 128$ (L2 norm)", bg=c_head_bg, stroke=c_head_stroke)
    draw_card(10.3, 5.1, 2.6, 1.1, "Layer 1: Binary Head", r"Linear: $768 \to 2$ (ROI / Non-ROI)", bg=c_head_bg, stroke=c_input_stroke)
    draw_card(10.3, 3.5, 2.6, 1.1, "Layer 2: Coarse Head", r"Linear: $768 \to 5$ Clinical Groups", bg=c_head_bg, stroke=c_head_stroke)
    draw_card(10.3, 1.9, 2.6, 1.1, "Layer 3: Fine Head", r"Linear: $768 \to 22$ Diagnostic Classes", bg=c_head_bg, stroke=c_head_stroke)

    # ---------------- 5. LOSS OBJECTIVES ----------------
    draw_card(13.6, 6.7, 3.8, 1.1, r"$\mathcal{L}_{\text{supcon}}$ (Contrastive)", r"$\tau=0.10$ | Intra-class cluster, inter-class push", bg=c_loss_bg, stroke=c_loss_stroke)
    draw_card(13.6, 5.1, 3.8, 1.1, r"$\mathcal{L}_{\text{bin}}$ (Binary CE)", "Initial joint representation pretraining", bg=c_loss_bg, stroke=c_loss_stroke)
    draw_card(13.6, 3.5, 3.8, 1.1, r"$\mathcal{L}_{\text{SBS}}(\text{Coarse})$", r"Patient prior: $\pi_k = (\text{pts}_k + \epsilon)^{0.5}$", bg=c_loss_bg, stroke=c_loss_stroke)
    draw_card(13.6, 1.9, 3.8, 1.1, r"$\mathcal{L}_{\text{SBS}}(\text{Fine})$", r"Patient prior: $\pi_j = (\text{pts}_j + \epsilon)^{0.5}$", bg=c_loss_bg, stroke=c_loss_stroke)

    # ---------------- 6. INFERENCE MARGINALIZATION (Bottom Container) ----------------
    draw_card(7.8, 0.35, 9.6, 1.15, "Hierarchical Marginalization & Multi-Head Blending (Inference)",
              r"$P_{\text{ens}}(C) = \lambda P_{\text{coarse}}(C) + (1-\lambda)\sum_{f \in \text{Children}(C)} P_{\text{fine}}(f) \quad (\lambda = 0.25 \to +5.24\%\text{ Coarse Accuracy})$",
              bg=c_infer_bg, stroke=c_infer_stroke, lw=1.8, t_size=10.5, s_size=9.0, t_color='#145A32')

    # ==================== ORTHOGONAL WIRING (90-DEGREE ANGLES, ZERO OVERLAP) ====================
    # 1. Input -> Backbone
    ax.annotate('', xy=(3.4, 4.7), xytext=(2.8, 4.7),
                arrowprops=dict(arrowstyle="-|>", color=c_dark, lw=2.0), zorder=5)

    # 2. Stage 4 -> Shared Feature z (Orthogonal from bottom of stage 4 to z)
    ax.plot([6.9, 7.35, 7.35, 7.8], [2.95, 2.95, 4.7, 4.7], color=c_dark, lw=2.0, zorder=5)
    ax.annotate('', xy=(7.8, 4.7), xytext=(7.5, 4.7),
                arrowprops=dict(arrowstyle="-|>", color=c_dark, lw=2.0), zorder=5)

    # 3. Shared Feature z -> Vertical Bus -> 4 Heads
    ax.plot([9.6, 9.95], [4.7, 4.7], color=c_dark, lw=2.0, zorder=5)
    ax.plot([9.95, 9.95], [2.45, 7.25], color=c_dark, lw=2.0, zorder=5)

    for y_h in [7.25, 5.65, 4.05, 2.45]:
        ax.annotate('', xy=(10.3, y_h), xytext=(9.95, y_h),
                    arrowprops=dict(arrowstyle="-|>", color=c_dark, lw=2.0), zorder=5)

    # 4. Heads -> Losses (Straight Horizontal)
    for y_l in [7.25, 5.65, 4.05, 2.45]:
        ax.annotate('', xy=(13.6, y_l), xytext=(12.9, y_l),
                    arrowprops=dict(arrowstyle="-|>", color=c_loss_stroke, lw=1.8), zorder=5)

    # 5. Coarse-Fine Consistency Loop (Right Side, Orthogonal U-turn)
    # Fine (y=2.45) -> go right to x=13.25 -> go up to y=4.05 -> Coarse (y=4.05)
    ax.plot([12.9, 13.25, 13.25, 12.9], [2.45, 2.45, 4.05, 4.05], color=c_head_stroke, lw=2.0, linestyle='--', zorder=5)
    ax.annotate('', xy=(12.9, 4.05), xytext=(13.15, 4.05),
                arrowprops=dict(arrowstyle="-|>", color=c_head_stroke, lw=2.0), zorder=5)

    # Consistency pill badge (centered neatly without overlapping anything)
    c_pill = patches.FancyBboxPatch((12.5, 3.0), 1.5, 0.50, boxstyle="round,pad=0.06",
                                   facecolor='#FADBD8', edgecolor=c_loss_stroke, lw=1.2, zorder=6)
    ax.add_patch(c_pill)
    ax.text(13.25, 3.33, r"$\mathcal{L}_{\text{cf}} = D_{\text{KL}}$", ha='center', va='center', fontsize=8.5, fontweight='bold', color=c_loss_stroke, zorder=7)
    ax.text(13.25, 3.12, r"Warmup $w(t)$", ha='center', va='center', fontsize=7.5, color='#78281F', zorder=7)

    # 6. Heads -> Inference Marginalization (Dotted downward connection)
    ax.plot([11.6, 11.6], [1.9, 1.5], color=c_infer_stroke, lw=1.8, linestyle=':', zorder=5)
    ax.annotate('', xy=(11.6, 1.5), xytext=(11.6, 1.7),
                arrowprops=dict(arrowstyle="-|>", color=c_infer_stroke, lw=1.8), zorder=5)

    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/output/pdf/paper_assets/fig09_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/output/pdf/paper_assets/fig09_model_architecture.pdf', bbox_inches='tight')
    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/paper/paper_assets/fig09_model_architecture.png', dpi=300, bbox_inches='tight')
    plt.savefig('/Volumes/WorkSpace/Project/CystoDS/Docs/paper/paper_assets/fig09_model_architecture.pdf', bbox_inches='tight')
    print("Generated sleek, minimal, publication-ready Fig 2 architecture diagram!")

if __name__ == '__main__':
    create_clean_minimal_architecture()
