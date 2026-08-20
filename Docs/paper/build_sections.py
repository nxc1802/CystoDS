#!/usr/bin/env python3
"""Build standalone 1-column LaTeX and PDF documents for individual paper sections:
  1. Dataset & Protocol (Mục 2)
  2. Methodology (Mục 3)
  3. Experiment & Ablations (Mục 4)

Preserves the original master PDF while providing clean, focused 1-column PDFs for research review.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pypandoc

PAPER_DIR = Path(__file__).resolve().parent
DOCS = PAPER_DIR.parent
ROOT = DOCS.parent
SOURCE = PAPER_DIR / "paper.md"
TMP_DIR = DOCS / "tmp" / "pdfs"
OUTPUT_DIR = DOCS / "output" / "pdf"

SECTION_DEFS = [
    {
        "id": "01_Dataset",
        "title": "CystoDS: Báo cáo Tập Dữ Liệu, Bối Cảnh và Giao Thức Đánh Giá Độc Lập Bệnh Nhân",
        "subtitle": "Kiểm định 8.067 ảnh, 160 bệnh nhân, phân tích liên hệ nghiên cứu và 3 phân hoạch hold-out chuẩn hóa (Stage 00)",
        "header_left": "CystoDS -- Báo cáo Tập Dữ Liệu và Giao Thức (Stage 00)",
        "regex": r"(## 1\. Đặt vấn đề[\s\S]*?)(?=## 4\. Phương pháp)",
        "filename_stem": "CystoDS_01_Dataset_VI",
    },
    {
        "id": "02_Methodology",
        "title": "CystoDS: Phương Pháp Phân Cấp Ba Giai Đoạn Tuần Tự (3S-HFT v3.1)",
        "subtitle": "Kiến trúc Swin-Tiny, Lịch trình Curriculum Warmup cho Hierarchy Loss và Cơ chế Hierarchical Marginalization",
        "header_left": "CystoDS -- Phương Pháp Nghiên Cứu và Mô Hình Đề Xuất (3S-HFT)",
        "regex": r"(## 4\. Phương pháp Đề xuất[\s\S]*?)(?=## 5\. Kết quả)",
        "filename_stem": "CystoDS_02_Methodology_VI",
    },
    {
        "id": "03_Experiment",
        "title": "CystoDS: Báo Cáo Kết Quả Thực Nghiệm Toàn Diện Theo Từng Giai Đoạn (Stages 10--40)",
        "subtitle": "Đánh giá đối chuẩn qua 3 hold-out splits độc lập và bóc tách định lượng 10 biến thể triệt tiêu thành phần",
        "header_left": "CystoDS -- Báo cáo Kết Quả Thực Nghiệm Theo Giai Đoạn (Stages 10--40)",
        "regex": r"(## 5\. Kết quả Thực nghiệm[\s\S]*)",
        "filename_stem": "CystoDS_03_Experiment_VI",
    },
]


def clean_cell_tex(cell: str) -> str:
    """Strip minipages and clean spacing in TeX table cells."""
    cell = re.sub(
        r"\\begin\{minipage\}\[[^\]]*\]\{\\linewidth\}\s*(?:\\raggedright|\\raggedleft|\\centering)?\s*(.*?)\s*\\end\{minipage\}",
        r"\1",
        cell,
        flags=re.DOTALL,
    )
    return cell.strip()


def transform_tables_1col(tex: str) -> str:
    """Transform longtable environments into clean 1-column publication table floats."""
    starts = [m.start() for m in re.finditer(r"\\begin\{longtable\}", tex)]
    ends = [m.end() for m in re.finditer(r"\\end\{longtable\}", tex)]

    if not starts:
        return tex

    replacements = []
    for idx, (s, e) in enumerate(zip(starts, ends)):
        pre = tex[max(0, s - 150) : s]
        b_start = s
        if "{\\def\\LTcaptype" in pre:
            pos = pre.rfind("{\\def\\LTcaptype")
            b_start = s - (len(pre) - pos)
            pre_sub = tex[max(0, b_start - 40) : b_start]
            if "{\\footnotesize" in pre_sub:
                b_start -= len(pre_sub) - pre_sub.rfind("{\\footnotesize")
            elif "{\\small" in pre_sub:
                b_start -= len(pre_sub) - pre_sub.rfind("{\\small")

        wrapper_prefix = tex[b_start:s]
        unclosed = wrapper_prefix.count("{") - wrapper_prefix.count("}")

        curr = e
        closed = 0
        while curr < len(tex) and closed < unclosed:
            if tex[curr] == "}":
                closed += 1
            curr += 1
        b_end = curr

        lt_code = tex[s:e]
        top_pos = lt_code.find("\\toprule")
        mid_pos = lt_code.find("\\midrule")
        foot_pos = lt_code.find("\\endlastfoot")

        if top_pos != -1 and mid_pos != -1:
            header_raw = lt_code[top_pos + len("\\toprule") : mid_pos]
            header_raw = re.sub(r"\\noalign\{\}", "", header_raw).strip()

            if foot_pos != -1:
                body_raw = lt_code[foot_pos + len("\\endlastfoot") :].strip()
            else:
                body_raw = lt_code[mid_pos + len("\\midrule") :].strip()
            body_raw = re.sub(r"\\noalign\{\}", "", body_raw).strip()
            body_raw = re.sub(r"\\endhead", "", body_raw).strip()
            body_raw = re.sub(r"\\end\{longtable\}", "", body_raw).strip()

            header_cleaned = clean_cell_tex(header_raw)
            header_first_line = header_cleaned.split("\\\\")[0]
            cols = [clean_cell_tex(c) for c in header_first_line.split("&")]
            num_cols = len(cols)
            header_str = " & ".join(cols) + " \\\\"

            align_chars = []
            for c_idx in range(num_cols):
                if c_idx == 0:
                    align_chars.append("l")
                else:
                    align_chars.append("r" if num_cols > 3 or c_idx > 0 else "l")

            align_str = "".join(align_chars)
            cap = f"Bảng {idx+1}."
            cap = re.sub(r"(?<!\\)%", r"\%", cap)

            if num_cols >= 8:
                new_tex = (
                    f"\\clearpage\n"
                    f"\\begin{{landscape}}\n"
                    f"\\begin{{table}}[p]\n"
                    f"\\centering\n"
                    f"\\caption{{{cap}}}\n"
                    f"\\label{{tab:sec_table{idx+1}}}\n"
                    f"\\vspace{{4pt}}\n"
                    f"\\adjustbox{{max width=0.98\\linewidth}}{{\n"
                    f"\\begin{{tabular}}{{{align_str}}}\n"
                    f"\\toprule\n"
                    f"{header_str}\n"
                    f"\\midrule\n"
                    f"{body_raw}\n"
                    f"\\bottomrule\n"
                    f"\\end{{tabular}}\n"
                    f"}}\n"
                    f"\\end{{table}}\n"
                    f"\\end{{landscape}}\n"
                    f"\\clearpage"
                )
            else:
                new_tex = (
                    f"\\begin{{table}}[htbp]\n"
                    f"\\centering\n"
                    f"\\caption{{{cap}}}\n"
                    f"\\label{{tab:sec_table{idx+1}}}\n"
                    f"\\vspace{{3pt}}\n"
                    f"\\adjustbox{{max width=\\textwidth}}{{\n"
                    f"\\begin{{tabular}}{{{align_str}}}\n"
                    f"\\toprule\n"
                    f"{header_str}\n"
                    f"\\midrule\n"
                    f"{body_raw}\n"
                    f"\\bottomrule\n"
                    f"\\end{{tabular}}\n"
                    f"}}\n"
                    f"\\end{{table}}"
                )
            replacements.append((b_start, b_end, new_tex))

    replacements.sort(key=lambda x: x[0], reverse=True)
    tex_list = list(tex)
    for b_start, b_end, new_code in replacements:
        tex_list[b_start:b_end] = list(new_code)

    return "".join(tex_list)


def transform_figures_1col(tex: str) -> str:
    """Transform centered images into clean 1-column figure environments."""
    fig_pattern = re.compile(
        r"(?:\\begin\{center\}\s*)?"
        r"\\pandocbounded\{\\includegraphics.*?\{(paper_assets/[^{}\s]+)\}\}\s*"
        r"(?:\\end\{center\}\s*)?"
        r"\\textbf\{Hình\s*(\d+)\.\}\s*(.*?)(?=\n\n|\n\\|\Z)",
        re.DOTALL,
    )

    def replace_fig(m: re.Match[str]) -> str:
        path = m.group(1)
        num = m.group(2)
        caption = m.group(3).strip()
        caption_clean = re.sub(r"\s+", " ", caption)
        caption_clean = re.sub(r"(?<!\\)%", r"\%", caption_clean)
        return (
            f"\\begin{{figure}}[htbp]\n"
            f"\\centering\n"
            f"\\includegraphics[width=0.88\\textwidth]{{{path}}}\n"
            f"\\caption{{{caption_clean}}}\n"
            f"\\label{{fig:sec_fig{num}}}\n"
            f"\\end{{figure}}\n"
        )

    return fig_pattern.sub(replace_fig, tex)


def create_section_header_tex(header_left: str) -> Path:
    """Generate a custom 1-column LaTeX header with dedicated fancyhdr."""
    header_content = (
        "% 1-column header for CystoDS section report\n"
        "\\PassOptionsToPackage{table,xcdraw}{xcolor}\n"
        "\\usepackage[section]{placeins}\n"
        "\\usepackage{caption}\n"
        "\\usepackage{fancyhdr}\n"
        "\\usepackage{titlesec}\n"
        "\\usepackage{enumitem}\n"
        "\\usepackage{seqsplit}\n"
        "\\usepackage{booktabs}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{microtype}\n"
        "\\usepackage{adjustbox}\n"
        "\\usepackage{pdflscape}\n"
        "\\usepackage{array}\n"
        "\n"
        "\\definecolor{CystoBlue}{HTML}{1F4E79}\n"
        "\\definecolor{CystoTeal}{HTML}{2A7F62}\n"
        "\\definecolor{CystoGray}{HTML}{4A5568}\n"
        "\\definecolor{CystoDark}{HTML}{1A202C}\n"
        "\n"
        "\\graphicspath{{../../Docs/}{Docs/}{paper_assets/}{../paper_assets/}}\n"
        "\n"
        "\\captionsetup{font=small,labelfont={bf,color=CystoBlue}}\n"
        "\\captionsetup[table]{position=top,skip=3pt}\n"
        "\\captionsetup[figure]{position=bottom,skip=4pt}\n"
        "\n"
        "\\renewcommand{\\figurename}{Hình}\n"
        "\\renewcommand{\\tablename}{Bảng}\n"
        "\n"
        "\\titleformat{\\section}{\\Large\\bfseries\\color{CystoBlue}}{\\thesection.}{0.4em}{}\n"
        "\\titleformat{\\subsection}{\\large\\bfseries\\color{CystoTeal}}{\\thesubsection.}{0.4em}{}\n"
        "\\titleformat{\\subsubsection}{\\normalsize\\bfseries\\color{CystoDark}}{\\thesubsubsection.}{0.4em}{}\n"
        "\n"
        "\\setlength{\\parindent}{1.0em}\n"
        "\\setlength{\\parskip}{0.35em plus 0.05em minus 0.05em}\n"
        "\\setlist{nosep,leftmargin=1.5em,topsep=0.3em}\n"
        "\n"
        "\\pagestyle{fancy}\n"
        "\\fancyhf{}\n"
        f"\\fancyhead[L]{{\\small\\color{{CystoGray}}{header_left}}}\n"
        "\\fancyhead[R]{\\small\\color{CystoGray}CystoDS 2026}\n"
        "\\fancyfoot[C]{\\small\\color{CystoGray}\\thepage}\n"
        "\\setlength{\\headheight}{14pt}\n"
        "\\renewcommand{\\headrulewidth}{0.35pt}\n"
        "\n"
        "\\AtBeginDocument{%\n"
        "  \\hypersetup{\n"
        "    colorlinks=true,\n"
        "    linkcolor=CystoBlue,\n"
        "    urlcolor=CystoTeal,\n"
        "    citecolor=CystoBlue,\n"
        "  }%\n"
        "}\n"
    )
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    header_path = TMP_DIR / f"header_{re.sub(r'[^a-zA-Z0-9]', '_', header_left)[:25]}.tex"
    header_path.write_text(header_content, encoding="utf-8")
    return header_path


def build_single_section(sec: dict[str, Any], full_markdown: str) -> Path:
    """Extract section markdown and compile into 1-column PDF."""
    match = re.search(sec["regex"], full_markdown)
    if not match:
        raise ValueError(f"Could not extract section {sec['id']} with regex {sec['regex']}")

    sec_content = match.group(1).strip()

    # Prepend Title and metadata
    sec_md = f"# {sec['title']}\n\n## {sec['subtitle']}\n\n{sec_content}\n"

    # Normalize unicode math
    sec_md = sec_md.replace("–", "--").replace("—", "---")
    sec_md = sec_md.replace("3×10⁻⁴", "$3\\times10^{-4}$")
    sec_md = sec_md.replace("7,5×10⁻⁵", "$7{,}5\\times10^{-5}$")

    tmp_md = TMP_DIR / f"{sec['filename_stem']}.md"
    tmp_md.write_text(sec_md, encoding="utf-8")

    tex_out = OUTPUT_DIR / f"{sec['filename_stem']}.tex"
    pdf_out = OUTPUT_DIR / f"{sec['filename_stem']}.pdf"

    header_tex = create_section_header_tex(sec["header_left"])

    pandoc_bin = pypandoc.get_pandoc_path()
    command = [
        pandoc_bin,
        str(tmp_md),
        "--from=markdown+pipe_tables+grid_tables+raw_tex+tex_math_dollars",
        "--to=latex",
        "--standalone",
        f"--include-in-header={header_tex}",
        "--metadata",
        f"title={sec['title']}",
        "--metadata",
        f"subtitle={sec['subtitle']}",
        "--metadata",
        "date=Báo cáo nghiên cứu 2026",
        "--variable",
        "documentclass=article",
        "--variable",
        "papersize=a4",
        "--variable",
        "fontsize=11pt",
        "--variable",
        "geometry:top=2.0cm,bottom=2.0cm,left=2.0cm,right=2.0cm",
        "--variable",
        "mainfont=Times New Roman",
        "--variable",
        "sansfont=Arial",
        "--variable",
        "monofont=Menlo",
        "--output",
        str(tex_out),
    ]

    subprocess.run(command, cwd=ROOT, check=True)

    tex = tex_out.read_text(encoding="utf-8")
    tex = transform_tables_1col(tex)
    tex = transform_figures_1col(tex)
    tex = tex.replace("→", "→\\allowbreak{}")
    tex_out.write_text(tex, encoding="utf-8")

    tectonic = shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError("Không tìm thấy Tectonic")

    subprocess.run(
        [
            tectonic,
            "--keep-logs",
            "--outdir",
            str(OUTPUT_DIR),
            str(tex_out),
        ],
        cwd=PAPER_DIR,
        check=True,
    )

    print(f"✅ Generated 1-column PDF: {pdf_out} ({pdf_out.stat().st_size / 1024 / 1024:.2f} MB)")
    return pdf_out


def main() -> None:
    full_markdown = SOURCE.read_text(encoding="utf-8")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure paper assets exist in output
    assets_src = PAPER_DIR / "paper_assets"
    assets_dst = OUTPUT_DIR / "paper_assets"
    if assets_src.is_dir() and not assets_dst.exists():
        shutil.copytree(assets_src, assets_dst)

    print("=" * 70)
    print("▶ BẮT ĐẦU BIÊN DỊCH CÁC PHẦN BÁO CÁO 1-COLUMN CHO CYSTODS")
    print("=" * 70)

    for sec in SECTION_DEFS:
        print(f"\n[Compiling Section] {sec['id']}...")
        build_single_section(sec, full_markdown)

    print("\n" + "=" * 70)
    print("🎉 HOÀN THÀNH TÁCH VÀ BIÊN DỊCH TOÀN BỘ 3 FILE PDF (1-COLUMN)!")
    print(f"📂 Thư mục chứa: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
