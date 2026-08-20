#!/usr/bin/env python3
"""Build the CystoDS Vietnamese manuscript as a publication-ready 2-column LaTeX and PDF."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pypandoc


PAPER_DIR = Path(__file__).resolve().parent
DOCS = PAPER_DIR.parent
ROOT = DOCS.parent
SOURCE = PAPER_DIR / "paper.md"
TMP_DIR = DOCS / "tmp" / "pdfs"
OUTPUT_DIR = DOCS / "output" / "pdf"
TEX_PATH = OUTPUT_DIR / "CystoDS_Hierarchical_Long_Tailed_VI.tex"
PDF_PATH = OUTPUT_DIR / "CystoDS_Hierarchical_Long_Tailed_VI.pdf"

TITLE = (
    "Phân loại phân cấp tổn thương bàng quang trong nội soi "
    "trên dữ liệu mất cân bằng dài đuôi CystoDS"
)
SUBTITLE = "Phương pháp tinh chỉnh hai giai đoạn tách rời trên hold-out độc lập theo bệnh nhân"

TABLE_CAPTIONS = [
    "Tổng hợp và đối chiếu các công trình nghiên cứu phân loại nội soi bàng quang liên quan.",
    "Thống kê mẫu và bệnh nhân bộ dữ liệu CystoDS theo 3 tầng phả hệ.",
    "Kết quả sàng lọc 4 họ kiến trúc mạng xương sống (Stage 10) qua 3 hold-out splits độc lập.",
    "Kết quả sàng lọc 7 hàm mất mát xử lý phân bố đuôi dài (Stage 20) qua 3 hold-out splits độc lập.",
    "Kết quả đánh giá mô hình đề xuất 3S-HFT v3.1 trên tập Validation (3-Fold Patient-Disjoint).",
    "Kết quả kiểm định độc lập của mô hình đề xuất 3S-HFT v3.1 trên tập Test (3-Fold Patient-Disjoint).",
    "Kết quả thực nghiệm triệt tiêu thành phần (Ablation Studies -- Stage 40) trên 10 biến thể qua 3 splits.",
    "Khảo sát vị trí trích xuất đặc trưng đa tầng: Shared Late-Stage so với Multi-Stage Intermediate Heads.",
    "Hiệu năng chi tiết theo từng lớp thô (per-class coarse metrics) của mô hình phân cấp.",
]


def prepare_markdown() -> Path:
    text = SOURCE.read_text(encoding="utf-8")
    text, replacements = re.subn(
        r"\A# [^\n]+\n\n## [^\n]+\n\n",
        "",
        text,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError("Không nhận diện được title/subtitle đầu Markdown")

    # Normalize unicode dashes and math notation
    text = text.replace("–", "--").replace("—", "---")
    text = text.replace("3×10⁻⁴", "$3\\times10^{-4}$")
    text = text.replace("7,5×10⁻⁵", "$7{,}5\\times10^{-5}$")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    prepared = TMP_DIR / "paper_for_latex.md"
    prepared.write_text(text, encoding="utf-8")
    return prepared


def clean_cell_tex(cell: str) -> str:
    """Strip minipages and clean spacing in TeX table cells."""
    cell = re.sub(
        r"\\begin\{minipage\}\[[^\]]*\]\{\\linewidth\}\s*(?:\\raggedright|\\raggedleft|\\centering)?\s*(.*?)\s*\\end\{minipage\}",
        r"\1",
        cell,
        flags=re.DOTALL,
    )
    return cell.strip()


def transform_tables(tex: str) -> str:
    """Transform all longtable environments into clean publication table floats."""
    starts = [m.start() for m in re.finditer(r"\\begin\{longtable\}", tex)]
    ends = [m.end() for m in re.finditer(r"\\end\{longtable\}", tex)]

    if not starts:
        return tex

    replacements = []

    for idx, (s, e) in enumerate(zip(starts, ends)):
        pre = tex[max(0, s - 150) : s]
        post = tex[e : min(len(tex), e + 100)]

        b_start = s
        if "\\clearpage\\begin{landscape}" in pre:
            pos = pre.rfind("\\clearpage\\begin{landscape}")
            b_start = s - (len(pre) - pos)
        elif "{\\def\\LTcaptype" in pre:
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

        post = tex[e : min(len(tex), e + 120)]
        if "\\end{landscape}\\clearpage" in post:
            pos = post.find("\\end{landscape}\\clearpage") + len(
                "\\end{landscape}\\clearpage"
            )
            b_end = max(b_end, e + pos)

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

            if idx in (3, 11, 18, 20):  # text-heavy tables
                align_chars = ["l"] * num_cols

            align_str = "".join(align_chars)

            cap = (
                TABLE_CAPTIONS[idx]
                if idx < len(TABLE_CAPTIONS)
                else f"Bảng {idx+1}."
            )
            cap = re.sub(r"(?<!\\)%", r"\%", cap)

            env_name = "table"
            width_str = "\\linewidth"
            pos_spec = "[htbp]"

            new_tex = (
                f"\\begin{{{env_name}}}{pos_spec}\n"
                f"\\centering\n"
                f"\\caption{{{cap}}}\n"
                f"\\label{{tab:table{idx+1}}}\n"
                f"\\vspace{{2pt}}\n"
                f"\\resizebox{{{width_str}}}{{!}}{{\n"
                f"\\begin{{tabular}}{{{align_str}}}\n"
                f"\\toprule\n"
                f"{header_str}\n"
                f"\\midrule\n"
                f"{body_raw}\n"
                f"\\bottomrule\n"
                f"\\end{{tabular}}\n"
                f"}}\n"
                f"\\end{{{env_name}}}"
            )
            replacements.append((b_start, b_end, new_tex))

    # Apply replacements from back to front
    replacements.sort(key=lambda x: x[0], reverse=True)
    tex_list = list(tex)
    for b_start, b_end, new_code in replacements:
        tex_list[b_start:b_end] = list(new_code)

    return "".join(tex_list)


def transform_figures(tex: str) -> str:
    """Transform centered images into single-column floating LaTeX figure environments."""
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
        caption = m.group(3).strip().replace("\n", " ")
        caption = re.sub(r"(?<!\\)%", r"\%", caption)

        env = "figure"
        width = "\\linewidth"
        pos = "[htbp]"

        return (
            f"\\begin{{{env}}}{pos}\n"
            f"\\centering\n"
            f"\\includegraphics[width={width},keepaspectratio]{{{path}}}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{fig:fig{num}}}\n"
            f"\\end{{{env}}}\n"
        )

    return fig_pattern.sub(replace_fig, tex)


def transform_header_and_abstract(tex: str) -> str:
    """Format the paper title, subtitle, metadata, abstract, and keywords into a full-width header block."""
    # Find abstract text block inside generated LaTeX
    abstract_match = re.search(
        r"\\subsection\{Tóm tắt\}\\label\{tuxf3m-tuxf5t\}\n\n(.*?)(?=\\section|\Z)",
        tex,
        re.DOTALL,
    )
    if not abstract_match:
        return tex

    abstract_text = abstract_match.group(1).strip()
    # Remove abstract section from body
    tex = tex[: abstract_match.start()] + tex[abstract_match.end() :]

    # Clean up abstract text formatting
    abstract_text = re.sub(
        r"\*\*Từ khóa:\*\*\s*(.*)",
        r"\\vspace{0.4em}\n\\textbf{Từ khóa:} \\textit{\1}",
        abstract_text,
    )

    header_block = (
        "\\twocolumn[\n"
        "  \\begin{center}\n"
        f"    {{\\LARGE\\bfseries\\color{{CystoBlue}} {TITLE} \\par}}\n"
        "    \\vspace{0.5em}\n"
        f"    {{\\large\\itshape\\color{{CystoTeal}} {SUBTITLE} \\par}}\n"
        "    \\vspace{0.7em}\n"
        "    {\\small \\textbf{Bản thảo nghiên cứu nội bộ} \\quad|\\quad \\textbf{Phiên bản:} 03-08-2026 \\quad|\\quad \\textbf{Giao thức:} Hold-out 70/15/15 \\par}\n"
        "    \\vspace{0.9em}\n"
        "  \\end{center}\n"
        "  \\begin{adjustwidth}{1.2cm}{1.2cm}\n"
        "    \\small\n"
        f"    {abstract_text}\n"
        "  \\end{adjustwidth}\n"
        "  \\vspace{1.0em}\n"
        "  {\\color{CystoBlue}\\hrule height 0.6pt}\n"
        "  \\vspace{1.2em}\n"
        "]\n"
    )

    # Replace \maketitle with header_block
    tex = re.sub(r"\\maketitle\n?", header_block, tex, count=1)
    tex = re.sub(r"\\tableofcontents\n?", "", tex)

    return tex


def build_latex(prepared: Path) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pandoc = pypandoc.get_pandoc_path()
    command = [
        pandoc,
        str(prepared),
        "--from=markdown-implicit_figures+raw_tex+tex_math_single_backslash",
        "--to=latex",
        "--standalone",
        "--shift-heading-level-by=-1",
        f"--resource-path={PAPER_DIR}:{PAPER_DIR / 'paper_assets'}:{DOCS}:{ROOT}",
        f"--lua-filter={PAPER_DIR / 'pandoc_paper_layout.lua'}",
        f"--include-in-header={PAPER_DIR / 'paper_latex_header.tex'}",
        "--metadata",
        f"title={TITLE}",
        "--metadata",
        f"subtitle={SUBTITLE}",
        "--metadata",
        "date=Phiên bản 03-08-2026",
        "--variable",
        "documentclass=article",
        "--variable",
        "classoption=twocolumn",
        "--variable",
        "papersize=a4",
        "--variable",
        "fontsize=10pt",
        "--variable",
        "geometry:top=1.8cm,bottom=1.8cm,left=1.5cm,right=1.5cm",
        "--variable",
        "mainfont=Times New Roman",
        "--variable",
        "sansfont=Arial",
        "--variable",
        "monofont=Menlo",
        "--output",
        str(TEX_PATH),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    tex = TEX_PATH.read_text(encoding="utf-8")

    # Apply structural transformations
    tex = transform_header_and_abstract(tex)
    tex = transform_tables(tex)
    tex = transform_figures(tex)

    # Permit line breaks for long identifiers and hashes
    tex = tex.replace("→", "→\\allowbreak{}")
    tex = re.sub(
        r"\\texttt\{([0-9a-f]{40,64})\}",
        r"\\texttt{\\seqsplit{\1}}",
        tex,
    )

    for label in (
        "HighGradePapillary",
        "LowGradePapillary",
        "Anatomical landmarks",
        "Multi-task binary+coarse+fine",
    ):
        breakable = re.sub(r"(?<=[a-z])(?=[A-Z])", r"\\allowbreak{}", label)
        breakable = breakable.replace("+", "+\\allowbreak{}")
        tex = tex.replace(label, breakable)

    local_link = re.compile(
        r"\\href\{((?!https?://)[^{}]+)\}\{\\texttt\{([^{}]+)\}\}"
    )

    def make_local_link_breakable(match: re.Match[str]) -> str:
        target = match.group(1)
        display = match.group(2).replace(r"\_", "_")
        return rf"\href{{{target}}}{{\nolinkurl{{{display}}}}}"

    tex = local_link.sub(make_local_link_breakable, tex)
    TEX_PATH.write_text(tex, encoding="utf-8")


def compile_pdf() -> None:
    tectonic = shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError("Không tìm thấy Tectonic")
    subprocess.run(
        [
            tectonic,
            "--keep-logs",
            "--outdir",
            str(OUTPUT_DIR),
            str(TEX_PATH),
        ],
        cwd=PAPER_DIR,
        check=True,
    )
    log_path = OUTPUT_DIR / f"{TEX_PATH.stem}.log"
    if log_path.exists():
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(log_path), str(TMP_DIR / log_path.name))
    if not PDF_PATH.exists() or PDF_PATH.stat().st_size == 0:
        raise RuntimeError("Tectonic không tạo được PDF")


def main() -> None:
    prepared = prepare_markdown()
    build_latex(prepared)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    assets_src = PAPER_DIR / "paper_assets"
    assets_dst = OUTPUT_DIR / "paper_assets"
    if assets_src.is_dir():
        if assets_dst.exists():
            shutil.rmtree(assets_dst)
        shutil.copytree(assets_src, assets_dst)
    compile_pdf()
    print(f"LaTeX file: {TEX_PATH}")
    print(f"PDF file:   {PDF_PATH}")


if __name__ == "__main__":
    main()
