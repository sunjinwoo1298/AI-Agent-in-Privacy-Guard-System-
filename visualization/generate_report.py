#!/usr/bin/env python3
"""Generate a combined PDF report with analysis text and comparison plots.

Creates `results/plots/comparison/report.pdf` containing:
- First page: contents of `results/analysis.txt` (if present)
- Following pages: comparison plot images from `results/plots/comparison`
"""

import os
import textwrap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


def generate_report(analysis_txt: str = "results/analysis.txt",
                    comparison_dir: str = "results/plots/comparison",
                    out_pdf: str = "results/plots/comparison/report.pdf") -> str:
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)

    # Ordered list of expected comparison images (best-effort)
    images_order = [
        "total_latency_thread_vs_process.png",
        "coordination_tax_thread_vs_process.png",
        "total_throughput_thread_vs_process.png",
        "efficiency_thread_vs_process.png",
    ]

    # Fallback: include any png in the folder if expected ones missing
    available = {f for f in os.listdir(comparison_dir)} if os.path.exists(comparison_dir) else set()

    with PdfPages(out_pdf) as pdf:
        # First page: analysis text
        if os.path.exists(analysis_txt):
            with open(analysis_txt, "r", encoding="utf-8") as fh:
                text = fh.read()
        else:
            text = "Analysis summary not found. Run `analysis/analyze_results.py` first."

        wrapped = "\n".join(textwrap.wrap(text, 100))
        fig = plt.figure(figsize=(8.27, 11.69))  # A4-ish
        fig.text(0.01, 0.99, wrapped, va="top", fontsize=10, family="monospace")
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Next pages: ordered images
        if os.path.exists(comparison_dir):
            for img_name in images_order:
                path = os.path.join(comparison_dir, img_name)
                if not os.path.exists(path):
                    continue
                img = mpimg.imread(path)
                fig = plt.figure(figsize=(8.27, 11.69))
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Add any remaining pngs that were not in images_order
            for fname in sorted(available):
                if not fname.lower().endswith(".png"):
                    continue
                if fname in images_order:
                    continue
                path = os.path.join(comparison_dir, fname)
                img = mpimg.imread(path)
                fig = plt.figure(figsize=(8.27, 11.69))
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print(f"Saved {out_pdf}")
    return out_pdf


if __name__ == "__main__":
    generate_report()
