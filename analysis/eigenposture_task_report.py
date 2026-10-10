"""Run the right-colony interactive cells headlessly and bundle the figures."""
import argparse
import os
from pathlib import Path
import runpy
import shutil


def run_analysis(output):
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["ANTS_ROLE_OUTPUT"] = str(output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.close("all")
    output = Path(output)
    source = Path(__file__).with_name("eigenposture_interactive.py")
    namespace = runpy.run_path(str(source))
    shutil.copy2(source, output / "eigenposture_interactive.py")
    shutil.copy2(source.with_suffix(".md"), output / "README.md")
    with PdfPages(output / "right_behavior_classes.pdf") as pdf:
        for number in plt.get_fignums():
            pdf.savefig(plt.figure(number), bbox_inches="tight")
    plt.close("all")
    return namespace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Full-body model packet; defaults to the block's published packet")
    args = parser.parse_args()
    if args.source is not None:
        os.environ["ANTS_BEHAVIOR_SOURCE"] = str(args.source)
    run_analysis(args.output)
