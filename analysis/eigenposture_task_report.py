"""Run the selected-colony interactive cells headlessly and bundle the figures."""
import argparse
import os
from pathlib import Path
import runpy
import shutil
import json
from analysis.behavior_report_figures import add_pdf_description


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
    descriptions=[]
    with PdfPages(output / (namespace["SIDE"]+"_behavior_classes.pdf")) as pdf:
        for number in plt.get_fignums():
            figure=plt.figure(number)
            text=add_pdf_description(figure)
            pdf.savefig(figure, bbox_inches="tight")
            text.remove()
            descriptions.append(dict(figure=number,title=figure._report_title,description=figure._report_caption,
                                     section=figure._report_section))
    (output/'figure_descriptions.json').write_text(json.dumps(descriptions,indent=2)+'\n')
    if any(row['section']=='02_state_reduction' for row in descriptions):
        sections={'02_state_reduction','02_state_repeat_support','05_ant_methods',
                  '05_class_reduction','05_class_state_sensitivity'}
        with PdfPages(output/'state_class_reduction_summary.pdf') as pdf:
            for number in plt.get_fignums():
                figure=plt.figure(number)
                if (figure._report_section in sections or
                    (figure._report_section=='04_occupancy' and figure._report_index==0)):
                    text=add_pdf_description(figure);pdf.savefig(figure,bbox_inches='tight');text.remove()
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
