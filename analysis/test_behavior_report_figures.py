import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest
import sys
from analysis.behavior_report_figures import overview
from scripts.eigenposture_after_tracking import check_environment


@pytest.mark.parametrize('side', ['left', 'right'])
def test_overview_renders_the_selected_colony(side):
    figure = overview(n_states=7, n_classes=3, side=side)
    figure.canvas.draw()
    labels = '\n'.join(text.get_text() for axis in figure.axes for text in axis.texts)
    assert f'{side.capitalize()} colony' in labels
    assert '7-state reference dictionary' in labels
    assert '3 classes' in labels
    plt.close(figure)


def test_submission_smoke_check_renders_the_report_api_from_any_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    check_environment(sys.executable)
