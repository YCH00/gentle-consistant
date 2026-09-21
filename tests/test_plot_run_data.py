import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import plot_tb_seed_average as plot


@pytest.mark.parametrize('compare', [False, True])
def test_plot_export_keeps_original_seed_points_and_saved_configs(tmp_path, monkeypatch, compare):
    experiments = ['global', 'semantic'] if compare else ['global']
    summaries = {}
    for experiment in experiments:
        run = tmp_path / experiment
        run.mkdir()
        variant = dict(algo_params={'virtual_transition_use_recon_weight': False})
        (run / 'variant.json').write_text(json.dumps(variant), encoding='utf-8')
        summaries[experiment] = dict(
            experiment=experiment, label=experiment, runs=[('seed0', run)],
            steps=np.array([0, 1, 2]), mean=np.array([1., 2., 3.]),
            std=np.zeros(3), counts=np.ones(3), matrix=np.array([[1., 2., 3.]]),
            # Step 1 in the plot matrix is interpolated; it must not be exported as raw data.
            series=[dict(seed='seed0', run_dir=run, steps=np.array([0, 2]), values=np.array([1., 3.]))],
        )
    monkeypatch.setattr(plot, 'find_seed_runs', lambda root, exp, *a: summaries[exp]['runs'])
    monkeypatch.setattr(plot, 'summarize_experiment', lambda args, exp, label, **kw: summaries[exp])
    monkeypatch.setattr(plot, 'plot_average', lambda **kwargs: None)
    monkeypatch.setattr(plot, 'plot_compare', lambda **kwargs: None)
    args = SimpleNamespace(
        experiment=experiments, label=None, root=tmp_path, seed_pattern='seed*', pick='error',
        list_tags=False, output=str(tmp_path / 'compare.png'), csv_output=None,
        tag='Return/test', title=None, ylabel=None, smooth_window=10, align='interpolate',
        no_std_shade=False, show_seeds=False, export_run_data=True,
    )
    (plot.run_compare_experiments if compare else plot.run_single_experiment)(args)
    with (tmp_path / 'compare_seeds.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 * len(experiments)
    assert [int(row['step']) for row in rows] == [0, 2] * len(experiments)
    assert [float(row['value']) for row in rows] == [1., 3.] * len(experiments)
    manifest = json.loads((tmp_path / 'compare_runs.json').read_text())
    assert len(manifest['runs']) == len(experiments)
    assert manifest['seed_values_smoothed'] is False
    assert manifest['aggregate_std_ddof'] == 0
    assert manifest['runs'][0]['variant'] == variant
    assert manifest['plot_smooth_window'] == 10


def test_missing_variant_is_reported_without_substituting_current_config(tmp_path, capsys):
    summary = dict(experiment='run', label='run', series=[
        dict(seed='seed0', run_dir=tmp_path, steps=[0], values=[2.0]),
    ])
    args = SimpleNamespace(tag='Return/test', align='intersection', smooth_window=1)
    plot.save_run_data(tmp_path / 'result.csv', [summary], args)
    manifest = json.loads((tmp_path / 'result_runs.json').read_text())
    assert manifest['runs'][0]['variant'] is None
    assert 'missing run-time configuration' in capsys.readouterr().out
