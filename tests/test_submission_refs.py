"""A floated appendix table must not be imported as a main-paper label."""
import importlib.util
from pathlib import Path


def test_appendix_labels_are_classified_by_page_not_aux_write_order():
    path = Path(__file__).resolve().parents[1] / 'paper' / 'prepare_submission.py'
    spec = importlib.util.spec_from_file_location('prepare_submission', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    aux = r'''\newlabel{eq:rule}{{2}{3}{}{equation.2}{}}
\bibcite{source}{{1}{2025}{{Author}}{{}}}
\newlabel{tab:calibration}{{5}{12}{Caption}{table.5}{}}
\newlabel{sec:appendix-start}{{9}{12}{}{section*.1}{}}
\newlabel{sec:measurement}{{A}{12}{}{appendix.A}{}}
'''
    main, appendix = map(''.join, module.split_labels(aux))
    assert 'eq:rule' in main and 'source' in main
    assert 'tab:calibration' not in main
    assert 'tab:calibration' in appendix and 'sec:measurement' in appendix
    assert 'sec:appendix-start' not in main + appendix
