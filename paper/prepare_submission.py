"""Separate external labels so each upload imports only its companion's labels."""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
LABEL = re.compile(r'\\newlabel\{([^}]+)\}\{\{[^}]*\}\{(\d+)\}')


def split_labels(aux):
    """Classify by rendered page: floats can be written before a section's label."""
    labels = {match[1]: int(match[2]) for match in LABEL.finditer(aux)}
    boundary = labels['sec:appendix-start']
    main, appendix = [], []
    for line in aux.splitlines(keepends=True):
        match = LABEL.match(line)
        if match:
            if match[1] != 'sec:appendix-start':
                (appendix if int(match[2]) >= boundary else main).append(line)
        elif line.startswith(r'\bibcite'):
            main.append(line)
    if not main or not appendix:
        raise ValueError('Compile submission-all before preparing companion references.')
    return main, appendix


def main():
    parts = split_labels((HERE / 'submission-all.aux').read_text())
    for stem, lines in zip(('main', 'appendix'), parts):
        (HERE / f'submission-{stem}-refs.aux').write_text(''.join(lines))


if __name__ == '__main__':
    main()
