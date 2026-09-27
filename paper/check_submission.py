"""Check the body page limit and visible identifying text in the upload PDFs."""
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
aux = (HERE / 'main-submission.aux').read_text()
match = re.search(r'\\newlabel\{sec:body-end\}\{\{[^}]*\}\{(\d+)\}', aux)
if match is None:
    raise SystemExit('Missing body-end label; compile the submission first.')
page = int(match[1])
if page > 10:
    raise SystemExit(f'Body ends on page {page}; the limit is 10.')
for name in ('main-submission.pdf', 'appendix-anon.pdf'):
    log = (HERE / name.replace('.pdf', '.log')).read_text()
    bad = [phrase for phrase in ('undefined on input line', 'multiply defined',
                                  'multiply-defined', 'destination with the same identifier',
                                  r'Overfull \hbox', r'Overfull \vbox') if phrase in log]
    if bad:
        raise SystemExit(f'{name}: build warnings: {bad}')
    text = subprocess.check_output(['pdftotext', str(HERE / name), '-'], text=True)
    for identifying in ('shardes', 'andres', 'gmx.net'):
        if identifying in text.lower():
            raise SystemExit(f'{name}: identifying text {identifying!r}')
print(f'Submission body ends on page {page}; both upload PDFs pass the text anonymity check.')
