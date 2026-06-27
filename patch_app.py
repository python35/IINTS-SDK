import re
from pathlib import Path

app_path = Path('/Volumes/Samsung SSD 990 EVO Plus Media/IINTS-AF/IINTS-SDK-main/src/iints_desktop/qt_app.py')
code = app_path.read_text()

# 1. Add imports
code = re.sub(
    r'(from iints_desktop.results import ResultPreview, load_results_preview)',
    r'\1\nfrom iints_desktop.fetcher import fetch_alphafold_structure\nfrom iints_desktop.render_3dmol import generate_3dmol_html',
    code
)

# 2. Add Worker
worker_code = """
    class AlphaFoldFetchWorker(QObject):
        finished = Signal(object)
        failed = Signal(str)

        def __init__(self, uniprot_id: str):
            super().__init__()
            self.uniprot_id = uniprot_id

        @Slot()
        def run(self) -> None:
            try:
                from pathlib import Path
                out_dir = Path("results") / "structural"
                cif_path = fetch_alphafold_structure(self.uniprot_id, out_dir)
                html_path = generate_3dmol_html(cif_path, out_dir)
                self.finished.emit((cif_path, html_path, self.uniprot_id))
            except Exception as exc:
                self.failed.emit(str(exc))
"""

code = code.replace(
    'class BiologyWorker(QObject):',
    worker_code + '\n    class BiologyWorker(QObject):'
)

app_path.write_text(code)
print("Patch 1 complete")
