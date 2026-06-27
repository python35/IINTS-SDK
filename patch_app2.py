import re
from pathlib import Path

app_path = Path('/Volumes/Samsung SSD 990 EVO Plus Media/IINTS-AF/IINTS-SDK-main/src/iints_desktop/qt_app.py')
code = app_path.read_text()

# Inject UI in selector_layout
ui_injection = """
            self.custom_uniprot_input = QLineEdit()
            self.custom_uniprot_input.setPlaceholderText("UniProt ID (e.g. Q13131)")
            self.fetch_uniprot_button = QPushButton("Fetch Live")
            self.fetch_uniprot_button.clicked.connect(self._fetch_custom_uniprot)
            fetch_layout = QHBoxLayout()
            fetch_layout.addWidget(self.custom_uniprot_input)
            fetch_layout.addWidget(self.fetch_uniprot_button)
            selector_layout.addWidget(QLabel("Fetch custom:"), 2, 0)
            selector_layout.addLayout(fetch_layout, 2, 1)
"""
code = code.replace(
    'selector_layout.setColumnStretch(1, 1)',
    'selector_layout.setColumnStretch(1, 1)\n' + ui_injection
)

# Inject 3d web viewer in viewer_box
viewer_injection = """
            if _QWEBENGINE_VIEW is not None and os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                self.molecule_web_view = _QWEBENGINE_VIEW()
                self.molecule_web_view.setMinimumHeight(260)
                viewer_layout.addWidget(self.molecule_web_view, stretch=1)
                self.molecule_viewer.hide() # fallback hidden
            else:
                self.molecule_web_view = None
"""
code = code.replace(
    'viewer_layout.addWidget(self.molecule_viewer, stretch=1)',
    'viewer_layout.addWidget(self.molecule_viewer, stretch=1)\n' + viewer_injection
)

app_path.write_text(code)
print("Patch 2 complete")
