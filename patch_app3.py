import re
from pathlib import Path

app_path = Path('/Volumes/Samsung SSD 990 EVO Plus Media/IINTS-AF/IINTS-SDK-main/src/iints_desktop/qt_app.py')
code = app_path.read_text()

# Inject the fetch and render logic methods
methods_injection = """
        def _fetch_custom_uniprot(self) -> None:
            uniprot_id = self.custom_uniprot_input.text().strip().upper()
            if not uniprot_id:
                QMessageBox.information(self, "IINTS-AF Desktop", "Please enter a UniProt ID.")
                return
            
            self.fetch_uniprot_button.setEnabled(False)
            self.molecule_structure_status.setText(f"Fetching {uniprot_id} from AlphaFold...")
            
            thread = QThread(self)
            worker = AlphaFoldFetchWorker(uniprot_id)
            worker.moveToThread(thread)
            
            thread.started.connect(worker.run)
            worker.finished.connect(self._handle_fetch_success)
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            worker.failed.connect(self._handle_fetch_error)
            worker.failed.connect(thread.quit)
            worker.failed.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            
            self.af_fetch_thread = thread
            self.af_fetch_worker = worker
            thread.start()

        @Slot(object)
        def _handle_fetch_success(self, result: object) -> None:
            self.fetch_uniprot_button.setEnabled(True)
            cif_path, html_path, uniprot_id = result
            
            # Create a dynamic MoleculeAsset
            from iints_desktop.molecules import MoleculeAsset
            new_mol = MoleculeAsset(
                key=uniprot_id.lower(),
                title=f"Custom: {uniprot_id}",
                uniprot_id=uniprot_id,
                image_path=Path("nonexistent.png"),
                structure_path=cif_path,
                explanation="Dynamically fetched AlphaFold structure.",
                sdk_link="Custom research target.",
                pae_target=uniprot_id.lower(),
                pae_note="Dynamic PAE heatmap target."
            )
            self.molecules.append(new_mol)
            self.molecule_selector.addItem(f"{new_mol.title} (UniProt {new_mol.uniprot_id})", new_mol.key)
            self.molecule_selector.setCurrentIndex(self.molecule_selector.count() - 1)
            self.molecule_structure_status.setText(f"Successfully loaded {uniprot_id}.")

        @Slot(str)
        def _handle_fetch_error(self, details: str) -> None:
            self.fetch_uniprot_button.setEnabled(True)
            self.molecule_structure_status.setText(f"Fetch failed: {details}")
            self.molecule_structure_status.setStyleSheet("background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa;")
"""

code = code.replace(
    'def _selected_molecule(self) -> MoleculeAsset:',
    methods_injection + '\n        def _selected_molecule(self) -> MoleculeAsset:'
)

# Update _on_molecule_changed to use 3Dmol.js if available
code = code.replace(
    'self.molecule_structure_status.setStyleSheet("")',
    """self.molecule_structure_status.setStyleSheet("")

            # 3Dmol.js rendering path
            if getattr(self, "molecule_web_view", None) is not None:
                from iints_desktop.render_3dmol import generate_3dmol_html
                try:
                    out_dir = Path("results") / "structural"
                    html_path = generate_3dmol_html(molecule.structure_path, out_dir)
                    self.molecule_web_view.setUrl(QUrl.fromLocalFile(str(html_path.absolute())))
                    self.molecule_viewer.hide()
                    self.molecule_web_view.show()
                except Exception as e:
                    print(f"3Dmol.js render failed: {e}")
                    self.molecule_web_view.hide()
                    self.molecule_viewer.show()
            """
)

app_path.write_text(code)
print("Patch 3 complete")
