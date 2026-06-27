"""Generates a local 3Dmol.js interactive HTML view for AlphaFold structures."""

from __future__ import annotations

from pathlib import Path


def generate_3dmol_html(structure_path: Path, output_dir: Path) -> Path:
    """Generate an HTML file that renders the mmCIF structure using 3Dmol.js."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{structure_path.stem}_3dmol.html"
    
    # Read the CIF data and escape it for JS embedding
    try:
        cif_data = structure_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Cannot read structure {structure_path}: {exc}") from exc
        
    cif_js = cif_data.replace('`', '\\`').replace('$', '\\$')
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>3Dmol.js AlphaFold Render</title>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; background-color: #071827; overflow: hidden; }}
        #container {{ width: 100vw; height: 100vh; position: relative; }}
    </style>
</head>
<body>
    <div id="container"></div>
    <script>
        $(function() {{
            let element = $('#container');
            let config = {{ backgroundColor: '#071827' }};
            let viewer = $3Dmol.createViewer(element, config);
            
            let cifData = `{cif_js}`;
            
            // Add model and set styling
            viewer.addModel(cifData, "cif");
            
            // Color by pLDDT (B-factor column in AlphaFold mmCIFs)
            let colorfunc = function(atom) {{
                let bf = atom.b;
                if (bf > 90) return '#0053d6'; // very high
                if (bf > 70) return '#65cbf3'; // confident
                if (bf > 50) return '#ffdb13'; // low
                return '#ff7d45';              // very low
            }};
            
            viewer.setStyle({{}}, {{ cartoon: {{ colorfunc: colorfunc }} }});
            viewer.zoomTo();
            viewer.render();
            viewer.zoom(1.2, 1000);
        }});
    </script>
</body>
</html>
"""
    out_file.write_text(html_content, encoding="utf-8")
    return out_file
