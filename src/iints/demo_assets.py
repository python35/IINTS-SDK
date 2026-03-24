from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def export_live_stage_demo(output_dir: str | Path = '.', *, overwrite: bool = False) -> dict[str, str]:
    """Export the bundled live stage demo script for users running the installed SDK."""
    resolved_output = Path(output_dir).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)

    script_target = resolved_output / '07_live_stage_demo.py'
    notes_target = resolved_output / 'RUN_ME_FIRST.txt'

    if not overwrite and script_target.exists():
        raise FileExistsError(f'Demo script already exists: {script_target}')
    if not overwrite and notes_target.exists():
        raise FileExistsError(f'Instruction file already exists: {notes_target}')

    script_content = files('iints.templates.demos').joinpath('live_stage_demo.py').read_text(encoding='utf-8')
    notes_content = (
        'IINTS LIVE DEMO EXPORT\n'
        '======================\n\n'
        'This folder was exported from the installed IINTS SDK.\n\n'
        '1. Activate the virtual environment that contains IINTS.\n'
        '2. Open 07_live_stage_demo.py and point to:\n'
        '   - PATIENT_CONFIG\n'
        '   - OUTPUT_DIR\n'
        '   - DURATION_MINUTES\n'
        '   - TIME_STEP_MINUTES\n'
        '   - SEED\n'
        '3. Explain that the script visibly calls:\n'
        '   - run_full(...)\n'
        '   - generate_results_poster(...)\n'
        '   - prepare_ai_ready_artifacts(...)\n'
        '4. Run:\n'
        '   python 07_live_stage_demo.py\n'
        '5. Open the generated files under results/booth_demo_live/.\n\n'
        'Tip: if you also cloned the SDK repo, you can run the repo wrapper instead:\n'
        '  ./scripts/run_live_stage_demo.sh\n'
    )

    script_target.write_text(script_content, encoding='utf-8')
    notes_target.write_text(notes_content, encoding='utf-8')

    return {
        'output_dir': str(resolved_output),
        'script_path': str(script_target),
        'notes_path': str(notes_target),
    }
