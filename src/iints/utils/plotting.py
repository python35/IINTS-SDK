from __future__ import annotations

from typing import Iterable, List


IINTS_BLUE = "#00798c"
IINTS_RED = "#d1495b"
IINTS_ORANGE = "#f4a261"
IINTS_TEAL = "#2a9d8f"
IINTS_NAVY = "#264653"
IINTS_GOLD = "#e9c46a"


def apply_plot_style(
    dpi: int = 150,
    font_scale: float = 1.1,
    palette: Iterable[str] | None = None,
) -> List[str]:
    """
    Apply IINTS paper-ready plotting defaults (Matplotlib/Seaborn).

    Returns the palette used.
    """
    try:
        import matplotlib as mpl
        import seaborn as sns
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("Plot styling requires matplotlib and seaborn.") from exc

    colors = list(
        palette
        if palette is not None
        else [IINTS_BLUE, IINTS_RED, IINTS_ORANGE, IINTS_TEAL, IINTS_NAVY, IINTS_GOLD]
    )

    sns.set_theme(context="paper", style="whitegrid", palette=colors, font_scale=font_scale)
    mpl.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return colors
