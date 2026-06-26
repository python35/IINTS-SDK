"""Small local 3D protein-chain renderer for the PySide desktop app.

The widget uses QPainter rather than a browser or an external WebGL viewer so it
keeps working in offline research demos and bundled desktop applications.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import QWidget

from iints_desktop.molecules import BackboneAtom, MoleculeBackbone, load_molecule_backbone


@dataclass(frozen=True)
class _ProjectedAtom:
    atom: BackboneAtom
    point: QPointF
    depth: float
    scale: float


class MolecularChainViewer(QWidget):
    """Mouse-rotatable C-alpha backbone view of one local mmCIF protein structure."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(280, 240)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._structure: MoleculeBackbone | None = None
        self._structure_name = "No structure loaded"
        self._error: str | None = None
        self._yaw_degrees = -35.0
        self._pitch_degrees = 18.0
        self._zoom = 1.0
        self._drag_origin: QPointF | None = None

    @property
    def structure(self) -> MoleculeBackbone | None:
        return self._structure

    @property
    def error(self) -> str | None:
        return self._error

    def set_structure(self, path: Path, *, display_name: str) -> None:
        """Load a packaged mmCIF structure and reset the camera deterministically."""

        try:
            self._structure = load_molecule_backbone(path)
        except ValueError as exc:
            self._structure = None
            self._error = str(exc)
        else:
            self._structure_name = display_name
            self._error = None
            self.reset_camera()
        self.update()

    def reset_camera(self) -> None:
        self._yaw_degrees = -35.0
        self._pitch_degrees = 18.0
        self._zoom = 1.0
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._drag_origin
            self._yaw_degrees += delta.x() * 0.55
            self._pitch_degrees = max(-85.0, min(85.0, self._pitch_degrees + delta.y() * 0.55))
            self._drag_origin = event.position()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_camera()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        self._zoom = max(0.45, min(3.5, self._zoom * (1.15**steps)))
        self.update()
        event.accept()

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#071827"))

        if self._structure is None:
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._error or self._structure_name)
            painter.end()
            return

        projected = self._project_atoms(self._structure)
        self._draw_backbone(painter, projected)
        self._draw_atoms(painter, projected)
        self._draw_overlay(painter, self._structure)
        painter.end()

    def _project_atoms(self, structure: MoleculeBackbone) -> list[_ProjectedAtom]:
        yaw = radians(self._yaw_degrees)
        pitch = radians(self._pitch_degrees)
        yaw_cos, yaw_sin = cos(yaw), sin(yaw)
        pitch_cos, pitch_sin = cos(pitch), sin(pitch)
        center_x, center_y, center_z = structure.center
        scale = min(self.width(), self.height()) * 0.39 / structure.radius * self._zoom
        camera_distance = structure.radius * 3.6
        projected: list[_ProjectedAtom] = []

        for atom in structure.atoms:
            x, y, z = atom.x - center_x, atom.y - center_y, atom.z - center_z
            rotated_x = yaw_cos * x + yaw_sin * z
            yaw_z = -yaw_sin * x + yaw_cos * z
            rotated_y = pitch_cos * y - pitch_sin * yaw_z
            rotated_z = pitch_sin * y + pitch_cos * yaw_z
            perspective = camera_distance / max(camera_distance + rotated_z, structure.radius * 0.55)
            projected.append(
                _ProjectedAtom(
                    atom=atom,
                    point=QPointF(
                        self.width() / 2.0 + rotated_x * scale * perspective,
                        self.height() / 2.0 - rotated_y * scale * perspective,
                    ),
                    depth=rotated_z,
                    scale=perspective,
                )
            )
        return projected

    def _draw_backbone(self, painter: QPainter, atoms: list[_ProjectedAtom]) -> None:
        # Draw segments back-to-front, and never bridge distinct protein chains.
        edges = [
            (left, right)
            for left, right in zip(atoms, atoms[1:])
            if left.atom.chain_id == right.atom.chain_id
            and right.atom.residue_index - left.atom.residue_index <= 1
        ]
        for left, right in sorted(edges, key=lambda edge: (edge[0].depth + edge[1].depth) / 2.0):
            confidence = ((left.atom.confidence or 0.0) + (right.atom.confidence or 0.0)) / 2.0
            tone = self._confidence_color(confidence)
            tone.setAlpha(180)
            painter.setPen(QPen(tone, max(1.3, 3.8 * (left.scale + right.scale) / 2.0)))
            painter.drawLine(left.point, right.point)

    def _draw_atoms(self, painter: QPainter, atoms: list[_ProjectedAtom]) -> None:
        for projected in sorted(atoms, key=lambda item: item.depth):
            color = self._confidence_color(projected.atom.confidence)
            radius = max(2.4, min(8.0, 5.0 * projected.scale))
            painter.setPen(QPen(QColor("#e0f2fe"), 0.65))
            painter.setBrush(color)
            painter.drawEllipse(projected.point, radius, radius)

    def _draw_overlay(self, painter: QPainter, structure: MoleculeBackbone) -> None:
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(16, 28, f"{self._structure_name} - interactive C-alpha backbone")
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(
            16,
            self.height() - 18,
            "Drag to rotate   |   Scroll to zoom   |   Double-click to reset   |   Color: AlphaFold pLDDT",
        )
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(
            self.width() - 190,
            28,
            f"{len(structure.atoms)} residues / {structure.chain_count} chain(s)",
        )

        legend = [
            ("very high", 95.0),
            ("confident", 80.0),
            ("low", 60.0),
            ("very low", 30.0),
        ]
        start_x = 16
        y = 52
        for label, confidence in legend:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._confidence_color(confidence))
            painter.drawEllipse(QPointF(start_x, y), 5, 5)
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(start_x + 10, y + 4, label)
            start_x += 102

    @staticmethod
    def _confidence_color(confidence: float | None) -> QColor:
        # AlphaFold pLDDT palette: blue -> cyan -> yellow -> orange.
        value = confidence if confidence is not None else 0.0
        if value >= 90.0:
            return QColor("#0053d6")
        if value >= 70.0:
            return QColor("#65cbf3")
        if value >= 50.0:
            return QColor("#ffdb13")
        return QColor("#ff7d45")
