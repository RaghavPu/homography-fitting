"""Interactive click-collection UI (local / non-headless only)."""

from __future__ import annotations

import cv2
import numpy as np

OBJ_COLORS_UI = [
    (0, 255, 0),
    (0, 100, 255),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
]


def collect_clicks(frame: np.ndarray) -> list[list[tuple[int, int]]]:
    """Show *frame* in an OpenCV window and collect grouped seed clicks.

    Controls
    --------
    - Left-click : add point to current object
    - N          : finish current object, start a new one
    - Enter/Space: finish all
    - Escape     : cancel (returns empty list)

    Returns a list of groups, e.g. ``[[(x1,y1),(x2,y2)], [(x3,y3)], ...]``.
    """
    groups: list[list[tuple[int, int]]] = [[]]
    display = frame.copy()
    win = "Click regions (N=next object, Enter=done, Esc=cancel)"

    def _current_color():
        return OBJ_COLORS_UI[(len(groups) - 1) % len(OBJ_COLORS_UI)]

    def _redraw_status():
        obj_idx = len(groups)
        n_pts = len(groups[-1])
        label = f"Object {obj_idx}  ({n_pts} pts)  |  N=next  Enter=done"
        cv2.rectangle(display, (0, 0), (frame.shape[1], 30), (30, 30, 30), -1)
        cv2.putText(
            display, label, (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.imshow(win, display)

    def _on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            groups[-1].append((x, y))
            col = _current_color()
            pt_idx = len(groups[-1])
            cv2.drawMarker(display, (x, y), col, cv2.MARKER_STAR, 20, 2)
            cv2.putText(
                display, f"{len(groups)}.{pt_idx}", (x + 12, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA,
            )
            _redraw_status()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, _on_mouse)
    _redraw_status()

    print("[UI] Left-click to add points. N = next object. Enter/Space = done. Esc = cancel.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):  # Enter / Space
            break
        if key == 27:  # Escape
            groups.clear()
            break
        if key in (ord("n"), ord("N")):
            if groups[-1]:
                print(
                    f"  Object {len(groups)} done ({len(groups[-1])} pts). "
                    f"Starting object {len(groups) + 1}…"
                )
                groups.append([])
                _redraw_status()

    cv2.destroyAllWindows()
    return [g for g in groups if g]


def select_polygon(frame: np.ndarray) -> np.ndarray | None:
    """Interactive polygon vertex selection.

    Controls
    --------
    - Left-click  : add a vertex
    - Right-click / Enter / Space : close polygon and accept
    - Escape      : cancel

    Returns an ``(N, 2)`` float32 array, or ``None`` if cancelled.
    """
    pts: list[tuple[int, int]] = []
    display = frame.copy()
    win = "Click corners (right-click/Enter=done, Esc=cancel)"

    def _on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.drawMarker(display, (x, y), (0, 255, 0), cv2.MARKER_CROSS, 16, 2)
            if len(pts) > 1:
                cv2.line(display, pts[-2], pts[-1], (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(
                display, str(len(pts)), (x + 8, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
            cv2.imshow(win, display)
        elif event == cv2.EVENT_RBUTTONDOWN and len(pts) >= 3:
            cv2.line(display, pts[-1], pts[0], (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(win, display)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame.shape[1], 1400), min(frame.shape[0], 900))
    cv2.setMouseCallback(win, _on_mouse)
    cv2.imshow(win, display)

    print("[UI] Left-click to add polygon vertices.")
    print("     Right-click or Enter/Space to finish | Esc to cancel")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (13, 32):
            break
        if key == 27:
            pts.clear()
            break

    cv2.destroyAllWindows()
    if len(pts) < 3:
        return None
    return np.array(pts, dtype=np.float32)
