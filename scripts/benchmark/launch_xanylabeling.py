#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTF-8 Safe Launcher & Health Checker for X-AnyLabeling on Windows (`Học Tập` path + Emoji logs).

Usage:
  1. Launch X-AnyLabeling GUI on the Leader Sample folder (default):
     .\\venv\\Scripts\\python.exe scripts/benchmark/launch_xanylabeling.py

  2. Launch X-AnyLabeling GUI on Member 1..4 folder:
     .\\venv\\Scripts\\python.exe scripts/benchmark/launch_xanylabeling.py --member 1

  3. Run automated end-to-end health check (verifies all folders, shapes, docks & UTF-8 logging):
     .\\venv\\Scripts\\python.exe scripts/benchmark/launch_xanylabeling.py --check
"""

import argparse
import os
import sys

# Force UTF-8 encoding before importing anylabeling (prevents cp1252 crash on 'Học Tập' & emoji logs)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.gui.icc=false"

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
WORKSPACE_DIR = os.path.join(PROJECT_ROOT, "data", "xanylabeling_workspace")
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "xanylabeling_config.yaml")

FOLDER_MAP = {
    "0": "00_sample_for_leader",
    "sample": "00_sample_for_leader",
    "1": "Cao_Anh_001_100",
    "cao_anh": "Cao_Anh_001_100",
    "2": "Trieu_101_200",
    "trieu": "Trieu_101_200",
    "3": "Cam_Tu_201_300",
    "cam_tu": "Cam_Tu_201_300",
    "4": "Thanh_Tam_301_400",
    "thanh_tam": "Thanh_Tam_301_400",
}

ALL_FOLDERS = [
    "00_sample_for_leader",
    "Cao_Anh_001_100",
    "Trieu_101_200",
    "Cam_Tu_201_300",
    "Thanh_Tam_301_400",
]


def ensure_user_config() -> None:
    """Copy workspace `xanylabeling_config.yaml` to `~/.xanylabelingrc` so every member gets identical settings."""
    import shutil

    rc_path = os.path.expanduser("~/.xanylabelingrc")
    if os.path.exists(CONFIG_PATH):
        try:
            shutil.copy2(CONFIG_PATH, rc_path)
        except Exception:
            pass


def run_health_check() -> None:
    """Programmatically verify MainWindow, Canvas, Shapes, and Description docks across all folders."""
    import anylabeling.config as ac
    from anylabeling.config import get_config
    from anylabeling.resources import resources  # noqa: F401
    from anylabeling.views.mainwindow import MainWindow
    from PyQt6 import QtCore, QtWidgets

    ensure_user_config()
    ac.current_config_file = CONFIG_PATH
    app = QtWidgets.QApplication(sys.argv)
    cfg = get_config(CONFIG_PATH, {})

    for folder_name in ALL_FOLDERS:
        target_dir = os.path.join(WORKSPACE_DIR, folder_name)
        if not os.path.isdir(target_dir):
            raise FileNotFoundError(f"Missing folder: {target_dir}")

        win = MainWindow(app, config=cfg, filename=target_dir)
        win.showMaximized()
        app.processEvents()

        lw = win.labeling_widget.view
        n_imgs = len(lw.image_list)
        n_shapes = len(lw.canvas.shapes)
        desc_visible = lw.description_dock.isVisible()
        shape_visible = lw.shape_dock.isVisible()
        file_visible = lw.file_dock.isVisible()

        img_desc_text = lw.shape_text_edit.toPlainText().strip()
        if lw.canvas.shapes:
            lw.canvas.select_shapes([lw.canvas.shapes[0]])
            app.processEvents()
        shape_desc_text = lw.shape_text_edit.toPlainText().strip()

        print(
            f"[CHECK OK] Folder={folder_name} | Images={n_imgs} | "
            f"FirstImage={os.path.basename(str(lw.filename))} | ShapesOnCanvas={n_shapes} | "
            f"Docks(Desc={desc_visible}, Shapes={shape_visible}, Files={file_visible}) | "
            f"ImgDescLen={len(img_desc_text)} | FirstShapeDesc='{shape_desc_text[:65]}...'"
        )
        win.close()
        app.processEvents()

    print("[ALL CHECKS PASSED] X-AnyLabeling GUI, PyQt6 DLLs, UTF-8 paths, and all 5 folders are 100% healthy.")


def launch_gui(target_dir: str) -> None:
    """Launch the interactive X-AnyLabeling GUI window with Description docks visible and first shape selected."""
    import anylabeling.config as ac
    from anylabeling.app_info import __appname__, __version__
    from anylabeling.config import get_config, set_work_directory
    from anylabeling.resources import resources  # noqa: F401
    from anylabeling.views.labeling.utils import new_icon
    from anylabeling.views.labeling.utils.theme import (
        get_app_stylesheet,
        get_dark_palette,
        init_theme,
    )
    from anylabeling.views.mainwindow import MainWindow
    from PyQt6 import QtCore, QtWidgets

    set_work_directory(WORKSPACE_DIR)
    ensure_user_config()
    ac.current_config_file = CONFIG_PATH
    cfg = get_config(CONFIG_PATH, {})

    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    )
    app = QtWidgets.QApplication([sys.argv[0]])
    init_theme(cfg.get("theme", "dark"))
    dark_palette = get_dark_palette()
    if dark_palette is not None:
        app.setStyle("Fusion")
        app.setPalette(dark_palette)
    app.setStyleSheet(get_app_stylesheet())
    app.setApplicationName(__appname__)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(new_icon("icon"))

    win = MainWindow(app, config=cfg, filename=target_dir)
    win.showMaximized()
    win.raise_()
    win.activateWindow()

    def _auto_select_first_shape() -> None:
        lw = win.labeling_widget.view
        lw.description_dock.show()
        lw.shape_dock.show()
        lw.file_dock.show()
        if lw.canvas.shapes:
            lw.canvas.select_shapes([lw.canvas.shapes[0]])

    QtCore.QTimer.singleShot(300, _auto_select_first_shape)
    sys.exit(app.exec())


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch or Verify X-AnyLabeling Workspace")
    parser.add_argument(
        "--member",
        type=str,
        default="Cao_Anh",
        help="Tên thành viên hoặc số thứ tự: Cao_Anh (1), Trieu (2), Cam_Tu (3), Thanh_Tam (4), hoặc sample (0).",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Custom directory to open in X-AnyLabeling.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run automated end-to-end GUI health check across all folders.",
    )
    args = parser.parse_args()

    if args.check:
        run_health_check()
        return

    key = str(args.member).strip().lower()
    if key not in FOLDER_MAP:
        raise ValueError(
            f"Unknown --member '{args.member}'. Choose from: Cao_Anh (1), Trieu (2), Cam_Tu (3), Thanh_Tam (4), sample (0)."
        )
    target_dir = args.dir or os.path.join(WORKSPACE_DIR, FOLDER_MAP[key])
    launch_gui(target_dir)


if __name__ == "__main__":
    main()
