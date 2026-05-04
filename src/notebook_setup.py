"""Standard notebook setup for the SteamSale project.

Each notebook starts with a small bootstrap (find project root, add to sys.path),
then imports `setup_notebook` from here for the rest of the plumbing:

    import sys
    from pathlib import Path
    PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.notebook_setup import setup_notebook
    conn, paths = setup_notebook()

After that call, plot style is applied, the DB is open, and `paths` carries
the resolved project layout (`paths.root`, `paths.db_path`, `paths.outputs_dir`).
"""
from __future__ import annotations

import atexit
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from .plot_style import apply_style


def _safe_close(conn: sqlite3.Connection) -> None:
    """Close a connection without raising — used as an atexit hook."""
    try:
        conn.close()
    except Exception:
        pass


def find_project_root() -> Path:
    """Return the project root regardless of where the notebook was launched.

    Looks for the directory containing both `src/` and `data/`. Falls back to
    the current working directory if neither parent matches — useful for
    edge cases like nbconvert runs from arbitrary cwds.
    """
    cwd = Path.cwd()
    for candidate in (cwd, cwd.parent, cwd.parent.parent):
        if (candidate / "src").is_dir() and (candidate / "data").is_dir():
            return candidate
    # Last resort — use the package location (src/notebook_setup.py → ../)
    return Path(__file__).resolve().parent.parent


def setup_notebook(
    *,
    require_db: bool = True,
    apply_plot_style: bool = True,
    verbose: bool = True,
) -> tuple[sqlite3.Connection | None, SimpleNamespace]:
    """Standard analysis-notebook setup.

    Returns
    -------
    conn :
        sqlite3 connection to ``data/steam.db``, or ``None`` if
        ``require_db=False`` and the DB doesn't exist.
    paths :
        SimpleNamespace with ``.root``, ``.data_dir``, ``.db_path``,
        ``.outputs_dir``, ``.notebooks_dir``.

    Parameters
    ----------
    require_db :
        If True (default), assert the DB exists and connect. If False, the
        DB may not yet exist — useful for the data-collection notebook which
        creates it.
    apply_plot_style :
        Call ``apply_style()`` from ``src.plot_style``. Default True.
    verbose :
        Print connection summary on success.
    """
    if apply_plot_style:
        apply_style()

    root = find_project_root()
    paths = SimpleNamespace(
        root=root,
        data_dir=root / "data",
        db_path=root / "data" / "steam.db",
        outputs_dir=root / "outputs",
        notebooks_dir=root / "notebooks",
    )
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)

    conn: sqlite3.Connection | None = None
    if require_db:
        assert paths.db_path.exists(), (
            f"DB not found at: {paths.db_path}\n"
            "Run 01_data_collection then 02_data_cleaning first."
        )
        # WAL mode is the default on this DB and supports concurrent readers
        # cleanly. Don't force journal_mode here — switching modes requires
        # an exclusive lock and will deadlock if any other notebook/DB Browser
        # has the file open. The .shm/.wal sidecar files clean up themselves
        # when the last connection closes.
        conn = sqlite3.connect(paths.db_path)

        # Belt + suspenders: when the kernel shuts down (or the interpreter
        # exits for any reason), close the connection. sqlite3.Connection.close()
        # is idempotent, so it's safe even if the user already called it
        # explicitly. Without this, a kernel that's killed mid-session leaves
        # the .shm/.wal sidecar files locked until the OS reaps the process.
        atexit.register(_safe_close, conn)

        if verbose:
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
                )
            ]
            print(f"Connected to: {paths.db_path}")
            print(f"Tables ({len(tables)}): {tables}")
    elif verbose:
        print(f"Project root: {root}")
        print(f"DB path:      {paths.db_path}  (exists={paths.db_path.exists()})")

    return conn, paths
