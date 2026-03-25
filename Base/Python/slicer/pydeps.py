"""Utilities for managing Python package dependencies in 3D Slicer.

This module provides functions for checking, installing, and managing Python
packages in the Slicer environment. It is the canonical home for all pip-related
functionality.

**Key functions:**

- :func:`load_requirements` / :func:`load_pyproject_dependencies` — parse dependency files
- :func:`pip_check` — check if requirements are satisfied (pure Python, no subprocess)
- :func:`pip_ensure` — high-level: check, prompt, install, restart detection
- :func:`pip_install` / :func:`pip_uninstall` — install/uninstall packages

For backward compatibility, :func:`pip_install` and :func:`pip_uninstall` are also
available as ``slicer.util.pip_install`` and ``slicer.util.pip_uninstall``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import os
import shlex
import shutil
import sys
import tomllib
from pathlib import Path
from subprocess import CalledProcessError
from typing import TYPE_CHECKING

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from slicer.util import launchConsoleProcess, logProcessOutput

if TYPE_CHECKING:
    from collections.abc import Callable

    import qt


def _generate_environment_constraints() -> str:
    """Generate a temporary constraints file pinning all currently installed packages.

    Snapshots every installed package and its version, writing ``name==version``
    lines to a temporary file. This file can be passed to pip as ``-c`` to prevent
    installation from upgrading or downgrading any package already present in
    Slicer's Python environment.

    Package names are canonicalized and deduplicated. If multiple dist-info
    directories exist for the same package, the last one encountered wins.

    The temporary file is registered for automatic cleanup on interpreter exit
    via :func:`atexit.register`.

    :returns: Path to the temporary constraints file.
    """
    import atexit
    import tempfile

    pins: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = canonicalize_name(dist.metadata["Name"])
        version = dist.metadata["Version"]
        pins[name] = version

    lines = [f"{name}=={version}" for name, version in sorted(pins.items())]

    fd, path = tempfile.mkstemp(suffix="-slicer-constraints.txt", prefix="pydeps-")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")

    atexit.register(lambda: os.unlink(path) if os.path.exists(path) else None)
    return path


def _executePythonModule(
    module: str,
    args: list[str],
    blocking: bool = True,
    logCallback: Callable[[str], None] | None = None,
    completedCallback: Callable[[int], None] | None = None,
) -> None:
    """Execute a Python module as a script in Slicer's Python environment.

    Internally python -m is called with the module name and additional arguments.

    :param module: Python module name to execute (e.g., "pip")
    :param args: command-line arguments to pass to the module
    :param blocking: If True (default), block until completion. If False, return immediately.
    :param logCallback: When blocking=False, called with each line of output.
    :param completedCallback: When blocking=False, called when process completes.
    :raises RuntimeError: if PythonSlicer executable not found
    :raises CalledProcessError: in blocking mode, if process fails
    """
    # Determine pythonSlicerExecutablePath
    try:
        from slicer import app  # noqa: F401

        # If we get to this line then import from "app" is succeeded,
        # which means that we run this function from Slicer Python interpreter.
        # PythonSlicer is added to PATH environment variable in Slicer
        # therefore shutil.which will be able to find it.
        pythonSlicerExecutablePath = shutil.which("PythonSlicer")
        if not pythonSlicerExecutablePath:
            raise RuntimeError("PythonSlicer executable not found")
    except ImportError:
        # Running from console
        pythonSlicerExecutablePath = os.path.dirname(sys.executable) + "/PythonSlicer"
        if os.name == "nt":
            pythonSlicerExecutablePath += ".exe"

    commandLine = [pythonSlicerExecutablePath, "-m", module, *args]

    if blocking:
        proc = launchConsoleProcess(commandLine, useStartupEnvironment=False)
        logProcessOutput(proc, logCallback=logCallback)
    else:
        launchConsoleProcess(
            commandLine,
            useStartupEnvironment=False,
            blocking=False,
            logCallback=logCallback,
            completedCallback=completedCallback,
        )


def load_requirements(path: str | Path) -> list[Requirement]:
    """Load requirements from a requirements.txt file.

    Parses a requirements file and returns a list of Requirement objects
    that can be used with :func:`pip_check` and :func:`pip_ensure`.

    :param path: Path to the requirements.txt file. Can be a string or Path object.

    :returns: List of :class:`packaging.requirements.Requirement` objects.

    Lines starting with ``#`` (comments), empty lines, and lines starting with
    ``-`` (pip options like ``-r``, ``-c``, ``--index-url``) are skipped.

    Example:

    .. code-block:: python

      from pathlib import Path
      reqs = slicer.pydeps.load_requirements(Path(__file__).parent / "requirements.txt")
      slicer.pydeps.pip_ensure(reqs, requester="MyExtension")

    """
    reqs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            # Skip comments, empty lines, and pip options (-r, -c, --index-url, etc.)
            if line and not line.startswith("#") and not line.startswith("-"):
                reqs.append(Requirement(line))
    return reqs


def load_pyproject_dependencies(path: str | Path) -> list[Requirement]:
    """Load dependencies from a pyproject.toml file.

    Reads the ``[project.dependencies]`` list (`PEP 621
    <https://peps.python.org/pep-0621/>`_) and returns
    :class:`~packaging.requirements.Requirement` objects that can be used
    with :func:`pip_check` and :func:`pip_ensure`.

    Only the ``dependencies`` list inside the ``[project]`` table is read.
    Other fields in the table (``name``, ``version``, etc.) are not read or
    validated.

    :param path: Path to the pyproject.toml file. Can be a string or Path object.

    :returns: List of :class:`packaging.requirements.Requirement` objects.
        Returns an empty list if ``[project]`` exists but has no
        ``dependencies`` key.

    :raises KeyError: If the file has no ``[project]`` table.

    Example pyproject.toml::

      [project]
      dependencies = [
          "numpy>=1.20",
          "scikit-image>=0.20",
      ]

    Example:

    .. code-block:: python

      from pathlib import Path
      reqs = slicer.pydeps.load_pyproject_dependencies(
          Path(__file__).parent / "pyproject.toml"
      )
      slicer.pydeps.pip_ensure(reqs, requester="MyExtension")

    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    if "project" not in data:
        msg = f"pyproject.toml has no [project] table: {path}"
        raise KeyError(msg)

    return [Requirement(dep) for dep in data["project"].get("dependencies", [])]


def pip_check(req: Requirement | list[Requirement], _seen: set[tuple[str, frozenset[str]]] | None = None) -> bool:
    """Check if requirement(s) are satisfied.

    For requirements with extras like ``package[extra1,extra2]>=1.0``, this:

    1. Checks if the base package is installed at an acceptable version
    2. Finds which dependencies are activated by the requested extras
    3. Recursively verifies those dependencies are satisfied

    Markers (e.g., ``; sys_platform == 'win32'``) are evaluated - if a marker
    doesn't apply to the current environment, the requirement is considered
    satisfied (since it doesn't need to be installed).

    :param req: Either a :class:`packaging.requirements.Requirement` object
        or a list of Requirement objects.
    :param _seen: Internal parameter for tracking circular dependencies.
        Do not pass this parameter.

    :returns: True if all requirements are satisfied, False otherwise.

    Example:

    .. code-block:: python

      from packaging.requirements import Requirement

      if slicer.pydeps.pip_check(Requirement("numpy>=1.20")):
          print("numpy is satisfied")

    """
    from importlib.metadata import PackageNotFoundError, requires, version

    if _seen is None:
        _seen = set()

    # Handle list of requirements, sharing _seen across all of them
    if isinstance(req, list):
        return all(pip_check(r, _seen) for r in req)

    # Check if requirement's marker applies to current environment
    # If not, consider it satisfied (doesn't need to be installed here)
    if req.marker is not None:
        env = default_environment()
        if not req.marker.evaluate(env):
            return True

    # Avoid rechecking the same requirement
    key = (req.name.lower(), frozenset(req.extras))
    if key in _seen:
        return True
    _seen.add(key)

    # Check if base package is installed at acceptable version
    try:
        installed = version(req.name)
    except PackageNotFoundError:
        return False
    if installed not in req.specifier:
        return False

    # If no extras then we are done
    if not req.extras:
        return True

    # Find dependencies activated by the requested extras
    dep_strings = requires(req.name) or []
    env = default_environment()
    activated = []

    for dep_str in dep_strings:
        dep = Requirement(dep_str)
        if dep.marker is None:
            continue

        # Check if any requested extra activates this dependency
        for extra in req.extras:
            if dep.marker.evaluate({**env, "extra": extra}):
                # Strip marker before recursive check - we've already determined it applies
                dep_str_no_marker = str(dep).split(";")[0].strip()
                activated.append(Requirement(dep_str_no_marker))
                break  # Don't check other extras for same dep

    # Recursively verify all activated dependencies
    return all(pip_check(dep, _seen) for dep in activated)


# Module-level flag to track whether a non-blocking pip install is in progress.
# Used to prevent concurrent non-blocking pip operations which could corrupt
# the Python environment.
_pip_install_in_progress: bool = False


def isPipInstallInProgress() -> bool:
    """Check if a non-blocking pip install is currently in progress.

    Use this to check before starting an operation that might conflict with
    an ongoing pip installation.

    :returns: True if a non-blocking pip install is in progress, False otherwise.

    Example:

    .. code-block:: python

      if slicer.pydeps.isPipInstallInProgress():
          slicer.util.warningDisplay(
              "Package installation is in progress. Please wait."
          )
      else:
          slicer.pydeps.pip_install("scipy", blocking=False)

    """
    return _pip_install_in_progress


def _get_installed_versions() -> dict[str, str]:
    """Return ``{canonical_distribution_name: version}`` for all installed packages.

    Calls :func:`importlib.invalidate_caches` first so that metadata changes made by
    a pip subprocess are visible to this process.
    """
    importlib.invalidate_caches()
    return {
        canonicalize_name(dist.metadata["Name"]): dist.metadata["Version"]
        for dist in importlib.metadata.distributions()
    }


def _find_updated_imported_packages(
    before_versions: dict[str, str],
    after_versions: dict[str, str],
) -> list[tuple[str, str | None, str]]:
    """Find packages that were updated AND had modules already imported.

    Compares *before_versions* and *after_versions* (both returned by
    :func:`_get_installed_versions`) and checks whether any changed distribution
    has top-level import names present in :data:`sys.modules`.

    :param before_versions: ``{canonical_dist_name: version}`` before installation.
    :param after_versions: ``{canonical_dist_name: version}`` after installation.
    :returns: List of ``(dist_name, old_version, new_version)`` tuples for packages
        that have import names present in ``sys.modules`` and either changed version
        or were freshly installed (``old_version`` is ``None`` in that case).
    """
    # Reverse mapping: canonical dist name → set of top-level import names
    dist_to_imports: dict[str, set[str]] = {}
    for import_name, dist_names in importlib.metadata.packages_distributions().items():
        for dn in dist_names:
            dist_to_imports.setdefault(canonicalize_name(dn), set()).add(import_name)

    result = []
    for name, old_ver in before_versions.items():
        new_ver = after_versions.get(name)
        if new_ver is None or old_ver == new_ver:
            continue
        import_names = dist_to_imports.get(name, set())
        if any(imp in sys.modules for imp in import_names):
            result.append((name, old_ver, new_ver))

    # Also check packages that were freshly installed (not in before_versions).
    # If their import names are already in sys.modules, that means they were
    # previously installed and imported, then uninstalled, then reinstalled.
    # The old module objects are still in memory.
    for name, new_ver in after_versions.items():
        if name in before_versions:
            continue  # Already handled above
        import_names = dist_to_imports.get(name, set())
        if any(imp in sys.modules for imp in import_names):
            result.append((name, None, new_ver))

    return result


def pip_ensure(
    requirements: list[Requirement],
    constraints: str | Path | None = None,
    protect_environment: bool = True,
    prompt_install: bool = True,
    prompt_restart: bool = True,
    requester: str | None = None,
    skip_in_testing: bool = True,
    show_progress: bool = True,
) -> None:
    """Ensure requirements are satisfied, installing if needed.

    Call at the point where dependencies are actually needed (e.g., in an
    ``onApplyButton`` method). This function checks which requirements are
    missing and installs them, optionally showing a confirmation dialog first.

    After installation, if any updated packages were already imported in the
    current session, a restart prompt is shown (the old versions remain loaded
    in memory and may not work correctly until Slicer is restarted).

    :param requirements: List of :class:`packaging.requirements.Requirement` objects,
        typically obtained from :func:`load_requirements` or
        :func:`load_pyproject_dependencies`.
    :param constraints: Path to a constraints file (string or Path object), or None.
        When provided, passed to pip as ``-c constraints.txt`` during installation.
        Constraints files use the same format as requirements files but only constrain
        versions without triggering installation. Useful for ensuring compatible
        versions across multiple extensions.
    :param protect_environment: If True (default), auto-generate constraints that pin
        all packages currently installed in Slicer's Python environment. This prevents
        pip from upgrading or downgrading any existing package. If a requirement
        genuinely conflicts with an installed version, pip will raise an error rather
        than silently modifying the environment.
    :param prompt_install: If True (default), show confirmation dialog before installing.
    :param prompt_restart: If True (default), check whether any updated packages were
        already imported and, if so, show a dialog recommending a restart. The user
        can choose to restart immediately or continue without restarting.
    :param requester: Name shown in dialog to identify who is requesting the packages
        (e.g., "TotalSegmentator", "MyFilter", "MyExtension").
    :param skip_in_testing: If True (default), skip installation when Slicer is running
        in testing mode (``slicer.app.testingEnabled()``). This prevents tests from
        modifying the Python environment. Set to False if your test explicitly
        needs to verify installation behavior.
    :param show_progress: If True (default), show progress dialog during installation
        with status updates and collapsible log details. If False, show only
        a busy cursor during installation.

    :raises RuntimeError: If user declines installation.
    :raises subprocess.CalledProcessError: If installation fails.

    Example:

    .. code-block:: python

      from typing import TYPE_CHECKING

      if TYPE_CHECKING:
          import skimage

      class MyFilterWidget(ScriptedLoadableModuleWidget):

          def onApplyButton(self):
              reqs = slicer.pydeps.load_requirements(
                  self.resourcePath("requirements.txt")
              )
              slicer.pydeps.pip_ensure(reqs, requester="MyFilter")
              import skimage

              # Now safe to use skimage
              filtered = skimage.filters.gaussian(array, sigma=2.0)

    For more examples, see
    :doc:`/developer_guide/script_repository` (Python package management section).

    """
    missing = [req for req in requirements if not pip_check(req)]

    if not missing:
        return None  # All satisfied

    # Check if we're in full Slicer or PythonSlicer
    if not _isSlicerAppAvailable():
        # Running in PythonSlicer - just do simple install (prompt not available)
        pip_install(
            [str(req) for req in missing],
            constraints=constraints,
            protect_environment=protect_environment,
        )
        return

    import slicer

    # Skip installation in testing mode to avoid modifying the environment
    if skip_in_testing and slicer.app.testingEnabled():
        missing_str = ", ".join(str(req) for req in missing)
        logging.info(f"Testing mode is enabled: skipping pip_ensure for [{missing_str}]")
        return None

    if prompt_install:
        package_list = "\n".join(f"• {req}" for req in missing)
        title = f"{requester} - Install Python Packages" if requester else "Install Python Packages"
        count = len(missing)
        message = (
            f"{count} Python package{'s' if count != 1 else ''} "
            f"need{'s' if count == 1 else ''} to be installed.\n\n"
            f"This will modify Slicer's Python environment. Continue?"
        )
        if not slicer.util.confirmOkCancelDisplay(message, title, detailedText=package_list):
            raise RuntimeError("User declined package installation")

    # Snapshot installed versions before installation
    before_versions = _get_installed_versions() if prompt_restart else None

    # Install missing packages with optional progress display
    pip_install(
        [str(req) for req in missing],
        constraints=constraints,
        protect_environment=protect_environment,
        blocking=True,
        show_progress=show_progress,
        requester=requester,
    )

    if not prompt_restart:
        return

    # Check if any updated packages were already imported
    after_versions = _get_installed_versions()
    updated_imported = _find_updated_imported_packages(before_versions, after_versions)

    if updated_imported:
        detail_lines = [
            f"• {name}: {old_ver} → {new_ver}" if old_ver else f"• {name}: (reinstalled) → {new_ver}"
            for name, old_ver, new_ver in updated_imported
        ]
        detail_text = "\n".join(detail_lines)
        logging.info(
            f"Updated packages already imported (restart recommended): {detail_text}",
        )

        # confirmYesNoDisplay auto-returns True in testing mode, which would
        # trigger an unwanted restart. Guard with testingEnabled() check and
        # rely on the log message above for test verification.
        if not slicer.app.testingEnabled():
            count = len(updated_imported)
            title = (
                f"{requester} - Restart Recommended"
                if requester
                else "Restart Recommended"
            )
            message = (
                f"{count} updated package{'s' if count != 1 else ''} "
                f"{'were' if count != 1 else 'was'} already loaded in memory "
                f"and may not work correctly until Slicer is restarted.\n\n"
                f"Would you like to restart now?"
            )
            if slicer.util.confirmYesNoDisplay(
                message, title, detailedText=detail_text,
            ):
                slicer.util.restart()

    return


def _isSlicerAppAvailable() -> bool:
    """Check if slicer.app is available (running in full Slicer, not PythonSlicer).

    :returns: True if slicer.app is available, False if running in PythonSlicer console.
    """
    try:
        from slicer import app
        return app is not None
    except ImportError:
        return False


def pip_install(
    requirements: str | list[str],
    constraints: str | Path | None = None,
    no_deps_requirements: str | list[str] | None = None,
    protect_environment: bool = True,
    blocking: bool = True,
    show_progress: bool = True,
    requester: str | None = None,
    parent: qt.QWidget | None = None,
    logCallback: Callable[[str], None] | None = None,
    completedCallback: Callable[[int], None] | None = None,
) -> None:
    """Install python packages.

    Currently, the method simply calls ``python -m pip install`` but in the future further checks, optimizations,
    user confirmation may be implemented, therefore it is recommended to use this method call instead of calling
    pip install directly.

    :param requirements: requirement specifier in the same format as used by pip (https://docs.python.org/3/installing/index.html).
      It can be either a single string or a list of command-line arguments. In general, passing all arguments as a single string is
      the simplest. The only case when using a list may be easier is when there are arguments that may contain spaces, because
      each list item is automatically quoted (it is not necessary to put quotes around each string argument that may contain spaces).
    :param constraints: Path to a constraints file (string or Path object), or None.
        When provided, passed to pip as ``-c constraints.txt``. Constraints files use the
        same format as requirements files but only constrain versions without triggering
        installation. Useful for ensuring compatible versions across multiple extensions.
    :param no_deps_requirements: Packages to install with ``--no-deps`` flag, installed before
        ``requirements``. Use this when a package has overly strict dependency requirements
        that conflict with other packages. Can be a string or list, same format as ``requirements``.
        When provided, installation happens in two steps: first ``no_deps_requirements`` are
        installed with ``--no-deps``, then ``requirements`` are installed normally.
    :param protect_environment: If True (default), auto-generate constraints that pin
        all packages currently installed in Slicer's Python environment. This prevents
        pip from upgrading or downgrading any existing package as a side effect of
        installing new packages. Set to False only when you explicitly intend to
        upgrade an existing package (e.g., ``pip_install("--upgrade numpy", protect_environment=False)``).
    :param blocking: If True (default), block until installation completes and raise
        CalledProcessError on failure. If False, return immediately and use callbacks.
        Note: When running in PythonSlicer (without the full application), blocking mode
        is always used regardless of this setting.
    :param show_progress: If True (default), show visual feedback during installation. When
        blocking=True, shows a modal progress dialog with status updates and collapsible log
        details. When blocking=False, shows pip output lines in the status bar as installation
        progresses. Note: When running in PythonSlicer or in testing mode, progress display
        is skipped and simple blocking mode is used.
    :param requester: Name shown in dialog/status bar to identify who is requesting the packages
        (e.g., "MyExtension"). Only used when show_progress=True.
    :param parent: Parent widget for the progress dialog. Only used when show_progress=True
        and blocking=True.
    :param logCallback: When blocking=False, called with each line of pip output.
        If None, output is printed to Python console as usual.
        Signature: ``logCallback(line: str) -> None``
    :param completedCallback: When blocking=False, called when pip completes.
        Receives return code (0 = success). Recommended when blocking=False to know
        when installation finished.
        Signature: ``completedCallback(returnCode: int) -> None``

    :raises subprocess.CalledProcessError: In blocking mode, if pip installation fails.
        When show_progress=True, an error dialog with the full log is shown before raising.

    .. warning::

        When using ``blocking=False``, the user can interact with the application
        while installation is in progress. Consider disabling relevant UI elements
        to prevent conflicts.

    Example:

    .. code-block:: python

      pip_install("pandas scipy scikit-learn")

    For more examples (constraints, non-blocking mode, no_deps_requirements),
    see :doc:`/developer_guide/script_repository` (Python package management section).

    """
    # Check if we're running in full Slicer or PythonSlicer
    if not _isSlicerAppAvailable():
        # Running in PythonSlicer - use simple blocking mode
        _pip_install_simple(requirements, constraints, no_deps_requirements,
                            protect_environment)
        return

    import slicer

    # In testing mode, skip UI and use simple blocking install
    if slicer.app.testingEnabled():
        logging.info("Testing mode is enabled: skipping progress UI for pip_install")
        _pip_install_simple(requirements, constraints, no_deps_requirements,
                            protect_environment)
        return

    # Check for concurrent non-blocking installs
    if not blocking and _pip_install_in_progress:
        raise RuntimeError(
            "A non-blocking pip_install is already in progress. "
            "Wait for it to complete or use blocking=True.",
        )

    # Determine which mode to use based on show_progress and blocking
    if show_progress and blocking:
        # Modal progress dialog (blocking from user's perspective, but UI responsive)
        _pip_install_with_dialog(
            requirements, constraints, no_deps_requirements, requester, parent,
            protect_environment,
        )
    elif show_progress and not blocking:
        # Status bar progress (non-blocking)
        _pip_install_with_statusbar(
            requirements, constraints, no_deps_requirements, requester,
            logCallback, completedCallback, protect_environment,
        )
    elif not show_progress and blocking:
        # Busy cursor only
        _pip_install_with_busy_cursor(requirements, constraints, no_deps_requirements,
                                      protect_environment)
    else:
        # Non-blocking, no progress (existing behavior)
        _pip_install_nonblocking(
            requirements, constraints, no_deps_requirements,
            logCallback, completedCallback, protect_environment,
        )


def _pip_install_simple(
    requirements: str | list[str],
    constraints: str | Path | None = None,
    no_deps_requirements: str | list[str] | None = None,
    protect_environment: bool = True,
) -> None:
    """Simple blocking pip install without any UI (for PythonSlicer and testing mode).

    :raises subprocess.CalledProcessError: If pip installation fails.
    """
    # Handle no_deps_requirements first
    if no_deps_requirements is not None:
        args = _build_pip_args(no_deps_requirements, constraints, no_deps=True,
                               protect_environment=protect_environment)
        _executePythonModule("pip", args, blocking=True)

    # Then install regular requirements
    args = _build_pip_args(requirements, constraints, no_deps=False,
                           protect_environment=protect_environment)
    _executePythonModule("pip", args, blocking=True)


def _pip_install_with_busy_cursor(
    requirements: str | list[str],
    constraints: str | Path | None = None,
    no_deps_requirements: str | list[str] | None = None,
    protect_environment: bool = True,
) -> None:
    """Blocking pip install with busy cursor.

    :raises subprocess.CalledProcessError: If pip installation fails.
    """
    import qt

    qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)
    try:
        _pip_install_simple(requirements, constraints, no_deps_requirements,
                            protect_environment)
    finally:
        qt.QApplication.restoreOverrideCursor()


def _pip_install_with_dialog(
    requirements: str | list[str],
    constraints: str | Path | None = None,
    no_deps_requirements: str | list[str] | None = None,
    requester: str | None = None,
    parent: qt.QWidget | None = None,
    protect_environment: bool = True,
) -> None:
    """Blocking pip install with modal progress dialog.

    :raises subprocess.CalledProcessError: If pip installation fails.
    """
    import threading

    import qt
    import slicer

    # Create the progress dialog
    dialog = _PipProgressDialog(requester=requester, parent=parent)
    dialog.show()
    slicer.app.processEvents()  # Ensure dialog is displayed

    completed = threading.Event()
    result = {"returnCode": None, "log": ""}

    def onLog(line):
        dialog.appendLog(line)
        print(line)  # Still log to console

    def onComplete(returnCode):
        result["returnCode"] = returnCode
        result["log"] = dialog.getFullLog()
        completed.set()

    # Run the installation (uses internal non-blocking path)
    _pip_install_nonblocking(
        requirements, constraints, no_deps_requirements,
        logCallback=onLog, completedCallback=onComplete,
        protect_environment=protect_environment,
    )

    # Wait for completion while keeping UI responsive
    while not completed.is_set():
        slicer.app.processEvents()
        qt.QThread.msleep(10)  # Reduce CPU usage

    dialog.close()

    if result["returnCode"] != 0:
        slicer.util.errorDisplay(
            "Package installation failed.",
            windowTitle=f"{requester} - Installation Failed" if requester else "Installation Failed",
            detailedText=result["log"],
        )
        raise CalledProcessError(result["returnCode"], "pip install")


def _pip_install_with_statusbar(
    requirements: str | list[str],
    constraints: str | Path | None = None,
    no_deps_requirements: str | list[str] | None = None,
    requester: str | None = None,
    logCallback: Callable[[str], None] | None = None,
    completedCallback: Callable[[int], None] | None = None,
    protect_environment: bool = True,
) -> None:
    """Non-blocking pip install with status bar messages.

    Shows pip output lines in the status bar as installation progresses.
    """
    import qt
    import slicer

    prefix = f"{requester}: " if requester else ""

    # Set busy cursor
    qt.QApplication.setOverrideCursor(qt.Qt.BusyCursor)

    def wrappedLogCallback(line):
        # Show pip output in status bar (no timeout so it stays until next update)
        slicer.util.showStatusMessage(f"{prefix}{line}", 0)
        if logCallback:
            logCallback(line)
        else:
            print(line)

    def wrappedCompletedCallback(returnCode):
        # Clean up cursor and clear status message
        qt.QApplication.restoreOverrideCursor()
        slicer.util.showStatusMessage("", 0)  # Clear status bar

        if completedCallback:
            completedCallback(returnCode)
        elif returnCode != 0:
            logging.error(f"pip install failed with return code {returnCode}")

    try:
        _pip_install_nonblocking(
            requirements, constraints, no_deps_requirements,
            logCallback=wrappedLogCallback, completedCallback=wrappedCompletedCallback,
            protect_environment=protect_environment,
        )
    except Exception:
        # Not `finally` - on success, cursor stays busy until wrappedCompletedCallback fires
        qt.QApplication.restoreOverrideCursor()
        slicer.util.showStatusMessage("", 0)
        raise


def _pip_install_nonblocking(
    requirements: str | list[str],
    constraints: str | Path | None = None,
    no_deps_requirements: str | list[str] | None = None,
    logCallback: Callable[[str], None] | None = None,
    completedCallback: Callable[[int], None] | None = None,
    protect_environment: bool = True,
) -> None:
    """Non-blocking pip install.

    Handles no_deps_requirements by chaining the two pip calls.
    Sets the _pip_install_in_progress flag to prevent concurrent operations.
    """
    global _pip_install_in_progress
    _pip_install_in_progress = True

    # Wrapper to clear the flag when the operation completes
    def wrappedCompletedCallback(returnCode):
        global _pip_install_in_progress
        _pip_install_in_progress = False
        if completedCallback:
            completedCallback(returnCode)

    try:
        if no_deps_requirements is None:
            # Simple case - single install
            args = _build_pip_args(requirements, constraints, no_deps=False,
                                   protect_environment=protect_environment)
            _executePythonModule("pip", args, blocking=False,
                                 logCallback=logCallback, completedCallback=wrappedCompletedCallback)
            return

        # Two-step installation: first no_deps, then regular
        def onNoDepsComplete(returnCode):
            if returnCode != 0:
                # First step failed - abort, clear flag, and notify caller
                global _pip_install_in_progress
                _pip_install_in_progress = False
                if completedCallback:
                    completedCallback(returnCode)
                return
            # Success - proceed to regular install (flag stays set until final completion)
            args = _build_pip_args(requirements, constraints, no_deps=False,
                                   protect_environment=protect_environment)
            _executePythonModule("pip", args, blocking=False,
                                 logCallback=logCallback, completedCallback=wrappedCompletedCallback)

        no_deps_args = _build_pip_args(no_deps_requirements, constraints, no_deps=True,
                                       protect_environment=protect_environment)
        _executePythonModule("pip", no_deps_args, blocking=False,
                             logCallback=logCallback, completedCallback=onNoDepsComplete)
    except Exception:
        _pip_install_in_progress = False
        raise



def _build_pip_args(
    requirements: str | list[str],
    constraints: str | Path | None = None,
    no_deps: bool = False,
    protect_environment: bool = True,
) -> list[str]:
    """Build pip install command-line arguments.

    :param requirements: Package requirements (string or list).
    :param constraints: Path to constraints file, or None.
    :param no_deps: If True, add --no-deps flag.
    :param protect_environment: If True, auto-generate constraints pinning all
        currently installed packages to prevent pip from modifying them.
    :returns: List of arguments for pip install command.
    """
    if type(requirements) == str:
        args = ["install", *(shlex.split(requirements))]
    elif type(requirements) == list:
        args = ["install", *requirements]
    else:
        raise ValueError("pip_install requirement input must be string or list")

    if no_deps:
        args.append("--no-deps")

    # Auto-generated constraints to protect existing packages.
    # Skip when --no-deps is set since pip won't install dependencies anyway.
    if protect_environment and not no_deps:
        env_constraints = _generate_environment_constraints()
        args.extend(["-c", env_constraints])

    # User-provided constraints layered on top
    if constraints is not None:
        args.extend(["-c", str(constraints)])

    return args



def pip_uninstall(
    requirements: str | list[str],
    blocking: bool = True,
    logCallback: Callable[[str], None] | None = None,
    completedCallback: Callable[[int], None] | None = None,
) -> None:
    """Uninstall python packages.

    Currently, the method simply calls ``python -m pip uninstall`` but in the future further checks, optimizations,
    user confirmation may be implemented, therefore it is recommended to use this method call instead of a plain
    pip uninstall.

    :param requirements: requirement specifier in the same format as used by pip (https://docs.python.org/3/installing/index.html).
      It can be either a single string or a list of command-line arguments. It may be simpler to pass command-line arguments as a list
      if the arguments may contain spaces (because no escaping of the strings with quotes is necessary).
    :param blocking: If True (default), block until uninstall completes. If False, return immediately.
        Note: When running in PythonSlicer (without the full application), blocking mode
        is always used regardless of this setting.
    :param logCallback: When blocking=False, called with each line of pip output.
    :param completedCallback: When blocking=False, called when pip completes with return code.

    Example: calling from Slicer GUI

    .. code-block:: python

      pip_uninstall("tensorflow keras scikit-learn ipywidgets")

    Example: calling from PythonSlicer console

    .. code-block:: python

      from slicer.pydeps import pip_uninstall
      pip_uninstall("tensorflow")

    """
    if type(requirements) == str:
        # shlex.split splits string the same way as the shell (keeping quoted string as a single argument)
        args = ["uninstall", *(shlex.split(requirements)), "--yes"]
    elif type(requirements) == list:
        args = ["uninstall", *requirements, "--yes"]
    else:
        raise ValueError("pip_uninstall requirement input must be string or list")

    # In PythonSlicer, always use blocking mode
    if not _isSlicerAppAvailable():
        _executePythonModule("pip", args, blocking=True)
        return

    _executePythonModule("pip", args, blocking=blocking,
                         logCallback=logCallback, completedCallback=completedCallback)


class _PipProgressDialog:
    """Modal dialog showing pip installation progress with collapsible log details.

    Internal class used by :func:`pip_install` when show_progress=True and blocking=True.
    """

    def __init__(self, requester: str | None = None, parent: qt.QWidget | None = None) -> None:
        import ctk
        import qt
        import slicer

        self._dialog = qt.QDialog(parent or slicer.util.mainWindow())
        self._dialog.setModal(True)
        self._dialog.setWindowTitle(
            f"{requester} - Installing Python Packages" if requester else "Installing Python Packages",
        )
        # Prevent closing via X button
        self._dialog.setWindowFlags(self._dialog.windowFlags() & ~qt.Qt.WindowCloseButtonHint)

        # Prevent closing via Escape key by capturing it with a shortcut that does nothing
        self._escapeShortcut = qt.QShortcut(qt.QKeySequence(qt.Qt.Key_Escape), self._dialog)
        self._escapeShortcut.setContext(qt.Qt.WidgetWithChildrenShortcut)

        layout = qt.QVBoxLayout(self._dialog)

        # Status label
        self.statusLabel = qt.QLabel("Installing packages...")
        layout.addWidget(self.statusLabel)

        # Indeterminate progress bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 0)  # Indeterminate mode
        layout.addWidget(self.progressBar)

        # Collapsible details section
        self.detailsButton = ctk.ctkCollapsibleButton()
        self.detailsButton.text = "Details"
        self.detailsButton.collapsed = True
        detailsLayout = qt.QVBoxLayout(self.detailsButton)

        self.logText = qt.QPlainTextEdit()
        self.logText.setReadOnly(True)
        self.logText.setMinimumHeight(150)
        self.logText.setMaximumHeight(300)
        # Use monospace font for log output
        font = qt.QFont("Monospace")
        font.setStyleHint(qt.QFont.TypeWriter)
        self.logText.setFont(font)
        detailsLayout.addWidget(self.logText)

        layout.addWidget(self.detailsButton)

        # Set reasonable default size
        self._dialog.resize(500, 120)

        # Store log lines for retrieval
        self._logLines = []

    def show(self) -> None:
        """Show the dialog."""
        self._dialog.show()

    def close(self) -> None:
        """Close the dialog."""
        self._dialog.close()

    def appendLog(self, line: str) -> None:
        """Append a line to the log display."""
        self._logLines.append(line)
        self.logText.appendPlainText(line)
        # Auto-scroll to bottom
        scrollBar = self.logText.verticalScrollBar()
        scrollBar.setValue(scrollBar.maximum)

    def getFullLog(self) -> str:
        """Return the complete log as a string."""
        return "\n".join(self._logLines)
