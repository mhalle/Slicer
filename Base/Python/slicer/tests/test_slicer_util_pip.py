"""Tests verifying slicer.util.pip_install/pip_uninstall wrappers delegate to slicer.pydeps."""

import unittest
import unittest.mock

import slicer
import slicer.pydeps
import slicer.util


class UtilPipInstallWrapperTest(unittest.TestCase):
    """Tests that slicer.util.pip_install delegates to slicer.pydeps.pip_install."""

    def test_delegates_positional_arg(self):
        """Test that a simple positional argument is forwarded."""
        with unittest.mock.patch("slicer.pydeps.pip_install") as mock:
            slicer.util.pip_install("some-package")
            mock.assert_called_once_with("some-package")

    def test_delegates_kwargs(self):
        """Test that keyword arguments are forwarded."""
        with unittest.mock.patch("slicer.pydeps.pip_install") as mock:
            slicer.util.pip_install("pkg", constraints="/tmp/c.txt", show_progress=False)
            mock.assert_called_once_with("pkg", constraints="/tmp/c.txt", show_progress=False)

    def test_delegates_protect_environment(self):
        """Test that protect_environment kwarg is forwarded."""
        with unittest.mock.patch("slicer.pydeps.pip_install") as mock:
            slicer.util.pip_install("pkg", protect_environment=False)
            mock.assert_called_once_with("pkg", protect_environment=False)


class UtilPipUninstallWrapperTest(unittest.TestCase):
    """Tests that slicer.util.pip_uninstall delegates to slicer.pydeps.pip_uninstall."""

    def test_delegates_positional_arg(self):
        """Test that a simple positional argument is forwarded."""
        with unittest.mock.patch("slicer.pydeps.pip_uninstall") as mock:
            slicer.util.pip_uninstall("some-package")
            mock.assert_called_once_with("some-package")

    def test_delegates_kwargs(self):
        """Test that keyword arguments are forwarded."""
        with unittest.mock.patch("slicer.pydeps.pip_uninstall") as mock:
            slicer.util.pip_uninstall("pkg", blocking=False)
            mock.assert_called_once_with("pkg", blocking=False)
