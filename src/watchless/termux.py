"""termux: install the Android share-sheet target.

Termux registers an ACTION_SEND receiver for text/plain. Sharing a link and
picking Termux makes that receiver exec the FIXED path ``$HOME/bin/termux-url-opener``
with the shared text as ``$1``. Nothing scans PATH -- the path is compiled into
the app -- so "setup" means putting a working script at exactly that path.

This module does that, and refuses to pretend it worked:

* If it is not running under Termux there is no point pretending: the path would
  be created somewhere useless, and the user would share a link months later and
  watch nothing happen.
* The opener's useful behaviour depends on three pieces that a bare Termux
  install does not have: the Termux:API app, the ``termux-am`` command (which
  makes progress visible instead of a blank flash), and ``termux-open`` (which
  opens the finished transcript). Missing them is not fatal -- the opener
  degrades -- so they are reported, not required.
* An existing opener is never overwritten without being asked. It might be
  someone else's script, or a version they hand-edited.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

# Termux sets PREFIX to its own private usr dir. Checking $HOME alone is not
# enough: a plain Linux user can have a ~/bin too, and macOS has $HOME as well.
_TERMUX_PREFIX = "/data/data/com.termux/files/usr"

# The exact path the Termux app execs. Not configurable: the receiver does not
# read a config file, so offering a flag would be offering a way to break it.
OPENER_PATH = Path("$HOME/bin/termux-url-opener")


@dataclass(frozen=True, slots=True)
class Requirement:
    """One thing the opener uses, and what to do when it is absent.

    ``fatal`` separates "the share sheet cannot work" (no Termux at all) from
    "the share sheet works, with a worse experience" (no toast).
    """

    name: str
    command: str
    package: str
    why: str
    fatal: bool = False


REQUIREMENTS: tuple[Requirement, ...] = (
    Requirement(
        name="Termux",
        command="termux-info",
        package="",
        why="the share target is only read by the Termux app",
        fatal=True,
    ),
    Requirement(
        name="Termux:API command line",
        command="termux-toast",
        package="termux-api",
        why="notifications and toasts; without it the run is silent",
    ),
    Requirement(
        name="Termux:API (am)",
        command="termux-am",
        package="termux-api",
        why="runs the fetch in a visible session instead of a blank flash",
    ),
    Requirement(
        name="Termux:API (open)",
        command="termux-open",
        package="termux-api",
        why="opens the finished transcript instead of leaving you to find it",
    ),
)


def installer_path() -> Path:
    """Where the opener goes: always $HOME/bin/termux-url-opener."""
    return Path.home() / "bin" / "termux-url-opener"


def in_termux() -> bool:
    """Are we running inside Termux?

    PREFIX is the reliable signal: Termux sets it to
    ``/data/data/com.termux/files/usr``. Checking only $HOME would match every
    Unix user with a home directory.

    The marker is matched as a substring rather than a prefix so the check can
    be exercised from a test with a rooted fake, and because installs relocated
    by Termux itself keep ``com.termux`` in the path either way.
    """
    prefix = os.environ.get("PREFIX", "")
    return "com.termux" in prefix or Path(_TERMUX_PREFIX).is_dir()


def missing(reqs: tuple[Requirement, ...] = REQUIREMENTS) -> list[Requirement]:
    """Which requirements are not on PATH."""
    return [r for r in reqs if shutil.which(r.command) is None]


def _pkg_install_hint(missing_reqs: list[Requirement]) -> list[str]:
    """The commands the user has to run, one per package, plus the app.

    ``pkg install termux-api`` gives the command-line tools; the Termux:API
    *app* is a separate F-Droid install and no pkg command can do it. Saying
    only the pkg part is how people end up with termux-toast present and
    "Termux:API app not installed" as the error.
    """
    packages = sorted({r.package for r in missing_reqs if r.package})
    lines = [f"  pkg install {' '.join(packages)}"] if packages else []
    lines += [
        "  and install the Termux:API app from F-Droid, then run",
        "  this command again (the app and the package are separate):",
        "    https://f-droid.org/packages/com.termux.api/",
    ]
    return lines


def _report_missing(reqs: list[Requirement]) -> None:
    for req in reqs:
        print(f"  missing: {req.name} ({req.command}) -- {req.why}")


def _confirm(prompt: str) -> bool:
    """Ask, defaulting to no.

    Only asks when a human can answer. With stdin redirected -- a pipe, cron, a
    script -- input() blocks forever waiting for a line that is never coming,
    which is worse than refusing: the command appears to hang. Silent refusal
    matches the default, and the caller prints how to force it.
    """
    if not sys.stdin.isatty():
        print("(not a terminal, so not asking; use --force to overwrite)")
        return False
    try:
        answer = input(f"{prompt} [y/N] ")
    except EOFError:
        print()
        return False
    return answer.strip().lower() in {"y", "yes"}


def install_opener(*, force: bool = False, assume_yes: bool = False) -> int:
    """Copy the bundled opener to ~/bin/termux-url-opener. -> exit status."""
    dest = installer_path()

    if not in_termux():
        print("not running inside Termux, so there is no share sheet to install.")
        print("The opener is only read by the Termux Android app, at the fixed")
        print(f"path {dest}. Nothing has been written.")
        print()
        print("On Termux, run:  watchless --install-termux-opener")
        return 1

    absent = missing()
    fatal = [r for r in absent if r.fatal]
    if fatal:
        print("this does not look like a usable Termux install:")
        _report_missing(fatal)
        print()
        print("Install Termux from F-Droid (not the Play Store copy, which is")
        print("deprecated), then run this command again.")
        return 1

    source = files("watchless.data").joinpath("termux-url-opener")
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:  # a wheel built without package-data
        print(f"the bundled opener is missing from this install: {exc}")
        print("This is a packaging fault, not something you can fix locally.")
        return 1

    # Refuse to silently replace something, including a previous install. The
    # file may be a hand-edited copy, or somebody else's script entirely.
    if dest.exists() and not force:
        print(f"an opener already exists: {dest}")
        print(f"  {dest.stat().st_size} bytes, modified {_mtime(dest)}")
        if not (assume_yes or _confirm("Overwrite it?")):
            print("left unchanged.")
            return 0
    elif dest.exists() and force:
        print(f"overwriting {dest} (--force)")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        dest.chmod(0o755)
    except OSError as exc:
        print(f"cannot write {dest}: {exc}")
        return 1

    print(f"installed {dest}")
    _report_installed(dest, absent)
    return 0


def _report_installed(dest: Path, absent: list[Requirement]) -> None:
    """What to do next, including anything still missing."""
    if absent:
        print()
        print("installed, but not everything it uses is available:")
        _report_missing(absent)
        print()
        print("To get the full experience:")
        for line in _pkg_install_hint(absent):
            print(line)

    print()
    print("Sharing a YouTube link to Termux now saves a transcript and opens it.")
    print("Configuration lives in ~/.config/watchless.env (exported into the opener):")
    print("  WATCHLESS_BASE_URL ... to summarise from the share sheet too")
    print("  WATCHLESS_DEST ....... where the .txt lands")
    print()
    print("Test it without sharing anything:")
    print(f'  bash {shlex.quote(str(dest))} "https://youtu.be/VIDEOID"')


def _mtime(path: Path) -> str:
    import datetime  # noqa: PLC0415

    return datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M"
    )


def main(argv: list[str] | None = None) -> int:
    """>= returns an exit status rather than raising, so it is testable."""
    args = argv if argv is not None else sys.argv[1:]
    force = "--force" in args
    assume_yes = "--yes" in args or "-y" in args
    return install_opener(force=force, assume_yes=assume_yes)
