"""Automate cutting a new Cheetah release."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the release script."""
    parser = argparse.ArgumentParser(description="Prepare and cut a Cheetah release.")
    parser.add_argument("version", help="Version to release (e.g. 0.8.5)")
    parser.add_argument(
        "--next-version",
        help="Next version for WIP changelog (defaults to next patch version)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying files or running git commands",
    )
    parser.add_argument(
        "--changelog-only",
        action="store_true",
        help="Only update CHANGELOG.md and write RELEASE_NOTES.md without git commands",
    )
    return parser.parse_args()


def default_next_version(version: str) -> str:
    """Calculate the default next patch version string."""
    parts = version.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except ValueError:
        return f"{version}.1"


def clean_section(text: str) -> str:
    """Remove empty `###` subsections from markdown text."""
    return re.sub(r"###\s+[^\n]+\n+(?=\s*###|\s*$)", "", text).strip()


def update_changelog(
    changelog_path: Path, version: str, next_version: str
) -> tuple[str, str]:
    """Advance the changelog, freeze current WIP release, and add new template.

    :param changelog_path: Path to `CHANGELOG.md`
    :param version: Version being released
    :param next_version: Version for the new work-in-progress section
    :return: Tuple of (new `CHANGELOG.md` text, extracted release notes text)
    """
    content = changelog_path.read_text(encoding="utf-8")

    wip_pattern = re.compile(
        r"(##\s+v?([0-9.]+)\s+\[🚧 Work in Progress\]\s*\n)(.*?)"
        r"(?=\n##\s+\[v|\n##\s+v|\Z)",
        re.DOTALL,
    )
    match = wip_pattern.search(content)
    if not match:
        raise ValueError(
            "Could not find '## vX.Y.Z [🚧 Work in Progress]' in `CHANGELOG.md`."
        )

    raw_notes = match.group(3)
    cleaned_notes = clean_section(raw_notes)
    today = date.today().isoformat()

    release_url = f"https://github.com/desy-ml/cheetah/releases/tag/v{version}"
    new_release_header = f"## [v{version}]({release_url}) ({today})"

    new_wip_section = (
        f"## v{next_version} [🚧 Work in Progress]\n\n"
        "### 🚨 Breaking Changes\n\n"
        "### 🚀 Features\n\n"
        "### 🐛 Bug fixes\n\n"
        "### 🐆 Other\n\n"
        "### 🌟 First Time Contributors\n\n"
    )

    replacement = f"{new_wip_section}{new_release_header}\n\n{cleaned_notes}\n"
    new_content = content[: match.start()] + replacement + content[match.end() :]
    return new_content, cleaned_notes


def main() -> int:
    """Run the release workflow."""
    args = parse_args()
    version = args.version.lstrip("v")
    next_version = (
        args.next_version.lstrip("v")
        if args.next_version
        else default_next_version(version)
    )

    changelog_path = Path("CHANGELOG.md")
    new_content, notes = update_changelog(changelog_path, version, next_version)

    if args.dry_run:
        print(f"[dry-run] Would update `CHANGELOG.md` for v{version}")
        print(f"[dry-run] Next WIP version would be v{next_version}")
        print("\n[dry-run] Release notes:\n" + notes)
        return 0

    changelog_path.write_text(new_content, encoding="utf-8")
    notes_path = Path("RELEASE_NOTES.md")
    notes_path.write_text(notes + "\n", encoding="utf-8")

    if args.changelog_only:
        print(f"Updated `CHANGELOG.md` and wrote `RELEASE_NOTES.md` for v{version}")
        return 0

    tag = f"v{version}"
    subprocess.run(["git", "add", "CHANGELOG.md"], check=True)
    subprocess.run(["git", "commit", "-m", f"Prepare {tag} release"], check=True)
    subprocess.run(["git", "tag", "-a", tag, "-m", f"Release {tag}"], check=True)
    subprocess.run(["git", "push", "origin", "master", tag], check=True)
    subprocess.run(
        [
            "gh",
            "release",
            "create",
            tag,
            "--title",
            tag,
            "--notes-file",
            str(notes_path),
        ],
        check=True,
    )
    notes_path.unlink(missing_ok=True)
    print(f"Successfully released {tag}!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
