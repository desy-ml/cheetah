import argparse
import os
import re
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the changelog update script."""
    parser = argparse.ArgumentParser(
        description=(
            "Update CHANGELOG.md and extract release notes for a Cheetah release."
        )
    )
    parser.add_argument("version", help="Version to release (e.g. 0.8.5)")
    parser.add_argument(
        "next_version", help="Next version for WIP changelog (e.g. 0.8.6)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without modifying files",
    )

    return parser.parse_args()


def remove_empty_subsections(text: str) -> str:
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
        r"(##\s+v?[0-9.]+\s+\[🚧 Work in Progress\]\s*\n)(.*?)(?=\n##\s|\Z)", re.DOTALL
    )
    match = wip_pattern.search(content)
    raw_notes = match.group(2)
    cleaned_notes = remove_empty_subsections(raw_notes)
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


def main() -> None:
    """Update changelog and write release notes."""
    args = parse_args()
    changelog_path = Path("CHANGELOG.md")
    new_content, notes = update_changelog(
        changelog_path, args.version, args.next_version
    )

    if args.dry_run:
        print(f"[dry-run] Would update `CHANGELOG.md` for v{args.version}")
        print(f"[dry-run] Next WIP version would be v{args.next_version}")
        print("\n[dry-run] Release notes:\n" + notes)
        return

    changelog_path.write_text(new_content, encoding="utf-8")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"release_notes<<EOF\n{notes}\nEOF\n")
    print(f"Updated `CHANGELOG.md` for v{args.version}")


if __name__ == "__main__":
    main()
