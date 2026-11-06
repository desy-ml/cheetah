import os
import re
import sys
from datetime import date
from pathlib import Path


def remove_empty_subsections(text: str) -> str:
    """Remove empty `###` subsections from markdown text."""
    return re.sub(r"###\s+[^\n]+\n+(?=\s*###|\s*$)", "", text).strip()


def advance_changelog(content: str, version: str, next_version: str) -> tuple[str, str]:
    """Advance the changelog, freeze current WIP release, and add new template.

    :param content: Current `CHANGELOG.md` content
    :param version: Version being released
    :param next_version: Version for the new work-in-progress section
    :return: Tuple of (new `CHANGELOG.md` text, extracted release notes text)
    """
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
    """Advance changelog and write release notes."""
    version, next_version = sys.argv[1], sys.argv[2]
    changelog_path = Path("CHANGELOG.md")
    content = changelog_path.read_text(encoding="utf-8")
    new_content, notes = advance_changelog(content, version, next_version)

    changelog_path.write_text(new_content, encoding="utf-8")
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
        f.write(f"release_notes<<EOF\n{notes}\nEOF\n")


if __name__ == "__main__":
    main()
