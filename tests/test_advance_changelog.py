from datetime import date
from unittest.mock import patch

import pytest

from scripts.advance_changelog import advance_changelog, remove_empty_subsections


def test_remove_empty_subsections():
    """Test removing empty markdown subsections while keeping populated ones."""
    sample = (
        "### 🚨 Breaking Changes\n\n"
        "### 🚀 Features\n\n"
        "- Add shiny new feature\n\n"
        "### 🐛 Bug fixes\n\n"
        "### 🌟 First Time Contributors\n\n"
    )
    cleaned = remove_empty_subsections(sample)
    expected = "### 🚀 Features\n\n- Add shiny new feature"
    assert cleaned == expected


def test_advance_changelog():
    """Test advancing the changelog and extracting release notes."""
    sample_changelog = (
        "# Changelog\n\n"
        "## v0.8.5 [🚧 Work in Progress]\n\n"
        "### 🚨 Breaking Changes\n\n"
        "### 🚀 Features\n\n"
        "- First feature (see #1) (@user)\n\n"
        "### 🐛 Bug fixes\n\n"
        "- Fixed bug (see #2) (@user)\n\n"
        "### 🌟 First Time Contributors\n\n"
        "## [v0.8.4](https://github.com/desy-ml/cheetah/releases/tag/v0.8.4) "
        "(2026-07-23)\n\n"
        "- Older release note\n"
    )

    with patch("scripts.advance_changelog.date") as mock_date:
        mock_date.today.return_value = date(2026, 11, 6)
        new_content, notes = advance_changelog(sample_changelog, "0.8.5", "0.8.6")

    # Assert new WIP header exists at the top
    assert "## v0.8.6 [🚧 Work in Progress]" in new_content
    # Assert release header has been created with expected URL and date
    expected_header = (
        "## [v0.8.5](https://github.com/desy-ml/cheetah/releases/tag/v0.8.5) "
        "(2026-11-06)"
    )
    assert expected_header in new_content
    # Assert empty subsections were pruned from release section
    assert "### 🚨 Breaking Changes" not in notes
    assert "### 🌟 First Time Contributors" not in notes
    # Assert populated subsections are in notes
    assert "### 🚀 Features\n\n- First feature (see #1) (@user)" in notes
    assert "### 🐛 Bug fixes\n\n- Fixed bug (see #2) (@user)" in notes
    # Assert older release section is preserved
    expected_older = (
        "## [v0.8.4](https://github.com/desy-ml/cheetah/releases/tag/v0.8.4) "
        "(2026-07-23)"
    )
    assert expected_older in new_content


def test_advance_changelog_strips_leading_v():
    """Test that leading `v` prefixes on versions are stripped."""
    sample_changelog = "# Changelog\n\n## v0.8.5 [🚧 Work in Progress]\n\n"
    with patch("scripts.advance_changelog.date") as mock_date:
        mock_date.today.return_value = date(2026, 11, 6)
        new_content, _ = advance_changelog(sample_changelog, "v0.8.5", "v0.8.6")

    assert "## v0.8.6 [🚧 Work in Progress]" in new_content
    assert (
        "## [v0.8.5](https://github.com/desy-ml/cheetah/releases/tag/v0.8.5) "
        "(2026-11-06)"
    ) in new_content


@pytest.mark.parametrize(
    "bad_version", ["0.8", "v", "0.8.5.1", "abc", "0.8.5-rc1", "vv0.8.5"]
)
def test_advance_changelog_rejects_invalid_version(bad_version: str):
    """Test that malformed version strings raise `ValueError`."""
    sample_changelog = "# Changelog\n\n## v0.8.5 [🚧 Work in Progress]\n\n"
    with pytest.raises(ValueError, match="Invalid version"):
        advance_changelog(sample_changelog, bad_version, "0.8.6")
    with pytest.raises(ValueError, match="Invalid version"):
        advance_changelog(sample_changelog, "0.8.5", bad_version)
