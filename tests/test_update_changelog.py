from datetime import date
from unittest.mock import patch

from scripts.update_changelog import advance_changelog, remove_empty_subsections


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

    with patch("scripts.update_changelog.date") as mock_date:
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
