import matplotlib.axis
import numpy as np


def format_axis_with_prefixed_unit(
    axis: matplotlib.axis.Axis, base_unit: str, data: list[float]
) -> None:
    """
    Adds an appropriately prefixed unit to the axis label and sets the tick formatter
    accordingly to best match the given data.
    """
    prefixed_unit, tick_formatter = determine_prefixed_unit_and_tick_formatter(
        base_unit, data
    )
    axis.set_label_text(f"{axis.get_label_text()} ({prefixed_unit})")
    axis.set_major_formatter(tick_formatter)
    axis.set_minor_formatter(tick_formatter)


def format_axis_as_percentage(axis: matplotlib.axis.Axis) -> None:
    """
    Adds a percentage symbol to the axis label and sets the tick formatter accordingly.
    """
    axis.set_label_text(f"{axis.get_label_text()} (%)")
    axis.set_major_formatter(NoSymbolPercentFormatter())
    axis.set_minor_formatter(NoSymbolPercentFormatter())


def determine_prefixed_unit_and_tick_formatter(
    base_unit: str, data: list[float]
) -> tuple[str, matplotlib.ticker.FuncFormatter]:
    """
    Considering the order of magnitude of some data points and their base unit,
    determines the prefixed unit and the corresponding matplotlib tick formatter.
    """
    if 1.0 <= np.max(np.abs(data)) < 1e3:
        return base_unit, IdentityFormatter()
    elif 1e-3 <= np.max(np.abs(data)) < 1.0:
        return f"m{base_unit}", MilliFormatter()
    elif 1e-6 <= np.max(np.abs(data)) < 1e-3:
        return f"μ{base_unit}", MicroFormatter()
    else:
        return base_unit, IdentityFormatter()


class NoSymbolPercentFormatter(matplotlib.ticker.FuncFormatter):
    """Formatter for percentages without the percent symbol."""

    def __init__(self):
        super().__init__(lambda x, _: f"{x * 100:.1f}")


class IdentityFormatter(matplotlib.ticker.FuncFormatter):
    """Formatter for base values."""

    def __init__(self):
        super().__init__(lambda x, _: f"{x:.0f}")


class MilliFormatter(matplotlib.ticker.FuncFormatter):
    """Formatter for milli values."""

    def __init__(self):
        super().__init__(lambda x, _: f"{x * 1e3:.0f}")


class MicroFormatter(matplotlib.ticker.FuncFormatter):
    """Formatter for micro values."""

    def __init__(self):
        super().__init__(lambda x, _: f"{x * 1e6:.0f}")
