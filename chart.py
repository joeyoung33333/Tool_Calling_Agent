"""Draw a simple line chart as an SVG string. Pure Python, no dependencies, so it works for any (label, value) series."""

from xml.sax.saxutils import escape

FONT = "ui-monospace, Menlo, Consolas, monospace"


def _number(value: float, span: float) -> str:
    """Format an axis value with enough decimals for the span of the axis."""
    decimals = 0 if span >= 200 else 1 if span >= 20 else 2 if span >= 2 else 4
    return f"{value:,.{decimals}f}"


def line_chart_svg(
    points: list[tuple[str, float]],
    title: str = "",
    y_label: str = "",
    width: int = 720,
    height: int = 300,
    line: str = "#ffb000",
    text: str = "#b88400",
    grid: str = "#2a1f00",
) -> str:
    """Return an SVG line chart of `points`, a list of (x label, y value), left to right.

    x labels are drawn as given (dates, strikes, contract months), so the same function charts any series.
    """
    if len(points) < 2:
        raise ValueError("a line chart needs at least 2 points")
    left, right, top, bottom = 64, 16, 34 if title else 16, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for _, v in points]
    low, high = min(values), max(values)
    pad = (high - low) * 0.08 or 1  # a flat series still needs some height
    low, high = low - pad, high + pad

    def x(i: int) -> float:
        return left + i * plot_w / (len(points) - 1)

    def y(v: float) -> float:
        return top + (high - v) / (high - low) * plot_h

    def label(px: float, py: float, s: str, anchor: str = "start") -> str:
        return f'<text x="{px:.1f}" y="{py:.1f}" fill="{text}" font-size="11" font-family="{FONT}" text-anchor="{anchor}">{escape(s)}</text>'

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">']
    if title:
        parts.append(label(left, 20, title))
    for k in range(5):  # horizontal grid lines with y values
        v = low + (high - low) * k / 4
        parts.append(f'<line x1="{left}" y1="{y(v):.1f}" x2="{width - right}" y2="{y(v):.1f}" stroke="{grid}"/>')
        parts.append(label(left - 8, y(v) + 4, _number(v, high - low), "end"))
    if y_label:
        parts.append(f'<text transform="translate(14 {top + plot_h / 2:.1f}) rotate(-90)" fill="{text}" font-size="11" font-family="{FONT}" text-anchor="middle">{escape(y_label)}</text>')
    ticks = sorted({round(i * (len(points) - 1) / 4) for i in range(5)})  # first, last and three in between
    for i in ticks:
        anchor = "start" if i == 0 else "end" if i == len(points) - 1 else "middle"
        parts.append(label(x(i), height - 10, points[i][0], anchor))
    path = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, (_, v) in enumerate(points))
    parts.append(f'<polygon points="{x(0):.1f},{top + plot_h} {path} {x(len(points) - 1):.1f},{top + plot_h}" fill="{line}" fill-opacity="0.1"/>')
    parts.append(f'<polyline points="{path}" fill="none" stroke="{line}" stroke-width="2" stroke-linejoin="round"/>')
    parts.append(f'<circle cx="{x(len(points) - 1):.1f}" cy="{y(values[-1]):.1f}" r="3.5" fill="{line}"/>')
    parts.append("</svg>")
    return "".join(parts)
