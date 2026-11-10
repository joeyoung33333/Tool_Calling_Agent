"""Plotly figure specs, themed to match the terminal.

One method for every visualization: each function returns `{"data": [...], "layout": {...}}`, which
the page hands straight to `Plotly.newPlot()`. Nothing here renders pixels, which buys three things
the old hand-rolled SVG could not have:

- the charts are interactive (zoom, pan, hover readout, drag to compare two dates);
- the model never carries an image, because only data crosses the wire;
- tables, heatmaps and candlesticks come from the same library and the same theme as the lines.

The palette below is duplicated from index.html's `:root` block, so a theme change is two edits -
here and there. Keeping them separate avoids the page having to fetch its colours before it can draw.

Raises `ValueError` on bad input. These are caller bugs, not the model-facing `DataError`s that
`data.py` uses, because nothing a model types reaches here.
"""

import plotly.graph_objects as go

AMBER = "#ffb000"  # --amber: the primary series and header text
MUTED = "#b88400"  # --muted: axis labels and cell text
DIM = "#a97500"  # --dim
LINE = "#5c4000"  # --line: grid, axes and table rules
PANEL = "#0d0900"  # --panel: the plotting area and table header, as on tool-call rows
MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace'

# Series order for multi-line charts. No red or green: the page uses those for down and up (see colorize in index.html).
SERIES = [AMBER, "#e0d0a0", DIM, "#7fb3d5"]

_TEXT = {"color": MUTED, "family": MONO, "size": 11}
# Plotly sets the mode bar background inline, which page CSS cannot override, so its theme has to come from the figure.
MODEBAR = {"bgcolor": "rgba(0,0,0,0)", "color": "#9c6b00", "activecolor": AMBER}
# One readout for every series at the hovered x, which is what a crosshair is for.
HOVER = {
    "hovermode": "x unified",
    "hoverlabel": {"bgcolor": PANEL, "bordercolor": LINE, "font": {"color": AMBER, "family": MONO, "size": 11}},
}


def axis(label: str = "", **extra) -> dict:
    """One themed axis.

    Pass the result as xaxis/yaxis/yaxis2/..., so a two-panel figure reuses the theme instead of
    copying it.
    """
    return {
        "gridcolor": LINE,
        "zerolinecolor": LINE,
        "linecolor": LINE,
        "tickfont": _TEXT,
        "title": {"text": label, "font": _TEXT},
        # Size the gutter from the tick labels actually drawn, so a narrow screen keeps its width.
        "automargin": True,
        **extra,
    }


def _layout(title: str = "", height: int = 320, margin: dict | None = None, **extra) -> dict:
    """The shared terminal look: title, fonts, colours, margins.

    Axes and interaction are the caller's, because a table wants neither and a heatmap wants
    different ones. That is why nothing here has to be removed afterwards.

    Margins are small and the axes carry `automargin`, so Plotly grows the gutters only as far as
    the labels need. The page resizes `height` on narrow screens (see fitChart in index.html).
    """
    return {
        "title": {
            "text": title,
            "font": {"color": AMBER, "family": MONO, "size": 13},
            "x": 0.01,
            "automargin": True,  # grow the top margin rather than letting a long title be clipped
        },
        "height": height,
        "margin": margin or {"l": 12, "r": 12, "t": 34 if title else 10, "b": 12},
        "paper_bgcolor": "rgba(0,0,0,0)",  # the page's own backdrop shows through
        "plot_bgcolor": PANEL,
        "font": _TEXT,
        "showlegend": False,
        "modebar": MODEBAR,
        **extra,
    }


def line(
    series: list[tuple[str, list[tuple[str, float]]]],
    title: str = "",
    y_label: str = "",
    height: int = 320,
) -> dict:
    """A line chart of one or more named series, each a list of (x label, y value), left to right.

    x labels are used as given (dates, strikes, contract months), so the same function charts any
    series. Pass several to overlay them, which is what the spread and equity charts need.
    """
    if not series or any(len(points) < 2 for _, points in series):
        raise ValueError("a line chart needs at least 2 points in every series")
    traces = []
    for i, (label, points) in enumerate(series):
        traces.append(
            go.Scatter(
                x=[x for x, _ in points],
                y=[y for _, y in points],
                name=label,
                mode="lines",
                line={"color": SERIES[i % len(SERIES)], "width": 1.6},
                hovertemplate="%{y:,.2f}<extra>" + label + "</extra>",
            ).to_plotly_json()
        )
    layout = _layout(
        title,
        height,
        xaxis=axis(),
        yaxis=axis(y_label),
        dragmode="zoom",  # drag selects an x-range; the page reports the move across it
        **HOVER,
    )
    if len(series) > 1:
        # Below the title, not beside it, so a long title cannot overlap the entries.
        layout["showlegend"] = True
        layout["legend"] = {"font": {"color": MUTED, "family": MONO, "size": 10}, "orientation": "h", "y": -0.18}
    return {"data": traces, "layout": layout}


def table(headers: list[str], rows: list[list], title: str = "", height: int | None = None) -> dict:
    """A table, themed like the rest of the terminal.

    `rows` is row-major, as a reader would write it; Plotly wants columns, so it is transposed here
    rather than at every call site.
    """
    if not headers:
        raise ValueError("a table needs at least one column")
    columns = [[row[i] if i < len(row) else "" for row in rows] for i in range(len(headers))]
    trace = go.Table(
        header={
            "values": [f"<b>{h}</b>" for h in headers],
            "fill_color": PANEL,
            "line_color": LINE,
            "align": "left",
            "font": {"color": AMBER, "family": MONO, "size": 11},
            "height": 26,
        },
        cells={
            "values": columns,
            "fill_color": "rgba(0,0,0,0)",
            "line_color": LINE,
            "align": "left",
            "font": {"color": MUTED, "family": MONO, "size": 11},
            "height": 24,
        },
    ).to_plotly_json()
    # The cells get height - margin.t - margin.b - header: miscount any term and the last row is cut off, with no visible scrollbar.
    margin = {"l": 8, "r": 8, "t": 34 if title else 8, "b": 8}
    header, row, slack = 30, 24, 8  # a bold header grows past 26 once Plotly measures the markup
    auto = margin["t"] + margin["b"] + header + row * max(len(rows), 1) + slack
    return {"data": [trace], "layout": _layout(title, height or min(auto, 520), margin)}


def zscore_panel(
    points: list[tuple[str, float]],
    entry_z: float,
    exit_z: float,
    title: str = "",
    episodes: list[dict] | None = None,
    height: int = 340,
) -> dict:
    """The spread's z-score over time, with the entry and exit bands drawn on it.

    This is the one chart that makes a pair trade legible: the bands show where a trade opens and
    closes, and the shaded runs mark divergences that stayed wide long enough to need explaining.
    """
    if len(points) < 2:
        raise ValueError("a z-score panel needs at least 2 points")
    x = [d for d, _ in points]
    trace = go.Scatter(
        x=x, y=[z for _, z in points], mode="lines", name="z",
        line={"color": AMBER, "width": 1.6}, hovertemplate="z = %{y:.2f}<extra></extra>",
    ).to_plotly_json()

    # Horizontal bands rather than extra traces, so the legend stays empty and hover stays clean.
    lines = []
    for level, colour, dash in [(entry_z, DIM, "dash"), (-entry_z, DIM, "dash"),
                                (exit_z, LINE, "dot"), (-exit_z, LINE, "dot"), (0, MUTED, "solid")]:
        lines.append({"type": "line", "xref": "paper", "x0": 0, "x1": 1, "yref": "y",
                      "y0": level, "y1": level, "line": {"color": colour, "width": 1, "dash": dash}})
    for episode in episodes or []:
        lines.append({"type": "rect", "xref": "x", "yref": "paper",
                      "x0": episode["start"], "x1": episode["end"], "y0": 0, "y1": 1,
                      "fillcolor": DIM, "opacity": 0.14, "line": {"width": 0}, "layer": "below"})

    layout = _layout(title, height, xaxis=axis(), yaxis=axis("z-score"), shapes=lines, **HOVER)
    return {"data": [trace], "layout": layout}
