"""Plotly figure specs, themed to match the terminal.

Each function returns `{"data": [...], "layout": {...}}`, which the page hands straight to
`Plotly.newPlot()`. Nothing here renders pixels: the charts stay interactive, and only data crosses
the wire, so the model never carries an image.

The palette is duplicated from index.html's `:root` block, so a theme change is two edits. Keeping
them separate saves the page a fetch before it can draw.

Raises `ValueError` on bad input: these are caller bugs, not the model-facing `DataError`s that
`sources/data.py` raises, because nothing a model types reaches here.
"""

import plotly.graph_objects as go

AMBER = "#ffb000"  # --amber: the primary series and header text
MUTED = "#b88400"  # --muted: axis labels and cell text
DIM = "#a97500"  # --dim
LINE = "#5c4000"  # --line: grid, axes and table rules
PANEL = "#0d0900"  # --panel: the plotting area and table header, as on tool-call rows
MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace'

_TEXT = {"color": MUTED, "family": MONO, "size": 11}
# Plotly sets the mode bar background inline, so page CSS cannot reach it; its theme comes from here.
MODEBAR = {"bgcolor": "rgba(0,0,0,0)", "color": "#9c6b00", "activecolor": AMBER}
HOVER = {
    "hovermode": "x unified",
    "hoverlabel": {"bgcolor": PANEL, "bordercolor": LINE, "font": {"color": AMBER, "family": MONO, "size": 11}},
}


def axis(label: str = "") -> dict:
    """One themed axis, optionally labelled."""
    return {
        "gridcolor": LINE,
        "zerolinecolor": LINE,
        "linecolor": LINE,
        "tickfont": _TEXT,
        "title": {"text": label, "font": _TEXT},
        "automargin": True,  # size the gutter from the labels drawn, so a narrow screen keeps its width
    }


def _layout(title: str = "", height: int = 320, margin: dict | None = None, **extra) -> dict:
    """The shared terminal look, plus whatever the caller adds in `extra`.

    `extra` must stay in the returned dict and merge last. Dropping it costs every chart its themed
    axes and hover, and the z-score panel the bands that are the only reason to look at it.
    """
    return {
        "title": {
            "text": title,
            "font": {"color": AMBER, "family": MONO, "size": 13},
            # Anchored to the container's top edge with explicit padding. Left to position itself,
            # Plotly puts the title's top flush against that edge and the first row of pixels is
            # clipped by the rounded border; `automargin` grows the gutter but cannot add padding
            # above the text itself.
            "xref": "container", "x": 0.012, "xanchor": "left",
            "yref": "container", "y": 1.0, "yanchor": "top",
            "pad": {"t": 12, "l": 2},
            "automargin": True,
        },
        "height": height,
        # 34px clipped a two-line title on a narrow screen. `automargin` grows the gutter but
        # cannot shrink a margin we set too small to begin with.
        "margin": margin or {"l": 12, "r": 12, "t": 44 if title else 10, "b": 12},
        "paper_bgcolor": "rgba(0,0,0,0)",  # the page's own backdrop shows through
        "plot_bgcolor": PANEL,
        "font": _TEXT,
        "showlegend": False,
        "modebar": MODEBAR,
        **extra,
    }


def line(label: str, points: list[tuple[str, float]], title: str = "", y_label: str = "") -> dict:
    """A line chart of one named series: `points` is (x label, y value), left to right.

    x labels are used as given, so this charts a date series or any other ordered one.
    """
    if len(points) < 2:
        raise ValueError("a line chart needs at least 2 points")
    trace = go.Scatter(
        x=[x for x, _ in points],
        y=[y for _, y in points],
        name=label,
        mode="lines",
        line={"color": AMBER, "width": 1.6},
        hovertemplate="%{y:,.2f}<extra>" + label + "</extra>",
    ).to_plotly_json()
    return {"data": [trace], "layout": _layout(
        title,
        xaxis=axis(),
        yaxis=axis(y_label),
        dragmode="zoom",  # drag selects an x-range; the page reports the move across it
        **HOVER,
    )}


def table(headers: list[str], rows: list[list], title: str = "") -> dict:
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
    # Cells get height - margin.t - margin.b - header. Miscount any term and the last row is cut
    # off with no visible scrollbar, so the height is computed rather than guessed.
    margin = {"l": 8, "r": 8, "t": 44 if title else 8, "b": 8}  # matches the title padding in _layout
    header, row, slack = 30, 24, 8  # a bold header grows past 26 once Plotly measures the markup
    auto = margin["t"] + margin["b"] + header + row * max(len(rows), 1) + slack
    return {"data": [trace], "layout": _layout(title, min(auto, 520), margin)}


def zscore_panel(
    points: list[tuple[str, float]],
    entry_z: float,
    exit_z: float,
    title: str = "",
    episodes: list[dict] | None = None,
    height: int = 340,
) -> dict:
    """The spread's z-score over time, with the entry and exit bands drawn on it.

    The chart that makes a pair trade legible: the bands show where a trade opens and closes, and
    the shaded runs mark divergences that stayed wide long enough to need explaining.
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
