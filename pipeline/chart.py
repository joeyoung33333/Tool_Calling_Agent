"""Plotly figure specs, themed to match the terminal; the page passes them to Plotly.newPlot().

Palette duplicated from index.html's `:root`. Raises ValueError on bad input (caller bugs).
"""

import plotly.graph_objects as go

AMBER = "#ffb000"  # --amber: the primary series and header text
MUTED = "#b88400"  # --muted: axis labels and cell text
DIM = "#a97500"  # --dim
LINE = "#5c4000"  # --line: grid, axes and table rules
PANEL = "#0d0900"  # --panel: the plotting area and table header, as on tool-call rows
MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace'

_TEXT = {"color": MUTED, "family": MONO, "size": 11}
# Plotly sets the mode bar background inline, so page CSS cannot reach it.
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
        "automargin": True,
    }


def _layout(title: str = "", height: int = 320, margin: dict | None = None, **extra) -> dict:
    """The shared terminal look; `extra` must merge last or charts lose their axes and bands."""
    return {
        "title": {
            "text": title,
            "font": {"color": AMBER, "family": MONO, "size": 13},
            # Anchored with explicit padding, or the title's top row is clipped by the border.
            "xref": "container", "x": 0.012, "xanchor": "left",
            "yref": "container", "y": 1.0, "yanchor": "top",
            "pad": {"t": 12, "l": 2},
            "automargin": True,
        },
        "height": height,
        # `automargin` cannot shrink a margin set too small to begin with.
        "margin": margin or {"l": 12, "r": 12, "t": 44 if title else 10, "b": 12},
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": PANEL,
        "font": _TEXT,
        "showlegend": False,
        "modebar": MODEBAR,
        **extra,
    }


def line(label: str, points: list[tuple[str, float]], title: str = "", y_label: str = "") -> dict:
    """A line chart of one named series: `points` is (x label, y value), left to right."""
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
        dragmode="zoom",  # the page reports the move across a dragged x-range
        **HOVER,
    )}


def table(headers: list[str], rows: list[list], title: str = "",
          groups: list[tuple[str, int]] | None = None, note: str = "",
          subtitles: list[str] | None = None) -> dict:
    """A table from row-major `rows`: `groups` label column runs above the header, `subtitles`
    print under each column name, and `note` is a caption.
    """
    if not headers:
        raise ValueError("a table needs at least one column")
    if groups and sum(n for _, n in groups) != len(headers):
        raise ValueError("table groups must cover every column exactly once")
    if subtitles and len(subtitles) != len(headers):
        raise ValueError("a table needs one subtitle per column")
    subs = subtitles or [""] * len(headers)
    columns = [[row[i] if i < len(row) else "" for row in rows] for i in range(len(headers))]
    # Estimated pixel widths (6.6 per character plus padding), so no cell wraps past the height.
    widths = [round(max(len(h), len(s) * 0.82, *(len(str(v)) for v in col)) * 6.6 + 16)
              for h, s, col in zip(headers, subs, columns)]
    trace = go.Table(
        header={
            "values": [f"<b>{h}</b>" + (f"<br><span style='font-size:9px;color:{DIM}'>{s}</span>" if s else "")
                       for h, s in zip(headers, subs)],
            "fill_color": PANEL,
            "line_color": LINE,
            "align": "left",
            "font": {"color": AMBER, "family": MONO, "size": 11},
            "height": 40 if subtitles else 26,
        },
        cells={
            "values": columns,
            "fill_color": "rgba(0,0,0,0)",
            "line_color": LINE,
            "align": "left",
            "font": {"color": MUTED, "family": MONO, "size": 11},
            "height": 24,
        },
        columnwidth=widths,
    ).to_plotly_json()
    # Height is computed: miscount any term and the last row is cut off with no scrollbar.
    bar = 24 if groups else 0
    margin = {"l": 8, "r": 8, "t": (44 if title else 8) + bar, "b": 8}  # matches _layout's title pad
    # A bold header grows past its set height once Plotly measures the markup.
    header, row, slack = (44 if subtitles else 30), 24, 8
    height = min(margin["t"] + margin["b"] + header + row * max(len(rows), 1) + slack, 520)
    extra = _group_bar(groups, widths, bar / (height - margin["t"] - margin["b"])) if groups else {}
    if note:
        extra["meta"] = {"note": note}
    return {"data": [trace], "layout": _layout(title, height, margin, **extra)}


def _group_bar(groups: list[tuple[str, int]], widths: list[int], depth: float) -> dict:
    """Labelled boxes above the header, in paper units: `depth` is the bar's height."""
    shapes, labels, start, total = [], [], 0, sum(widths)
    for label, n in groups:
        x0, x1 = sum(widths[:start]) / total, sum(widths[:start + n]) / total
        shapes.append({"type": "rect", "xref": "paper", "yref": "paper", "x0": x0, "x1": x1,
                       "y0": 1, "y1": 1 + depth, "fillcolor": PANEL, "line": {"color": LINE, "width": 1}})
        labels.append({"xref": "paper", "yref": "paper", "x": (x0 + x1) / 2, "y": 1 + depth / 2,
                       "text": f"<b>{label}</b>", "showarrow": False, "xanchor": "center",
                       "yanchor": "middle", "font": {"color": DIM, "family": MONO, "size": 10}})
        start += n
    return {"shapes": shapes, "annotations": labels}


def zscore_panel(
    points: list[tuple[str, float]],
    entry_z: float,
    exit_z: float,
    title: str = "",
    episodes: list[dict] | None = None,
    height: int = 340,
    note: str = "",
) -> dict:
    """The spread's z-score over time, with entry/exit bands and shaded divergence episodes."""
    if len(points) < 2:
        raise ValueError("a z-score panel needs at least 2 points")
    x = [d for d, _ in points]
    trace = go.Scatter(
        x=x, y=[z for _, z in points], mode="lines", name="z",
        line={"color": AMBER, "width": 1.6}, hovertemplate="z = %{y:.2f}<extra></extra>",
    ).to_plotly_json()

    # Shapes rather than traces, so the legend stays empty and hover stays clean.
    lines = []
    for level, colour, dash in [(entry_z, DIM, "dash"), (-entry_z, DIM, "dash"),
                                (exit_z, LINE, "dot"), (-exit_z, LINE, "dot"), (0, MUTED, "solid")]:
        lines.append({"type": "line", "xref": "paper", "x0": 0, "x1": 1, "yref": "y",
                      "y0": level, "y1": level, "line": {"color": colour, "width": 1, "dash": dash}})
    for episode in episodes or []:
        lines.append({"type": "rect", "xref": "x", "yref": "paper",
                      "x0": episode["start"], "x1": episode["end"], "y0": 0, "y1": 1,
                      "fillcolor": DIM, "opacity": 0.14, "line": {"width": 0}, "layer": "below"})

    layout = _layout(title, height, xaxis=axis(), yaxis=axis("z-score"), shapes=lines, **HOVER,
                     meta={"kind": "zscore", "note": note})
    return {"data": [trace], "layout": layout}
