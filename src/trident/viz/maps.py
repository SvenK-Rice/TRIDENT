from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import plotly.graph_objects as go
import xarray as xr

from trident.grids.align import mask_feasible


DisplayMode = Literal["raw", "nearest_display"]


_WEST_COAST = [
    (-124.8, 48.7), (-124.3, 47.9), (-124.1, 47.1), (-123.8, 46.3),
    (-124.0, 45.6), (-123.9, 45.0), (-123.8, 44.5), (-124.1, 43.9),
    (-124.3, 43.2), (-124.4, 42.5), (-124.2, 41.9), (-124.1, 41.3),
    (-124.0, 40.7), (-123.8, 40.0), (-123.5, 39.4), (-123.1, 38.9),
    (-122.7, 38.3), (-122.5, 37.8), (-122.3, 37.4), (-122.0, 37.0),
    (-121.8, 36.6), (-121.6, 36.0), (-121.2, 35.5), (-120.8, 35.1),
    (-120.4, 34.7), (-119.9, 34.4), (-119.5, 34.1), (-119.0, 33.8),
    (-118.5, 33.5), (-118.0, 33.0), (-117.6, 32.7), (-117.1, 32.5),
    (-116.6, 32.2), (-116.0, 31.9), (-115.4, 31.5), (-114.8, 31.0),
    (-114.5, 30.5), (-114.2, 29.9), (-114.0, 29.4), (-113.7, 28.9),
    (-113.4, 28.4), (-113.1, 27.9),
]


@dataclass(frozen=True)
class Bounds:
    west: float
    south: float
    east: float
    north: float


def _fmt_lon(value: float) -> str:
    return f"{abs(float(value)):g}°" + ("W" if value < 0 else "E")


def _fmt_lat(value: float) -> str:
    return f"{abs(float(value)):g}°" + ("S" if value < 0 else "N")


def _outcode(x: float, y: float, bounds: Bounds) -> int:
    code = 0
    if x < bounds.west:
        code |= 1
    elif x > bounds.east:
        code |= 2
    if y < bounds.south:
        code |= 4
    elif y > bounds.north:
        code |= 8
    return code


def _clip_segment(
    p0: tuple[float, float],
    p1: tuple[float, float],
    bounds: Bounds,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip one line segment to a rectangular extent."""
    x0, y0 = map(float, p0)
    x1, y1 = map(float, p1)
    code0 = _outcode(x0, y0, bounds)
    code1 = _outcode(x1, y1, bounds)

    while True:
        if not (code0 | code1):
            return (x0, y0), (x1, y1)
        if code0 & code1:
            return None

        code = code0 or code1

        if code & 8:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (bounds.north - y0) / (y1 - y0)
            y = bounds.north
        elif code & 4:
            if y1 == y0:
                return None
            x = x0 + (x1 - x0) * (bounds.south - y0) / (y1 - y0)
            y = bounds.south
        elif code & 2:
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (bounds.east - x0) / (x1 - x0)
            x = bounds.east
        else:
            if x1 == x0:
                return None
            y = y0 + (y1 - y0) * (bounds.west - x0) / (x1 - x0)
            x = bounds.west

        if code == code0:
            x0, y0 = x, y
            code0 = _outcode(x0, y0, bounds)
        else:
            x1, y1 = x, y
            code1 = _outcode(x1, y1, bounds)


def _clip_polyline(
    points: Iterable[tuple[float, float]],
    bounds: Bounds,
) -> list[tuple[float, float]]:
    """Clip a polyline to bounds and return its longest connected section."""
    pts = list(points)
    sections: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for start, end in zip(pts[:-1], pts[1:]):
        clipped = _clip_segment(start, end, bounds)

        if clipped is None:
            if current:
                sections.append(current)
                current = []
            continue

        a, b = clipped
        if not current:
            current = [a, b]
        elif np.allclose(current[-1], a):
            current.append(b)
        else:
            sections.append(current)
            current = [a, b]

    if current:
        sections.append(current)

    if not sections:
        return []

    return max(sections, key=len)


def nearest_display_fill(
    values: np.ndarray,
    *,
    max_distance_cells: float | None = None,
    chunk_size: int = 2048,
) -> np.ndarray:
    """Fill NaN cells from nearest calculated pixels for display only."""
    source = np.asarray(values, dtype=float)
    result = source.copy()

    finite_positions = np.argwhere(np.isfinite(source))
    missing_positions = np.argwhere(~np.isfinite(source))

    if finite_positions.size == 0 or missing_positions.size == 0:
        return result

    finite_values = source[
        finite_positions[:, 0],
        finite_positions[:, 1],
    ]

    for start in range(0, len(missing_positions), chunk_size):
        target = missing_positions[start : start + chunk_size]
        distance2 = (
            (target[:, None, 0] - finite_positions[None, :, 0]) ** 2
            + (target[:, None, 1] - finite_positions[None, :, 1]) ** 2
        )
        nearest_index = np.argmin(distance2, axis=1)
        nearest_distance = np.sqrt(
            distance2[np.arange(len(target)), nearest_index]
        )

        if max_distance_cells is None:
            accepted = np.ones(len(target), dtype=bool)
        else:
            accepted = nearest_distance <= float(max_distance_cells)

        accepted_target = target[accepted]
        result[
            accepted_target[:, 0],
            accepted_target[:, 1],
        ] = finite_values[nearest_index[accepted]]

    return result


def _limits(variable: str, values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None

    name = variable.lower()

    if name == "difference":
        limit = max(1.0, float(np.nanpercentile(np.abs(finite), 99)))
        return -limit, limit

    if "npp" in name or "cafe" in name:
        return 0.0, max(
            500.0,
            min(5000.0, float(np.nanpercentile(finite, 99))),
        )

    return (
        float(np.nanpercentile(finite, 1)),
        float(np.nanpercentile(finite, 99)),
    )


def _add_land(fig: go.Figure, bounds: Bounds) -> None:
    if not (
        bounds.west < -113
        and bounds.east > -132.5
        and bounds.south < 49
        and bounds.north > 24
    ):
        return

    coast = _clip_polyline(_WEST_COAST, bounds)
    if len(coast) < 2:
        return

    coast_x = [x for x, _ in coast]
    coast_y = [y for _, y in coast]

    polygon_x = coast_x + [bounds.east, bounds.east, coast_x[0]]
    polygon_y = coast_y + [coast_y[-1], coast_y[0], coast_y[0]]

    fig.add_trace(
        go.Scatter(
            x=polygon_x,
            y=polygon_y,
            mode="lines",
            fill="toself",
            fillcolor="rgba(166,124,82,0.96)",
            line={"width": 0.5, "color": "rgba(166,124,82,0.96)"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=coast_x,
            y=coast_y,
            mode="lines",
            line={"color": "black", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def make_geo_heatmap(
    dataset: xr.Dataset,
    variable: str,
    title: str | None = None,
    *,
    display_mode: DisplayMode = "raw",
    display_fill_distance: float | None = None,
) -> go.Figure:
    """Render a geographic map from a 2-D variable on 1-D lat/lon axes."""
    raw = mask_feasible(
        variable,
        np.asarray(dataset[variable].values),
    ).astype(float)

    latitude = np.asarray(dataset["lat"].values, dtype=float)
    longitude = np.asarray(dataset["lon"].values, dtype=float)

    if raw.ndim != 2 or latitude.ndim != 1 or longitude.ndim != 1:
        raise ValueError(
            "Map variables must be 2-D with 1-D lat/lon coordinates."
        )

    if latitude[0] > latitude[-1]:
        latitude = latitude[::-1]
        raw = raw[::-1, :]
    if longitude[0] > longitude[-1]:
        longitude = longitude[::-1]
        raw = raw[:, ::-1]

    if display_mode == "nearest_display":
        shown = nearest_display_fill(
            raw,
            max_distance_cells=display_fill_distance,
        )
        subtitle = "display-interpolated; statistics use raw pixels"
    else:
        shown = raw
        subtitle = "raw calculated pixels"

    bounds = Bounds(
        west=float(np.nanmin(longitude)),
        south=float(np.nanmin(latitude)),
        east=float(np.nanmax(longitude)),
        north=float(np.nanmax(latitude)),
    )
    zmin, zmax = _limits(variable, shown)
    colorscale = "RdBu_r" if variable.lower() == "difference" else "Viridis"

    fig = go.Figure(
        go.Heatmap(
            x=longitude,
            y=latitude,
            z=shown,
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            connectgaps=False,
            colorbar={
                "title": (
                    "mg C m⁻² d⁻¹"
                    if (
                        "npp" in variable.lower()
                        or variable.lower() == "difference"
                    )
                    else variable
                )
            },
            customdata=raw,
            hovertemplate=(
                "Longitude: %{x:.3f}°<br>"
                "Latitude: %{y:.3f}°<br>"
                "Displayed: %{z:.4g}<br>"
                "Raw: %{customdata:.4g}<extra></extra>"
            ),
        )
    )

    _add_land(fig, bounds)

    xticks = np.linspace(bounds.west, bounds.east, 7)
    yticks = np.linspace(bounds.south, bounds.north, 7)

    fig.update_xaxes(
        title="Longitude",
        range=[bounds.west, bounds.east],
        tickvals=xticks,
        ticktext=[_fmt_lon(x) for x in xticks],
    )
    fig.update_yaxes(
        title="Latitude",
        range=[bounds.south, bounds.north],
        tickvals=yticks,
        ticktext=[_fmt_lat(y) for y in yticks],
        scaleanchor="x",
        scaleratio=1,
    )
    fig.update_layout(
        title=f"{title or variable}<br><sup>{subtitle}</sup>",
        height=650,
        margin={"l": 10, "r": 10, "t": 70, "b": 10},
        plot_bgcolor="white",
    )

    return fig
