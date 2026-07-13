#!/usr/bin/env python3
"""Generate an interactive HTML report from one validated CAFE test pixel."""

from __future__ import annotations

from pathlib import Path

import plotly.io as pio

from trident.models.cafe import cafe_profile, integrate_npp_profile
from trident.visualization.cafe_profiles import build_cafe_diagnostics_figures


def main() -> None:
    profile = cafe_profile(
        40.0, 0.15, 35.0, 30.0, 180,
        0.010, 0.005, 0.0015, 1.0, 22.0,
    )
    reconstructed = integrate_npp_profile(
        profile.depth_m,
        profile.npp_z,
        delz_m=profile.delz_m,
    )
    figures = build_cafe_diagnostics_figures(profile)
    output = Path.home() / "Downloads" / "TRIDENT_CAFE_profile_check.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    sections = [
        "<html><head><meta charset='utf-8'><title>TRIDENT CAFE Profile Check</title></head><body>",
        "<h1>TRIDENT CAFE Profile Check</h1>",
        f"<p>Integrated NPP: {profile.npp:.12f} mg C m⁻² day⁻¹<br>",
        f"Reintegrated NPP(z): {reconstructed:.12f} mg C m⁻² day⁻¹<br>",
        f"Absolute difference: {abs(profile.npp - reconstructed):.3e}</p>",
    ]
    for index, (name, fig) in enumerate(figures.items()):
        sections.append(f"<h2>{name.replace('_', ' ').title()}</h2>")
        sections.append(
            pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if index == 0 else False,
            )
        )
    sections.append("</body></html>")
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"CAFE profile valid: {profile.valid}")
    print(f"Integrated NPP: {profile.npp:.12f}")
    print(f"Reintegrated NPP(z): {reconstructed:.12f}")
    print(f"Absolute difference: {abs(profile.npp - reconstructed):.3e}")
    print(f"Interactive report: {output}")


if __name__ == "__main__":
    main()
