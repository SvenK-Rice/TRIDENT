from pathlib import Path

repo = Path(__file__).resolve().parents[1]
workbench = repo / "src/trident/app/workbench.py"
text = workbench.read_text(encoding="utf-8")

import_line = (
    "from trident.viz.maps import make_geo_heatmap as make_geo_heatmap_v2\n"
)
if import_line not in text:
    anchor = "from trident.grids.align import mask_feasible\n"
    if anchor not in text:
        raise SystemExit(
            "Could not find the expected import anchor in workbench.py."
        )
    text = text.replace(anchor, anchor + import_line, 1)

old_call = (
    'st.plotly_chart(make_geo_heatmap(ds, var, '
    'f"{entry.label} — {tag}"), use_container_width=True)'
)
new_call = (
    'st.plotly_chart(make_geo_heatmap_v2(ds, var, '
    'f"{entry.label} — {tag}"), use_container_width=True)'
)

if old_call in text:
    text = text.replace(old_call, new_call, 1)
elif "make_geo_heatmap_v2(ds, var" not in text:
    raise SystemExit(
        "Could not find the Explore map call in workbench.py. "
        "No file was changed."
    )

workbench.write_text(text, encoding="utf-8")
print(f"Updated {workbench}")
