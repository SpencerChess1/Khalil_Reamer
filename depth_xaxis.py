
# depth_xaxis.py
# Streamlit depth-based strip-log dashboard with Depth on the x-axis (horizontal)
# Run: streamlit run depth_xaxis.py

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go


st.set_page_config(page_title="Depth Strip-Log (X-Axis)", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    depth_col = None
    for c in df.columns:
        if c.strip().lower().startswith("depth"):
            depth_col = c
            break
    if not depth_col:
        raise RuntimeError("No depth column found.")
    df[depth_col] = pd.to_numeric(df[depth_col], errors="coerce")
    df = df.dropna(subset=[depth_col]).sort_values(depth_col).reset_index(drop=True)
    return df, depth_col

folder = Path(__file__).parent
default_csv = folder / "MC518_002_NOVReq_R600_RecDepth.csv"
if not default_csv.exists():
    st.error("Place your CSV file with a 'Depth' column in the same folder as this app.")
    st.stop()

df, DEPTH = load_data(default_csv)
cols = set(df.columns)

st.title("Depth-Based Strip-Log Dashboard (Depth on X-Axis)")
st.caption(f"File: {default_csv.name} — depth column: {DEPTH}")

# --- Helper: first existing column from list
def first_existing(options):
    for c in options:
        if c in df.columns:
            return c
    return None

# --- Track definitions ---
tracks = {
    "Hookload Avg": [("Hookload Avg(klb)", "Hookload", "L", "hook")],
    "Block Position / ROP": [
        ("Block Position(ft)", "Block Position", "L", "block"),
        ("ROP Avg(fph)", "Avg ROP", "R", "rop"),
    ],
    "Surface WOB": [("WOB Avg(klb)", "Surface WOB", "L", "wob")],
    "Surface Torque": [("Torque Abs Avg(f-kp)", "Surface Torque", "L", "torque")],
    "Total Flow": [("Total Flow(gpm)", "Total Flow", "L", "flow")],
    "Annulus Pressure": [
        (first_existing(["Ann Press - EMW(ppg)", "Ann Pres", "Ann Pres - PWD"]), "Annulus Pres", "L", "annpres"),
        ("Pres Xducer 1(psig)", "Pump/Surf Pres", "R", "surfpres"),
    ],
    "Mud Density In": [("Dens Mud In Avg(ppg)", "Mud In", "L", "mud")],
    "Annulus Temp": [("Temp Annulus(degF)", "Ann Temp", "L", "temp")],
    "Surface RPM": [
        ("RPM(rpm)", "RPM", "L", "rpm"),
        ("RPM Max(rpm)", "RPM Max", "R", "rpm_max"),
        ("RPM Min(rpm)", "RPM Min", "R", "rpm_min"),
        ("RPM Surface Avg(rpm)", "RPM Surface Avg", "L", "rpm_surf"),
    ],
    "Downhole/Motor RPM": [
        (first_existing(["GP_Mean_RPM(rpm)", "GP Mean RPM"]), "GP Mean RPM", "L", "gp_mean"),
        (first_existing(["GP_Max_RPM(rpm)", "GP Max RPM"]), "GP Max RPM", "R", "gp_max"),
        (first_existing(["GP_Min_RPM(rpm)", "GP Min RPM"]), "GP Min RPM", "R", "gp_min"),
    ],
    "StickSlip Indicator": [("StickSlip Ind(NONE)", "StickSlip", "L", "stick")],
    "DOC / Toolface / Inclination": [
        ("DCBA(f-p)", "DOC Bend Mom A", "L", "doc_a"),
        ("DCTA(f-p)", "DOC Avg Bend Mom", "L", "doc_ta"),
        ("DCTX(f-p)", "DOC Avg Torque", "L", "doc_tx"),
        ("DCD(deg)", "DOC Bend Mom Dir", "R", "doc_dir"),
    ],
    "Vibration Severity (M5)": [
        ("AVGX-M5(g)", "Avg X", "L", "vib_avgx"),
        ("AVGY-M5(g)", "Avg Y", "L", "vib_avgy"),
        ("AVGZ-M5(g)", "Avg Z", "L", "vib_avgz"),
        ("PEAKX-M5(g)", "Peak X", "R", "vib_peakx"),
        ("PEAKY-M5(g)", "Peak Y", "R", "vib_peaky"),
        ("PEAKZ-M5(g)", "Peak Z", "R", "vib_peakz"),
    ],
}

# NEW: Combined overlay track user requested: Annulus Pressure + StickSlip + DOC Mom A + ROP
overlay_track_key = "Overlay: Ann Pres + StickSlip + DOC Mom A + ROP"
overlay_specs = [
    (first_existing(["Ann Press - EMW(ppg)", "Ann Pres", "Ann Pres - PWD"]), "Annulus Pres", "L", "annpres"),
    ("StickSlip Ind(NONE)", "StickSlip", "R", "stick"),
    ("DCBA(f-p)", "DOC Mom A", "L", "doc_a"),
    ("ROP Avg(fph)", "Avg ROP", "R", "rop"),
]
# Keep only those that exist
overlay_specs = [(c,n,a,k) for (c,n,a,k) in overlay_specs if c and (c in df.columns)]
if overlay_specs:
    tracks[overlay_track_key] = overlay_specs

# Only keep series whose column exists
for k in list(tracks.keys()):
    tracks[k] = [s for s in tracks[k] if s[0] in cols]
    if not tracks[k]:
        del tracks[k]

# Color palette
palette = {
    "hook":"#1f77b4", "block":"#ff7f0e", "rop":"#2ca02c",
    "wob":"#9467bd", "torque":"#8c564b", "flow":"#e377c2",
    "annpres":"#17becf", "surfpres":"#bcbd22", "mud":"#7f7f7f",
    "temp":"#d62728", "rpm":"#1f77b4", "rpm_max":"#ff7f0e","rpm_min":"#2ca02c","rpm_surf":"#8c564b",
    "gp_mean":"#9467bd","gp_max":"#8c564b","gp_min":"#e377c2",
    "stick":"#17becf",
    "doc_a":"#1f77b4","doc_ta":"#ff7f0e","doc_tx":"#2ca02c","doc_dir":"#d62728",
    "vib_avgx":"#1f77b4","vib_avgy":"#ff7f0e","vib_avgz":"#2ca02c",
    "vib_peakx":"#d62728","vib_peaky":"#9467bd","vib_peakz":"#8c564b",
}

# --- Controls
st.sidebar.header("Filters")
dmin = float(df[DEPTH].min())
dmax = float(df[DEPTH].max())
invert_depth = st.sidebar.checkbox("Invert depth axis (increasing leftward)", value=False)
depth_range = st.sidebar.slider("Depth range (ft)", float(dmin), float(dmax), (float(dmin), float(dmax)))
selected_tracks = st.sidebar.multiselect("Tracks", list(tracks.keys()), default=list(tracks.keys()))

mask = (df[DEPTH] >= depth_range[0]) & (df[DEPTH] <= depth_range[1])
dfv = df[mask]

def build_track_fig(title, specs, df_section):
    fig = go.Figure()
    for col, disp, axis, color in specs:
        if col not in df_section.columns: continue
        fig.add_trace(go.Scatter(
            x=df_section[DEPTH], y=df_section[col],
            mode="lines", name=disp, line=dict(color=palette.get(color))
        ))
    fig.update_layout(
        height=360, margin=dict(l=50,r=50,t=50,b=40),
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        xaxis=dict(title=DEPTH, autorange="reversed" if invert_depth else True, showgrid=True),
        yaxis=dict(title="Value", showgrid=True)
    )
    return fig

for t in selected_tracks:
    fig = build_track_fig(t, tracks[t], dfv)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Data preview"):
    st.dataframe(dfv.head(200))
