import streamlit as st
import json
import os
import tempfile
from datetime import date
from google import genai
from google.oauth2 import service_account

from layer1_perception import run_session
from layer2_aggregation import run_aggregation
from layer3_reasoner    import run_layer3
from taxonomy           import STAGE_TAXONOMY

st.set_page_config(page_title="Progress Proof", layout="wide")

STATUS_STYLE = {
    "confirmed_visible": ("background:#EAF3DE;border:0.5px solid #3B6D11", "color:#27500A", "color:#3B6D11"),
    "not_confirmed":     ("background:#FAEEDA;border:0.5px solid #854F0B", "color:#633806", "color:#854F0B"),
    "not_visible":       ("background:#F1EFE8;border:0.5px solid #B4B2A9", "color:#5F5E5A", "color:#888780"),
    "needs_review":      ("background:#FCEBEB;border:0.5px solid #A32D2D", "color:#791F1F", "color:#A32D2D"),
}

STAGE_STYLE = {
    1: ("background:#FAECE7;border:1px solid #F0997B", "color:#993C1D", "color:#712B13"),
    2: ("background:#EAF3DE;border:1px solid #97C459", "color:#3B6D11", "color:#27500A"),
    3: ("background:#EEEDFE;border:1px solid #AFA9EC", "color:#534AB7", "color:#3C3489"),
    4: ("background:#E6F1FB;border:1px solid #85B7EB", "color:#185FA5", "color:#0C447C"),
    5: ("background:#FAEEDA;border:1px solid #EF9F27", "color:#854F0B", "color:#633806"),
}


@st.cache_resource
def get_client():
    key_dict = dict(st.secrets["GCP_SERVICE_ACCOUNT"])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(key_dict, f)
        tmp_path = f.name
    credentials = service_account.Credentials.from_service_account_file(
        tmp_path,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    os.unlink(tmp_path)
    return genai.Client(
        vertexai=True,
        project="intellus-build-dev",
        location="us-central1",
        credentials=credentials,
    )


def render_evidence_grid(evidence_pool):
    cols = st.columns(5)
    for idx, (stage_num, info) in enumerate(STAGE_TAXONOMY.items()):
        box_s, num_s, name_s = STAGE_STYLE[stage_num]
        with cols[idx]:
            st.markdown(
                f'<div style="padding:8px 6px;border-radius:8px;text-align:center;'
                f'margin-bottom:8px;{box_s}">'
                f'<div style="font-size:9px;font-weight:500;margin-bottom:2px;{num_s}">Stage {stage_num}</div>'
                f'<div style="font-size:11px;font-weight:500;{name_s}">{info["name"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            for el in info["elements"]:
                ev     = evidence_pool.get(el, {})
                status = ev.get("status", "not_visible")
                conf   = ev.get("confidence", 0.0)
                flag   = ev.get("flag") or ""
                box_s2, name_s2, meta_s2 = STATUS_STYLE.get(status, STATUS_STYLE["not_visible"])
                conf_str = f"{int(conf*100)}%" if conf > 0 else ""
                flag_str = (
                    f'<div style="font-size:9px;{meta_s2};opacity:.8;margin-top:2px">'
                    f'{flag.replace("_"," ")}</div>'
                ) if flag else ""
                st.markdown(
                    f'<div style="padding:6px 8px;border-radius:8px;margin-bottom:5px;{box_s2}">'
                    f'<div style="font-size:11px;font-weight:500;line-height:1.3;{name_s2}">'
                    f'{el.replace("_"," ")}</div>'
                    f'<div style="font-size:10px;margin-top:2px;{meta_s2}">'
                    f'{conf_str} {status.replace("_"," ")}</div>'
                    f'{flag_str}'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ── Page ──────────────────────────────────────────────────────────────────────

st.title("Progress Proof")
st.caption("Construction stage detection · Intellus Build · HCAD Harris County")

with st.sidebar:
    st.header("Session settings")
    assessment_date = st.date_input(
        "Assessment date (HCAD anchor)",
        value=date(2026, 1, 1),
        help="Evidence dated after this date is excluded from the appeal analysis.",
    )
    user_date = st.text_input(
        "Photo date override (YYYY-MM-DD)",
        value="",
        help="Applied to photos with no EXIF date. Leave blank to mark as unknown.",
    )
    st.divider()
    st.caption("Optional: add non-photo evidence sources.")
    inspection_file = st.file_uploader("Inspection records CSV", type=["csv"])
    draw_file       = st.file_uploader("Draw schedule CSV",      type=["csv"])
    st.divider()
    st.caption("Inspection records CSV columns: element, status (pass/fail), date (YYYY-MM-DD), inspector")
    st.caption("Draw schedule CSV columns: stage (1-5), approved (yes/no), date (YYYY-MM-DD), amount")

uploaded_photos = st.file_uploader(
    "Upload construction site photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

run_button = st.button(
    "Run analysis",
    type="primary",
    disabled=not uploaded_photos,
)

if run_button and uploaded_photos:

    client      = get_client()
    image_files = [(f.name, f.read()) for f in uploaded_photos]

    inspection_records = []
    draw_schedule      = []

    if inspection_file:
        import csv, io
        reader = csv.DictReader(io.StringIO(inspection_file.read().decode()))
        for row in reader:
            inspection_records.append({
                "element":   row["element"].strip(),
                "status":    row["status"].strip().lower(),
                "date":      row["date"].strip(),
                "inspector": row.get("inspector", "").strip(),
            })

    if draw_file:
        import csv, io
        reader = csv.DictReader(io.StringIO(draw_file.read().decode()))
        for row in reader:
            draw_schedule.append({
                "stage":    int(row["stage"].strip()),
                "approved": row["approved"].strip().lower() == "yes",
                "date":     row["date"].strip(),
            })

    with st.status("Running analysis...", expanded=True) as status_box:

        st.write(f"Layer 1 — processing {len(image_files)} photos...")
        session = run_session(
            image_files=image_files,
            client=client,
            user_date=user_date.strip() or None,
        )
        st.write(f"  {session['successful']}/{session['total_photos']} photos processed.")
        if session["failed"]:
            for f in session["failed"]:
                st.write(f"  Failed: {f['filename']} — {f['reason']}")

        st.write("Layer 2 — aggregating evidence...")
        layer2 = run_aggregation(
            session=session,
            inspection_records=inspection_records,
            draw_schedule=draw_schedule,
            assessment_date=str(assessment_date),
        )
        confirmed_count = layer2["summary"]["confirmed_count"]
        review_count    = layer2["summary"]["needs_review"]
        st.write(f"  {confirmed_count} elements confirmed, {review_count} flagged for review.")

        st.write("Layer 3 — stage reasoning...")
        layer3 = run_layer3(layer2_output=layer2, client=client)

        status_box.update(label="Analysis complete", state="complete")

    conclusion = layer3["conclusion"]
    c          = conclusion

    st.divider()

    # ── Summary metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Photos",          f"{session['successful']}/{session['total_photos']}")
    col2.metric("Assessment date", str(assessment_date))
    col3.metric("Stage",           f"{c['stage']} — {c['stage_name']}")
    col4.metric("Confidence",      f"{int(c['confidence']*100)}%")
    col5.metric("Human review",    "Required" if c["needs_human_review"] else "Not required")

    # ── Review alert ──────────────────────────────────────────────────────────
    if c["needs_human_review"]:
        st.error("Human review required before using this output in a tax appeal.")
        for r in c["review_reasons"]:
            st.caption(f"· {r}")
    else:
        st.success("No human review required. Conclusion is ready for appeal package.")

    # ── Reasoning ─────────────────────────────────────────────────────────────
    st.subheader("Stage reasoning")
    st.write(c["reasoning"])

    if c["objections"]:
        with st.expander("Challenger objections"):
            for obj in c["objections"]:
                st.caption(f"· {obj}")

    if c["blocking_next_stage"]:
        st.caption(
            f"Blocking Stage {c['stage'] + 1}: "
            f"{', '.join(c['blocking_next_stage'])}"
        )

    if c["coverage_gaps"]:
        st.caption(
            f"Coverage gaps — better photos needed for: "
            f"{', '.join(c['coverage_gaps'])}"
        )

    # ── Evidence grid ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Evidence pool")
    st.caption("Green = confirmed · Amber = not confirmed · Gray = not visible · Red = needs review")
    render_evidence_grid(layer2["evidence_pool"])

    # ── Failed photos ─────────────────────────────────────────────────────────
    if session["failed"]:
        st.divider()
        st.subheader("Failed photos")
        for f in session["failed"]:
            st.warning(f"{f['filename']}: {f['reason']}")

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download Layer 2 JSON",
            data=json.dumps(layer2, indent=2),
            file_name=f"{session['session_id']}_layer2.json",
            mime="application/json",
        )
    with col_b:
        st.download_button(
            "Download Layer 3 JSON",
            data=json.dumps(layer3, indent=2),
            file_name=f"{session['session_id']}_layer3.json",
            mime="application/json",
        )
