"""
MediScan AI — Streamlit UI
Upload blood reports, get AI analysis, ask follow-up questions.
"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MediScan AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ─────────────────────────────────────
# Flat cards, muted category colors (not severity colors) for the ML risk
# panels, and soft/light disclaimer banners instead of solid loud ones —
# matches the dashboard mockup design.
st.markdown("""
<style>
    .metric-card {
        background:#f8f9fa; border-radius:12px; padding:16px 18px;
        border:1px solid #eee;
    }
    .metric-label { font-size:13px; color:#6b7280; margin:0 0 4px; }
    .metric-value { font-size:24px; font-weight:600; margin:0; color:#111; }

    .risk-card {
        border-radius:12px; padding:14px 16px; height:100%;
    }
    .risk-metabolic  { background:#FAECE7; color:#4A1B0C; }
    .risk-hematology { background:#F1EFE8; color:#2C2C2A; }
    .risk-liver      { background:#F1EFE8; color:#2C2C2A; }
    .risk-renal      { background:#E1F5EE; color:#04342C; }
    .risk-title { font-size:13px; font-weight:600; margin:0 0 8px; }
    .risk-bar-track { height:6px; background:rgba(0,0,0,0.08); border-radius:3px; overflow:hidden; margin-bottom:6px; }
    .risk-bar-fill { height:100%; background:currentColor; }
    .risk-caption { font-size:12px; opacity:0.85; }

    .disclaimer-safe      { background:#EAF3DE; color:#173404; border-radius:10px; padding:12px 16px; }
    .disclaimer-warning   { background:#FAEEDA; color:#412402; border-radius:10px; padding:12px 16px; }
    .disclaimer-critical  { background:#FAECE7; color:#4A1B0C; border-radius:10px; padding:12px 16px; }
    .disclaimer-emergency { background:#FCEBEB; color:#501313; border-radius:10px; padding:12px 16px; }

    .status-normal     { color:#3B6D11; font-weight:600; }
    .status-borderline  { color:#854F0B; font-weight:600; }
    .status-critical   { color:#993C1D; font-weight:600; }
    .status-panic      { color:#A32D2D; font-weight:700; }
    .status-unknown    { color:#888780; font-weight:500; }

    .explanation-card {
        background:#f8f9fa; border-radius:12px; padding:16px 18px; border:1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────
for key, default in {
    "token": None, "username": None, "role": "user",
    "page": "upload", "current_report": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def get_headers():
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_post(endpoint, **kwargs):
    try:
        r = requests.post(f"{API_URL}{endpoint}", headers=get_headers(), timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Is the server running?")
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ {e.response.status_code}: {e.response.json().get('detail', e.response.text)}")
    return None


def api_get(endpoint):
    try:
        r = requests.get(f"{API_URL}{endpoint}", headers=get_headers(), timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"❌ {e}")
    return None


# ── Small render helpers ──────────────────────────────

PANEL_META = {
    "metabolic":  {"label": "Metabolic",  "css": "risk-metabolic",  "icon": "🩸"},
    "hematology": {"label": "Hematology", "css": "risk-hematology", "icon": "🧫"},
    "liver":      {"label": "Liver",      "css": "risk-liver",      "icon": "🫁"},
    "renal":      {"label": "Renal",      "css": "risk-renal",      "icon": "💧"},
}

STATUS_ICON = {"panic": "🚨", "critical": "🔴", "borderline": "🟡", "normal": "🟢", "unknown": "⚪"}


def render_metric_row(data):
    extracted = data.get("extracted_values", {})
    results = data.get("analysis_results", [])
    abnormal_count = sum(1 for r in results if r.get("severity") not in ("normal", "unknown"))
    disc_level = data.get("disclaimer", {}).get("level", "safe")
    extraction_method = data.get("extraction_method", "regex")
    method_label = {"hybrid": "Hybrid (regex + LLM)", "regex": "Regex only", "llm": "LLM only"}.get(
        extraction_method, extraction_method.title())

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value in [
        (c1, "Values extracted", len(extracted)),
        (c2, "Abnormal", abnormal_count),
        (c3, "Severity", disc_level.title()),
        (c4, "Extraction method", method_label),
    ]:
        col.markdown(
            f'<div class="metric-card"><p class="metric-label">{label}</p>'
            f'<p class="metric-value">{value}</p></div>',
            unsafe_allow_html=True,
        )


def render_risk_panels(ml_risk_predictions: dict):
    st.markdown("**Specialist risk panels**")
    st.caption("A secondary AI-derived risk signal alongside the rule-based severity above — not a diagnosis.")
    cols = st.columns(4)
    for col, (panel_key, meta) in zip(cols, PANEL_META.items()):
        panel = ml_risk_predictions.get(panel_key, {})
        prob = panel.get("risk_probability")
        eligible = panel.get("eligible", False)

        if not eligible or prob is None:
            body = (
                f'<p class="risk-title">{meta["icon"]} {meta["label"]}</p>'
                f'<p class="risk-caption">Not enough matching tests present '
                f'({panel.get("covered_features", 0)}/{panel.get("total_features", "?")})</p>'
            )
        else:
            pct = round(prob * 100)
            label = panel.get("risk_label", "").replace("_", " ").title()
            body = (
                f'<p class="risk-title">{meta["icon"]} {meta["label"]}</p>'
                f'<div class="risk-bar-track"><div class="risk-bar-fill" style="width:{pct}%;"></div></div>'
                f'<p class="risk-caption">{label} · {prob:.2f}</p>'
            )

        col.markdown(f'<div class="risk-card {meta["css"]}">{body}</div>', unsafe_allow_html=True)


# ── Login / Signup ────────────────────────────────────
def show_login():
    st.title("🩺 MediScan AI")
    st.subheader("AI-powered Blood Report Analysis")
    st.divider()

    tab_login, tab_signup = st.tabs(["Login", "Sign Up"])

    with tab_login:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login", use_container_width=True, type="primary"):
                try:
                    r = requests.post(f"{API_URL}/auth/token",
                                      data={"username": username, "password": password},
                                      timeout=10)
                    if r.status_code == 200:
                        data = r.json()
                        st.session_state.token = data["access_token"]
                        st.session_state.username = username
                        st.session_state.role = data.get("role", "user")
                        st.rerun()
                    else:
                        st.error(r.json().get("detail", "Invalid credentials."))
                except Exception as e:
                    st.error(f"Login failed: {e}")

    with tab_signup:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            new_user = st.text_input("Username", key="signup_user")
            new_email = st.text_input("Email", key="signup_email")
            new_pass = st.text_input("Password", type="password", key="signup_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if new_pass != confirm_pass:
                    st.error("Passwords don't match.")
                elif len(new_pass) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        r = requests.post(f"{API_URL}/auth/signup",
                                          json={"username": new_user, "email": new_email,
                                                "password": new_pass}, timeout=10)
                        if r.status_code == 200:
                            data = r.json()
                            st.session_state.token = data["access_token"]
                            st.session_state.username = new_user
                            st.session_state.role = data.get("role", "user")
                            st.success("Account created! Logging you in...")
                            st.rerun()
                        else:
                            st.error(r.json().get("detail", "Signup failed."))
                    except Exception as e:
                        st.error(f"Signup failed: {e}")


# ── Main App ──────────────────────────────────────────
def show_app():
    with st.sidebar:
        st.title("🩺 MediScan AI")
        st.caption(f"👤 {st.session_state.username}")
        st.divider()

        if st.button("📤 Upload Report", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()

        if st.button("📋 My Reports", use_container_width=True):
            st.session_state.page = "history"
            st.rerun()

        if st.session_state.role == "admin":
            if st.button("📊 Analytics", use_container_width=True):
                st.session_state.page = "analytics"
                st.rerun()

        st.divider()
        st.caption("⚠️ This app is for educational purposes only. Always consult a doctor.")

        if st.button("🚪 Logout"):
            for k in ["token", "username", "role", "current_report"]:
                st.session_state[k] = None
            st.session_state.page = "upload"
            st.rerun()

    if st.session_state.page == "upload":
        show_upload()
    elif st.session_state.page == "results":
        show_results()
    elif st.session_state.page == "history":
        show_history()
    elif st.session_state.page == "analytics":
        show_analytics()


# ── Upload Page ───────────────────────────────────────
def show_upload():
    st.title("📤 Upload Blood Report")
    st.markdown("Upload your PDF blood report and get an AI-powered plain-English explanation.")
    st.warning("⚠️ MediScan AI is educational only. Always consult a qualified doctor.")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        pdf_file = st.file_uploader("Choose your blood report PDF", type=["pdf"])
    with col2:
        gender = st.selectbox("Your gender (for accurate reference ranges)",
                              ["male", "female"], index=0)

    if pdf_file and st.button("🔬 Analyze Report", type="primary", use_container_width=True):
        with st.spinner("Extracting and analyzing your report... this may take 30-60 seconds."):
            try:
                r = requests.post(
                    f"{API_URL}/report/upload",
                    headers=get_headers(),
                    files={"file": (pdf_file.name, pdf_file.getvalue(), "application/pdf")},
                    params={"gender": gender},
                    timeout=120,
                )
                if r.status_code == 200:
                    st.session_state.current_report = r.json()
                    st.session_state.page = "results"
                    st.rerun()
                else:
                    st.error(f"Analysis failed: {r.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error: {e}")


# ── Results Page ──────────────────────────────────────
def show_results():
    data = st.session_state.current_report
    if not data:
        st.session_state.page = "upload"
        st.rerun()

    st.title("🔬 Analysis Results")

    # Disclaimer banner — soft/light, not solid loud color
    disc = data.get("disclaimer", {})
    level = disc.get("level", "safe")
    disc_class = f"disclaimer-{level}" if level in ("safe", "warning", "critical", "emergency") else "disclaimer-warning"
    st.markdown(f'<div class="{disc_class}">{disc.get("message", "")}</div>', unsafe_allow_html=True)
    st.markdown("")

    # Top metric row
    render_metric_row(data)
    st.markdown("")

    # Specialist ML risk panels
    ml_predictions = data.get("ml_risk_predictions", {})
    if ml_predictions:
        render_risk_panels(ml_predictions)
        st.divider()

    # Specialists (rule-based recommendation)
    specialists = disc.get("specialists", [])
    if specialists:
        st.subheader("👨‍⚕️ Recommended Specialists")
        cols = st.columns(len(specialists))
        for i, s in enumerate(specialists):
            cols[i].info(f"**{s}**")
        st.divider()

    # Results table
    st.subheader("📊 Your Lab Values")
    results = data.get("analysis_results", [])

    col1, col2, col3, col4, col5 = st.columns([3, 1.5, 2.5, 1.5, 2])
    col1.markdown("**Test**")
    col2.markdown("**Your Value**")
    col3.markdown("**Normal Range**")
    col4.markdown("**Unit**")
    col5.markdown("**Status**")
    st.markdown("---")

    for r in results:
        nr = r.get("normal_range", {})
        low = nr.get("min", "")
        high = nr.get("max", "")
        range_str = f"{low} – {high}" if low != "" and str(high) != "9999" else "—"
        sev = r.get("severity", "normal")
        icon = STATUS_ICON.get(sev, "⚪")

        c1, c2, c3, c4, c5 = st.columns([3, 1.5, 2.5, 1.5, 2])
        c1.write(r["test_name"].title())
        c2.write(f"**{r['value']}**")
        c3.write(range_str)
        c4.write(r.get("unit", ""))
        c5.markdown(f'<span class="status-{sev}">{icon} {r.get("status", "")}</span>', unsafe_allow_html=True)

    st.divider()

    # AI Explanation
    st.subheader("🤖 AI Explanation")
    st.markdown(
        f'<div class="explanation-card">{data.get("ai_explanation", "No explanation available.")}</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Follow-up Q&A
    st.subheader("💬 Ask a Follow-up Question")
    report_id = data.get("report_id")
    question = st.text_input("Ask about your results...",
                              placeholder="e.g. What should I eat to improve my haemoglobin?")

    if question and st.button("Ask", type="primary"):
        with st.spinner("Thinking..."):
            result = api_post("/report/ask", json={"question": question, "report_id": report_id})
            if result:
                st.info(result.get("answer", ""))

    st.divider()

    # Download
    col1, col2 = st.columns(2)
    with col1:
        if report_id and st.button("📥 Download PDF Report", use_container_width=True):
            try:
                r = requests.get(f"{API_URL}/report/{report_id}/download",
                                 headers=get_headers(), timeout=30)
                if r.status_code == 200:
                    st.download_button(
                        "💾 Save PDF", data=r.content,
                        file_name=f"mediscan_report_{report_id}.pdf",
                        mime="application/pdf", use_container_width=True,
                    )
                else:
                    st.error("PDF generation failed.")
            except Exception as e:
                st.error(f"Download error: {e}")
    with col2:
        if st.button("📤 Upload Another Report", use_container_width=True):
            st.session_state.current_report = None
            st.session_state.page = "upload"
            st.rerun()


# ── History Page ──────────────────────────────────────
def show_history():
    st.title("📋 My Reports")
    reports = api_get("/report/history")

    if not reports:
        st.info("No reports uploaded yet. Upload your first blood report to get started.")
        return

    for r in reports:
        sev = r.get("severity_score", "unknown")
        icon = {"emergency": "🚨", "critical": "🔴", "warning": "🟡",
                "safe": "🟢"}.get(sev, "⚪")
        with st.expander(f"{icon} {r['filename']} — {r['uploaded_at'][:10]}"):
            col1, col2 = st.columns(2)
            col1.write(f"**Severity:** {icon} {sev.upper()}")
            col2.write(f"**Uploaded:** {r['uploaded_at'][:16].replace('T', ' ')}")

            if st.button("View Full Analysis", key=f"view_{r['id']}"):
                full = api_get(f"/report/{r['id']}")
                if full:
                    st.session_state.current_report = {
                        "report_id": full["id"],
                        "extracted_values": full.get("extracted_values", {}),
                        "analysis_results": full["analysis_results"],
                        "ml_risk_predictions": full.get("ml_risk_predictions", {}),
                        "ai_explanation": full["ai_explanation"],
                        "extraction_method": full.get("extraction_method", "regex"),
                        "disclaimer": generate_disclaimer_local(full["analysis_results"]),
                    }
                    st.session_state.page = "results"
                    st.rerun()


def generate_disclaimer_local(results):
    """Simple local disclaimer generation for history view."""
    levels = [r.get("severity") for r in results]
    if "panic" in levels:
        return {"level": "emergency", "message": "🚨 Emergency values detected.", "specialists": []}
    if "critical" in levels:
        return {"level": "critical", "message": "🔴 Critical values detected.", "specialists": []}
    if "borderline" in levels:
        return {"level": "warning", "message": "🟡 Borderline values detected.", "specialists": []}
    return {"level": "safe", "message": "🟢 All values normal.", "specialists": []}


# ── Analytics Page ────────────────────────────────────
def show_analytics():
    st.title("📊 Analytics Dashboard")
    data = api_get("/analytics")
    if not data or data.get("total_queries", 0) == 0:
        st.info("No analytics data yet.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Analyses", data.get("total_queries", 0))
    col2.metric("Unique Users", data.get("unique_users", 0))
    col3.metric("Avg Latency", f"{data.get('avg_latency_ms', 0):.0f}ms")

    if data.get("queries_per_day"):
        import pandas as pd
        df = pd.DataFrame(list(data["queries_per_day"].items()), columns=["Date", "Count"])
        st.subheader("📈 Analyses Per Day")
        st.bar_chart(df.set_index("Date"))

    if data.get("top_topics"):
        st.subheader("🔥 Top Topics")
        import pandas as pd
        st.dataframe(pd.DataFrame(data["top_topics"], columns=["Word", "Count"]),
                     use_container_width=True, hide_index=True)


# ── Router ────────────────────────────────────────────
if not st.session_state.token:
    show_login()
else:
    show_app()
