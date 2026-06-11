import streamlit as st
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="E-Wallet Customer Segmentation",
    page_icon="💳",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8f9fb; }
.block-container { padding: 1.5rem 2rem 3rem; }
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #3266AD; border-bottom: 2px solid #3266AD;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=0.2):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def metric_card(col, label, value, sub=""):
    col.markdown(f"""
    <div style='background:white;border-radius:12px;padding:16px 18px;border:1px solid #e8eaed;text-align:center;'>
        <div style='font-size:11px;color:#999;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;'>{label}</div>
        <div style='font-size:20px;font-weight:700;color:#1a1a2e;'>{value}</div>
        <div style='font-size:11px;color:#bbb;margin-top:2px;'>{sub}</div>
    </div>""", unsafe_allow_html=True)

def mini_bar(label, value, color, label_width="50px"):
    return f"""
    <div style='display:flex;align-items:center;gap:6px;margin-bottom:4px;'>
      <span style='font-size:10px;color:#aaa;width:{label_width};flex-shrink:0;'>{label}</span>
      <div style='flex:1;height:5px;background:#f0f0f0;border-radius:3px;overflow:hidden;'>
        <div style='width:{value}%;height:100%;background:{color};border-radius:3px;'></div>
      </div>
      <span style='font-size:10px;color:#aaa;width:26px;text-align:right;'>{value}</span>
    </div>"""

def tag_badge(label, bg, color):
    return f"<span style='display:inline-block;font-size:10px;font-weight:600;padding:2px 9px;border-radius:20px;background:{bg};color:{color};margin-bottom:6px;'>{label}</span>"

def strategy_box(label, color, bg, actions):
    acts = " &nbsp;·&nbsp; ".join(actions)
    return f"""
    <div style='background:{bg};border-left:4px solid {color};border-radius:0 8px 8px 0;padding:10px 14px;margin-top:8px;'>
        <div style='font-size:12px;font-weight:700;color:{color};margin-bottom:3px;'>{label}</div>
        <div style='font-size:12px;color:#555;line-height:1.6;'>{acts}</div>
    </div>"""

# ── Data ─────────────────────────────────────────────────────────────────────
COLORS = ["#3266AD", "#1D9E75", "#D85A30", "#8B5CF6"]

RFM_CLUSTERS = [
    dict(id=0, name="Champions",       tag="VIP",          tag_bg="#E6F1FB", tag_color="#185FA5", color=COLORS[0],
         pct=18, size="~105K users",   recency=95, frequency=92, monetary=90,
         aov="320K VND", freq="9.2×/mo", voucher="38%", channel="App-first",
         desc="High recency, high frequency, high spend. Most valuable cohort.",
         strategy_label="LTV Expansion", strategy_color="#185FA5", strategy_bg="#EBF2FD",
         actions=["VIP tier with exclusive perks", "App gamification & loyalty streaks", "Early access to new merchants"]),
    dict(id=1, name="Loyal customers", tag="Retain",       tag_bg="#E1F5EE", tag_color="#0F6E56", color=COLORS[1],
         pct=24, size="~140K users",   recency=78, frequency=75, monetary=65,
         aov="210K VND", freq="5.8×/mo", voucher="55%", channel="Mixed",
         desc="Consistent transactors. Moderate spend, voucher-sensitive.",
         strategy_label="Engagement",   strategy_color="#0F6E56", strategy_bg="#E6F5EF",
         actions=["Conditional promotions ($5 off >$20)", "Month-over-month streak rewards", "Cross-merchant bundles"]),
    dict(id=2, name="At-risk",         tag="Win-back",     tag_bg="#FAECE7", tag_color="#993C1D", color=COLORS[2],
         pct=31, size="~181K users",   recency=28, frequency=35, monetary=40,
         aov="145K VND", freq="1.4×/mo", voucher="71%", channel="Store walk-in",
         desc="Declining activity. Last purchase was 60–90 days ago.",
         strategy_label="Re-activation", strategy_color="#993C1D", strategy_bg="#FBF0EC",
         actions=["Win-back push notification", "Time-limited voucher (7 days)", "Personalised 'we miss you' offer"]),
    dict(id=3, name="New / one-time",  tag="Convert",      tag_bg="#F0EBFF", tag_color="#6D28D9", color=COLORS[3],
         pct=27, size="~157K users",   recency=60, frequency=8,  monetary=20,
         aov="95K VND",  freq="1.1×/mo", voucher="82%", channel="In-store",
         desc="One-time or very occasional users. High voucher reliance.",
         strategy_label="Conversion",   strategy_color="#6D28D9", strategy_bg="#F5F0FF",
         actions=["Newcomer welcome combo", "In-store QR discount", "Second-order incentive within 14 days"]),
]

CUSTOM_CLUSTERS = [
    dict(id=0, name="Casual walk-ins",    tag="Low retention",   tag_bg="#F0EBFF", tag_color="#6D28D9", color=COLORS[3],
         pct=28, size="~163K users",
         bars={"Dine-in": 82, "Delivery": 12, "Take-away": 6, "App usage": 18, "Voucher use": 78},
         aov="88K VND",  stores="1.1 stores", channel="Store / Call",   province="Tier-2 cities",
         desc="Primarily one-time physical visits. Low digital footprint.",
         strategy_label="Conversion",    strategy_color="#6D28D9", strategy_bg="#F5F0FF",
         actions=["In-store discount combos", "Newcomer QR voucher", "Convert to app with first-order reward"]),
    dict(id=1, name="In-store loyalists", tag="2nd highest value",tag_bg="#E1F5EE", tag_color="#0F6E56", color=COLORS[1],
         pct=22, size="~128K users",
         bars={"Dine-in": 91, "Delivery": 5,  "Take-away": 4, "App usage": 32, "Voucher use": 44},
         aov="195K VND", stores="2.4 stores", channel="Store / App",    province="HCMC / HN",
         desc="Highest dine-in rate. Store is a 'third place' for them.",
         strategy_label="Experience",    strategy_color="#0F6E56", strategy_bg="#E6F5EF",
         actions=["QR-table ordering upgrade", "Group & family-size sets", "Loyalty stamp card for regulars"]),
    dict(id=2, name="Tech-savvies",       tag="Highest value",   tag_bg="#E6F1FB", tag_color="#185FA5", color=COLORS[0],
         pct=20, size="~117K users",
         bars={"Dine-in": 22, "Delivery": 48, "Take-away": 30, "App usage": 96, "Voucher use": 61},
         aov="340K VND", stores="4.8 stores", channel="App-first",      province="HCMC",
         desc="Multi-merchant app power users. Highest AOV and breadth.",
         strategy_label="LTV Expansion", strategy_color="#185FA5", strategy_bg="#EBF2FD",
         actions=["VIP tier & app gamification", "Conditional promotions to grow basket", "Beta feature early access"]),
    dict(id=3, name="Delivery spenders",  tag="High value",      tag_bg="#FAECE7", tag_color="#993C1D", color=COLORS[2],
         pct=30, size="~175K users",
         bars={"Dine-in": 8,  "Delivery": 88, "Take-away": 4, "App usage": 74, "Voucher use": 68},
         aov="265K VND", stores="3.1 stores", channel="App / Website",  province="HCMC / HN",
         desc="Delivery-dominant. Office workers and busy urban segment.",
         strategy_label="Retention",     strategy_color="#993C1D", strategy_bg="#FBF0EC",
         actions=["Office combo meal bundles", "Peak-hour push notifications", "Free delivery threshold incentive"]),
]

# ── Chart functions ───────────────────────────────────────────────────────────
def donut_chart(clusters, title):
    fig = go.Figure(go.Pie(
        labels=[c["name"] for c in clusters],
        values=[c["pct"]  for c in clusters],
        hole=0.6,
        marker=dict(colors=[c["color"] for c in clusters], line=dict(color="#fff", width=2)),
        textinfo="percent", textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value}% of users<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color="#555"), x=0),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=-0.18, font=dict(size=11)),
        margin=dict(t=40, b=20, l=10, r=10), height=300,
        paper_bgcolor="white", plot_bgcolor="white",
    )
    return fig

def rfm_bar_chart(clusters):
    fig = go.Figure()
    for dim, col in [("recency","#7F77DD"),("frequency","#1D9E75"),("monetary","#D85A30")]:
        fig.add_trace(go.Bar(
            name=dim.capitalize(),
            x=[c["name"] for c in clusters],
            y=[c[dim] for c in clusters],
            marker_color=col, marker_line_width=0,
        ))
    fig.update_layout(
        barmode="group",
        title=dict(text="RFM scores by cluster", font=dict(size=13, color="#555"), x=0),
        yaxis=dict(range=[0, 115], tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation="h", x=0, y=1.18, font=dict(size=11)),
        margin=dict(t=50, b=10, l=10, r=10), height=300,
        paper_bgcolor="white", plot_bgcolor="#fafafa",
    )
    return fig

def channel_bar_chart(clusters):
    fig = go.Figure()
    for ch, col in [("Dine-in","#3266AD"),("Delivery","#1D9E75"),("Take-away","#D85A30")]:
        fig.add_trace(go.Bar(
            name=ch,
            x=[c["name"] for c in clusters],
            y=[c["bars"][ch] for c in clusters],
            marker_color=col, marker_line_width=0,
        ))
    fig.update_layout(
        barmode="stack",
        title=dict(text="Channel mix across clusters (%)", font=dict(size=13, color="#555"), x=0),
        yaxis=dict(range=[0, 115], ticksuffix="%", tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=11)),
        legend=dict(orientation="h", x=0, y=1.18, font=dict(size=11)),
        margin=dict(t=50, b=10, l=10, r=10), height=300,
        paper_bgcolor="white", plot_bgcolor="#fafafa",
    )
    return fig

def radar_chart(cluster):
    cats   = list(cluster["bars"].keys())
    vals   = list(cluster["bars"].values())
    cats_c = cats + [cats[0]]
    vals_c = vals + [vals[0]]
    fig = go.Figure(go.Scatterpolar(
        r=vals_c, theta=cats_c,
        fill="toself",
        fillcolor=hex_to_rgba(cluster["color"], 0.2),
        line=dict(color=cluster["color"], width=2),
        name=cluster["name"],
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        margin=dict(t=20, b=20, l=30, r=30), height=300,
        paper_bgcolor="white",
    )
    return fig

# ── Detail panel ─────────────────────────────────────────────────────────────
def render_detail(c, mode="rfm"):
    st.markdown(
        tag_badge(c["tag"], c["tag_bg"], c["tag_color"]) +
        f" <span style='font-size:19px;font-weight:700;color:#1a1a2e;margin-left:6px;'>{c['name']}</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"<p style='font-size:13px;color:#666;margin:4px 0 14px;'>{c['desc']}</p>", unsafe_allow_html=True)

    if mode == "rfm":
        items = [("Avg order value", c["aov"]), ("Order frequency", c["freq"]),
                 ("Voucher rate", c["voucher"]), ("Preferred channel", c["channel"])]
    else:
        items = [("Avg order value", c["aov"]), ("Stores visited", c["stores"]),
                 ("Order source", c["channel"]), ("Geography", c["province"])]

    m_cols = st.columns(4)
    for col, (label, val) in zip(m_cols, items):
        col.markdown(f"""
        <div style='background:#f8f9fb;border-radius:10px;padding:12px 14px;text-align:center;border:1px solid #eee;'>
            <div style='font-size:10px;color:#aaa;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px;'>{label}</div>
            <div style='font-size:16px;font-weight:700;color:#1a1a2e;'>{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#bbb;text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'>Recommended strategy</div>" +
        strategy_box(c["strategy_label"], c["strategy_color"], c["strategy_bg"], c["actions"]),
        unsafe_allow_html=True,
    )

# ── Cluster summary card HTML ─────────────────────────────────────────────────
def rfm_card_html(c, selected=False):
    border = "border:1.5px solid #5b8dee;box-shadow:0 4px 16px rgba(91,141,238,.18);" if selected else "border:1px solid #e8eaed;"
    bars = "".join(mini_bar(d[:3].capitalize(), c[d], c["color"]) for d in ["recency","frequency","monetary"])
    return f"""
    <div style='background:white;border-radius:12px;padding:14px 16px;{border}'>
      {tag_badge(c["tag"], c["tag_bg"], c["tag_color"])}
      <div style='font-size:14px;font-weight:700;color:#1a1a2e;margin-bottom:2px;'>{c["name"]}</div>
      <div style='font-size:11px;color:#bbb;margin-bottom:10px;'>{c["pct"]}% &nbsp;·&nbsp; {c["size"]}</div>
      {bars}
    </div>"""

def custom_card_html(c, selected=False):
    border = "border:1.5px solid #5b8dee;box-shadow:0 4px 16px rgba(91,141,238,.18);" if selected else "border:1px solid #e8eaed;"
    bar_items = list(c["bars"].items())[:3]
    bars = "".join(mini_bar(k, v, c["color"], label_width="58px") for k, v in bar_items)
    return f"""
    <div style='background:white;border-radius:12px;padding:14px 16px;{border}'>
      {tag_badge(c["tag"], c["tag_bg"], c["tag_color"])}
      <div style='font-size:14px;font-weight:700;color:#1a1a2e;margin-bottom:2px;'>{c["name"]}</div>
      <div style='font-size:11px;color:#bbb;margin-bottom:10px;'>{c["pct"]}% &nbsp;·&nbsp; {c["size"]}</div>
      {bars}
    </div>"""

# ════════════════════════════════════════════════════════════════════════════
# PAGE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='margin-bottom:1.5rem;'>
  <div style='font-size:11px;font-weight:600;color:#aaa;text-transform:uppercase;letter-spacing:.1em;'>Vietnamese F&B e-wallet · Oct 2021 – Jan 2023</div>
  <h1 style='font-size:26px;font-weight:700;color:#1a1a2e;margin:4px 0 2px;'>Customer Segmentation Dashboard</h1>
  <div style='font-size:13px;color:#999;'>1,048,575 transactions · 583,618 unique users · K-Means + PCA</div>
</div>
""", unsafe_allow_html=True)

# KPI row
k1, k2, k3, k4 = st.columns(4)
metric_card(k1, "Total transactions",   "1.05M", "Oct 2021 – Jan 2023")
metric_card(k2, "Unique users",         "583,618", "F&B sector")
metric_card(k3, "Clusters (custom)",    "4", "K-Means + PCA")
metric_card(k4, "PCA variance",         "77%", "3 components")

st.markdown("<br>", unsafe_allow_html=True)

tab_rfm, tab_custom = st.tabs(["📊  RFM Segmentation", "🧠  Custom Behavioral Features"])

# ════════════════════════════════════════════════════════════════
# TAB 1 — RFM
# ════════════════════════════════════════════════════════════════
with tab_rfm:
    st.markdown("""
    <p style='font-size:13px;color:#666;margin-bottom:1rem;'>
    RFM clustering applies <b>Yeo-Johnson transformation</b> then <b>K-Means</b> on Recency, Frequency, and Monetary scores.
    Select a cluster to explore its profile.
    </p>""", unsafe_allow_html=True)

    rfm_names   = [c["name"] for c in RFM_CLUSTERS]
    selected_rfm = st.radio("Cluster", rfm_names, horizontal=True, key="rfm_radio", label_visibility="collapsed")
    rfm_idx      = rfm_names.index(selected_rfm)
    rc           = RFM_CLUSTERS[rfm_idx]

    # Cluster cards
    cc1, cc2, cc3, cc4 = st.columns(4)
    for col, c, i in zip([cc1, cc2, cc3, cc4], RFM_CLUSTERS, range(4)):
        col.markdown(rfm_card_html(c, selected=(i == rfm_idx)), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detail panel
    with st.container(border=True):
        render_detail(rc, mode="rfm")

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(donut_chart(RFM_CLUSTERS, "User distribution by cluster"), use_container_width=True)
    with ch2:
        st.plotly_chart(rfm_bar_chart(RFM_CLUSTERS), use_container_width=True)

    # Table
    st.markdown("<div style='font-size:11px;font-weight:600;color:#bbb;text-transform:uppercase;letter-spacing:.06em;margin:1rem 0 .5rem;'>RFM score summary</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{
        "Cluster": c["name"], "Share (%)": c["pct"], "Users": c["size"],
        "Recency": c["recency"], "Frequency": c["frequency"], "Monetary": c["monetary"],
        "AOV": c["aov"], "Order freq": c["freq"], "Voucher rate": c["voucher"],
    } for c in RFM_CLUSTERS]), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════
# TAB 2 — Custom Features
# ════════════════════════════════════════════════════════════════
with tab_custom:
    st.markdown("""
    <p style='font-size:13px;color:#666;margin-bottom:1rem;'>
    Advanced behavioral clustering uses high-dimensional feature engineering (channel mix, voucher reliance,
    geographic footprint, AOV, store diversity) reduced via <b>PCA (77% variance)</b> then clustered with <b>K-Means</b>.
    </p>""", unsafe_allow_html=True)

    custom_names   = [c["name"] for c in CUSTOM_CLUSTERS]
    selected_custom = st.radio("Cluster", custom_names, horizontal=True, key="custom_radio", label_visibility="collapsed")
    custom_idx      = custom_names.index(selected_custom)
    cc              = CUSTOM_CLUSTERS[custom_idx]

    # Cluster cards
    cc1, cc2, cc3, cc4 = st.columns(4)
    for col, c, i in zip([cc1, cc2, cc3, cc4], CUSTOM_CLUSTERS, range(4)):
        col.markdown(custom_card_html(c, selected=(i == custom_idx)), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Detail panel
    with st.container(border=True):
        render_detail(cc, mode="custom")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:11px;font-weight:600;color:#bbb;text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;'>Feature breakdown</div>", unsafe_allow_html=True)

        fb1, fb2 = st.columns(2)
        items_list = list(cc["bars"].items())
        for k, v in items_list[:3]:
            fb1.markdown(f"""
            <div style='margin-bottom:8px;'>
              <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;'>
                <span style='color:#555;'>{k}</span><span style='font-weight:700;color:#1a1a2e;'>{v}%</span>
              </div>
              <div style='height:8px;background:#f0f0f0;border-radius:4px;overflow:hidden;'>
                <div style='width:{v}%;height:100%;background:{cc["color"]};border-radius:4px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)
        for k, v in items_list[3:]:
            fb2.markdown(f"""
            <div style='margin-bottom:8px;'>
              <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:3px;'>
                <span style='color:#555;'>{k}</span><span style='font-weight:700;color:#1a1a2e;'>{v}%</span>
              </div>
              <div style='height:8px;background:#f0f0f0;border-radius:4px;overflow:hidden;'>
                <div style='width:{v}%;height:100%;background:{cc["color"]};border-radius:4px;'></div>
              </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts
    ch1, ch2 = st.columns(2)
    with ch1:
        st.plotly_chart(donut_chart(CUSTOM_CLUSTERS, "User distribution by cluster"), use_container_width=True)
    with ch2:
        st.plotly_chart(channel_bar_chart(CUSTOM_CLUSTERS), use_container_width=True)

    # Radar
    st.markdown("<div style='font-size:11px;font-weight:600;color:#bbb;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px;'>Behavioral radar — selected cluster</div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns([1, 2, 1])
    with r2:
        st.markdown(f"<p style='text-align:center;font-size:14px;font-weight:700;color:{cc['color']};margin-bottom:0;'>{cc['name']}</p>", unsafe_allow_html=True)
        st.plotly_chart(radar_chart(cc), use_container_width=True)

    # Table
    st.markdown("<div style='font-size:11px;font-weight:600;color:#bbb;text-transform:uppercase;letter-spacing:.06em;margin:1rem 0 .5rem;'>Cluster summary table</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{
        "Cluster": c["name"], "Share (%)": c["pct"], "Users": c["size"],
        "AOV": c["aov"], "Stores visited": c["stores"],
        "Preferred channel": c["channel"], "Geography": c["province"],
        "Strategy": c["strategy_label"],
    } for c in CUSTOM_CLUSTERS]), use_container_width=True, hide_index=True)
