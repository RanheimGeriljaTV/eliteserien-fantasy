"""
Eliteserien Fantasy Predictor & Transfer Optimizer
===================================================
En komplett web-applikasjon i Streamlit for analyse, poengprediksjoner (xP),
og matematisk overgangsoptimering med PuLP (MILP) for Eliteserien Fantasy.

Installasjon:
    pip install streamlit requests pandas numpy pulp

Oppstart:
    streamlit run app.py
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import pulp
import altair as alt
import json
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional

# ==============================================================================
# 1. KONFIGURASJON & DESIGN
# ==============================================================================
st.set_page_config(
    page_title="Eliteserien Fantasy Predictor & Transfer Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

def build_custom_css(theme_choice: str = "🖥️ Auto (Følg system)") -> str:
    """
    Genererer dynamisk CSS tilpasset valgt fargetema:
    - ☀️ Lys: Rent, skarpt hvitt design med høykontrast slate-tekst
    - 🌙 Mørk: Moderne dypblå/slate-dark med glødende aksenter og skarp hvit/lysgrå tekst
    - 🖥️ Auto: Følger OS/nettleserens 'prefers-color-scheme' automatisk
    """
    light_vars = """
        --bg-main: #f8fafc;
        --sidebar-bg: #f8fafc;
        --card-bg: #ffffff;
        --card-border: #e2e8f0;
        --card-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.05);
        --text-primary: #0f172a;
        --text-secondary: #334155;
        --text-muted: #64748b;
        --input-bg: #ffffff;
        --input-border: #cbd5e1;
        --input-text: #0f172a;
        --tab-bg: #e2e8f0;
        --tab-border: #cbd5e1;
        --tab-text: #475569;
        --tab-hover-bg: rgba(255, 255, 255, 0.7);
        --pitch-card-bg: #ffffff;
        --pitch-card-text: #0f172a;
        --pitch-card-sub: #475569;
        --pitch-card-border: #cbd5e1;
        --pitch-card-shadow: 0 8px 22px rgba(0, 0, 0, 0.28);
        --pitch-c-card-bg: #fffdf5;
        --pitch-pts-green: #047857;
        --pitch-status-ft-bg: #e2e8f0;
        --pitch-status-ft-text: #1e293b;
        --pitch-status-live-bg: #fee2e2;
        --pitch-status-live-text: #b91c1c;
        --pitch-status-sched-bg: #f1f5f9;
        --pitch-status-sched-text: #64748b;
        --pitch-pts-box-bg: rgba(241, 245, 249, 0.9);
        --pitch-pts-box-border: #e2e8f0;
        --pitch-card-dim-bg: #f1f5f9;
        --pitch-card-dim-border: #cbd5e1;
        --trade-out-bg: #fff5f5;
        --trade-out-border: #fecdd3;
        --trade-in-bg: #f0fdf4;
        --trade-in-border: #bbf7d0;
        --sidebar-accent: #0284c7;
        --photo-halo: rgba(0, 0, 0, 0.05);
        --pitch-status-done-bg: #dcfce7;
        --pitch-status-done-text: #15803d;
        --pitch-status-done-border: #86efac;
    """

    dark_vars = """
        --bg-main: #0b1120;
        --sidebar-bg: #0f172a;
        --card-bg: #1e293b;
        --card-border: #334155;
        --card-shadow: 0 6px 20px -2px rgba(0, 0, 0, 0.5);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --input-bg: #0f172a;
        --input-border: #334155;
        --input-text: #f8fafc;
        --tab-bg: #1e293b;
        --tab-border: #334155;
        --tab-text: #94a3b8;
        --tab-hover-bg: rgba(255, 255, 255, 0.08);
        --pitch-card-bg: #1e293b;
        --pitch-card-text: #f8fafc;
        --pitch-card-sub: #cbd5e1;
        --pitch-card-border: #475569;
        --pitch-card-shadow: 0 8px 24px rgba(0, 0, 0, 0.55);
        --pitch-c-card-bg: #292524;
        --pitch-pts-green: #34d399;
        --pitch-status-ft-bg: #334155;
        --pitch-status-ft-text: #f1f5f9;
        --pitch-status-live-bg: rgba(239, 68, 68, 0.25);
        --pitch-status-live-text: #fca5a5;
        --pitch-status-sched-bg: #334155;
        --pitch-status-sched-text: #94a3b8;
        --pitch-pts-box-bg: rgba(15, 23, 42, 0.85);
        --pitch-pts-box-border: #334155;
        --pitch-card-dim-bg: #0f172a;
        --pitch-card-dim-border: #334155;
        --trade-out-bg: rgba(239, 68, 68, 0.12);
        --trade-out-border: rgba(239, 68, 68, 0.35);
        --trade-in-bg: rgba(16, 185, 129, 0.12);
        --trade-in-border: rgba(16, 185, 129, 0.35);
        --sidebar-accent: #38bdf8;
        --photo-halo: rgba(255, 255, 255, 0.14);
        --pitch-status-done-bg: rgba(34, 197, 94, 0.18);
        --pitch-status-done-text: #4ade80;
        --pitch-status-done-border: rgba(34, 197, 94, 0.4);
    """

    if "Mørk" in theme_choice or "Dark" in theme_choice:
        theme_scope = f":root {{ {dark_vars} }}"
    elif "Lys" in theme_choice or "Light" in theme_choice:
        theme_scope = f":root {{ {light_vars} }}"
    else:
        theme_scope = f"""
        :root {{ {light_vars} }}
        @media (prefers-color-scheme: dark) {{
            :root {{ {dark_vars} }}
        }}
        """

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

    {theme_scope}

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}

    /* Global bakgrunn og tekstfarge */
    .stApp {{
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }}

    /* Header og markdown farger */
    h1, h2, h3, h4, h5, h6 {{
        color: var(--text-primary) !important;
        font-weight: 800 !important;
    }}
    .stMarkdown p, .stMarkdown span {{
        color: var(--text-primary);
    }}

    /* =========================================================================
       HERO BANNER
       ========================================================================= */
    .hero-banner {{
        background: linear-gradient(135deg, #091e3a 0%, #0f3966 50%, #1e528a 100%);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 10px 30px -5px rgba(9, 30, 58, 0.35);
        color: #ffffff !important;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.18);
    }}
    .hero-banner::after {{
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 340px;
        height: 340px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.35) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }}
    .hero-banner * {{
        color: #ffffff !important;
    }}
    .hero-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #93c5fd !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 8px;
    }}
    .hero-title {{
        font-size: 2.15rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0 0 12px 0;
        color: #ffffff !important;
        text-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }}
    .hero-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        align-items: center;
    }}
    .hero-pill {{
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(6px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        padding: 6px 14px;
        font-size: 0.88rem;
        color: #f1f5f9 !important;
    }}
    .hero-pill b {{
        color: #ffffff !important;
        font-weight: 700;
    }}

    /* =========================================================================
       METRIKK-KORT (st.metric)
       ========================================================================= */
    div[data-testid="stMetric"] {{
        background: var(--card-bg) !important;
        border: 1.5px solid var(--card-border) !important;
        border-radius: 16px !important;
        padding: 16px 18px !important;
        box-shadow: var(--card-shadow) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }}
    div[data-testid="stMetric"]::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3.5px;
        background: linear-gradient(90deg, #2563eb, #38bdf8);
    }}
    div[data-testid="stMetric"]:hover {{
        transform: translateY(-3px) !important;
        box-shadow: 0 10px 24px -4px rgba(37, 99, 235, 0.2) !important;
        border-color: #3b82f6 !important;
    }}
    div[data-testid="stMetricLabel"] p {{
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.8px !important;
        color: var(--text-muted) !important;
        margin-bottom: 2px !important;
    }}
    div[data-testid="stMetricValue"] div {{
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.5px !important;
    }}

    /* =========================================================================
       FORM- & INPUT-FELTER
       ========================================================================= */
    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p,
    label[data-testid="stWidgetLabel"] {{
        font-size: 0.82rem !important;
        font-weight: 800 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.6px !important;
        color: var(--text-primary) !important;
        margin-bottom: 6px !important;
        display: flex !important;
        align-items: center !important;
        gap: 4px !important;
    }}

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {{
        background-color: var(--input-bg) !important;
        border: 1.5px solid var(--input-border) !important;
        border-radius: 10px !important;
        color: var(--input-text) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 9px 14px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus {{
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25) !important;
    }}

    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        border: 1.5px solid var(--input-border) !important;
        border-radius: 10px !important;
        color: var(--input-text) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stSelectbox"] div[data-baseweb="select"] span {{
        color: var(--input-text) !important;
    }}
    div[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div {{
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.25) !important;
    }}

    /* Dropdown popover list items */
    div[data-baseweb="popover"] ul,
    ul[role="listbox"] {{
        background-color: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
    }}
    li[role="option"] {{
        color: var(--text-primary) !important;
    }}
    li[role="option"]:hover, li[aria-selected="true"] {{
        background-color: var(--tab-hover-bg) !important;
    }}

    /* Radio buttons */
    div[data-testid="stRadio"] div[role="radiogroup"] {{
        background: var(--input-bg) !important;
        padding: 4px !important;
        border-radius: 12px !important;
        border: 1px solid var(--input-border) !important;
        display: inline-flex !important;
        gap: 6px !important;
    }}
    div[data-testid="stRadio"] label {{
        background: transparent !important;
        padding: 5px 14px !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
    }}
    div[data-testid="stRadio"] label p {{
        color: var(--text-primary) !important;
    }}

    /* =========================================================================
       SEGMENTERTE TABS
       ========================================================================= */
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {{
        gap: 8px !important;
        background: var(--tab-bg) !important;
        padding: 6px !important;
        border-radius: 16px !important;
        border: 1px solid var(--tab-border) !important;
        margin-bottom: 24px !important;
    }}
    button[data-baseweb="tab"] {{
        border-radius: 11px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        color: var(--tab-text) !important;
        padding: 10px 18px !important;
        border: none !important;
        background: transparent !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }}
    button[data-baseweb="tab"]:hover {{
        background: var(--tab-hover-bg) !important;
        color: var(--text-primary) !important;
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        font-weight: 800 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
    }}

    /* =========================================================================
       SIDEBAR KORT & SEKSJONER
       ========================================================================= */
    section[data-testid="stSidebar"] {{
        background-color: var(--sidebar-bg) !important;
        border-right: 1.5px solid var(--card-border) !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: var(--text-primary);
    }}
    .sidebar-card {{
        background: var(--card-bg);
        border: 1.5px solid var(--card-border);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: var(--card-shadow);
    }}
    .sidebar-title {{
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--sidebar-accent) !important;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* =========================================================================
       KORT, PANELER & TRADE-BOXER
       ========================================================================= */
    .filter-panel {{
        background: var(--card-bg);
        border: 1.5px solid var(--card-border);
        border-radius: 16px;
        padding: 20px 22px;
        margin-bottom: 22px;
        box-shadow: var(--card-shadow);
    }}
    .panel-header {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.92rem;
        font-weight: 800;
        color: var(--text-primary);
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 14px;
        border-bottom: 1.5px solid var(--card-border);
        padding-bottom: 8px;
    }}

    .trade-box-out {{
        background: var(--trade-out-bg);
        border: 1.5px solid var(--trade-out-border);
        border-left: 5px solid #ef4444;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }}
    .trade-box-in {{
        background: var(--trade-in-bg);
        border: 1.5px solid var(--trade-in-border);
        border-left: 5px solid #10b981;
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 14px;
    }}
    .trade-badge-out {{
        display: inline-block;
        background: #ef4444;
        color: #ffffff !important;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }}
    .trade-badge-in {{
        display: inline-block;
        background: #10b981;
        color: #ffffff !important;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 3px 8px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }}

    /* Primærknapper */
    div.stButton > button {{
        border-radius: 11px !important;
        font-weight: 700 !important;
        font-size: 0.93rem !important;
        padding: 9px 18px !important;
        border: 1.5px solid var(--input-border) !important;
        transition: all 0.2s ease !important;
    }}
    div.stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15) !important;
    }}
    div.stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35) !important;
    }}

    /* Tabell-rammer */
    div[data-testid="stDataFrame"] {{
        border: 1.5px solid var(--card-border) !important;
        border-radius: 14px !important;
        overflow: hidden !important;
        box-shadow: var(--card-shadow) !important;
        background: var(--card-bg) !important;
    }}

    /* Expanders */
    div[data-testid="stExpander"] {{
        background: var(--card-bg) !important;
        border: 1.5px solid var(--card-border) !important;
        border-radius: 14px !important;
        box-shadow: var(--card-shadow) !important;
        margin-bottom: 14px !important;
    }}
    div[data-testid="stExpander"] details summary span,
    div[data-testid="stExpander"] details summary p {{
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] span,
    div[data-testid="stExpander"] li {{
        color: var(--text-secondary);
    }}

    /* =========================================================================
       FOTBALLBANE & STORE SPILLERKORT
       ========================================================================= */
    .pitch-container {{
        background: radial-gradient(circle, #238b38 0%, #166528 75%, #0f491c 100%);
        border: 3px solid #ffffff;
        border-radius: 22px;
        padding: 26px 18px;
        margin: 18px 0;
        position: relative;
        box-shadow: inset 0 0 50px rgba(0,0,0,0.6), 0 10px 28px rgba(0,0,0,0.3);
        min-height: 1150px;
        display: flex;
        flex-direction: column;
        justify-content: space-around;
    }}
    .pitch-midfield-line {{
        position: absolute;
        top: 50%;
        left: 3%;
        right: 3%;
        height: 2px;
        background: rgba(255,255,255,0.35);
        pointer-events: none;
    }}
    .pitch-center-circle {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 150px;
        height: 150px;
        border: 2px solid rgba(255,255,255,0.35);
        border-radius: 50%;
        pointer-events: none;
    }}
    .pitch-row {{
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
        position: relative;
        z-index: 2;
        margin: 10px 0;
    }}

    /* Store Spillerkort på banen */
    .pitch-player-card {{
        position: relative;
        width: 200px;
        min-width: 185px;
        max-width: 225px;
        background: var(--pitch-card-bg);
        border: 2px solid var(--pitch-card-border);
        border-radius: 18px;
        padding: 10px 12px 14px 12px;
        text-align: center;
        box-shadow: var(--pitch-card-shadow);
        margin: 6px 8px;
        overflow: hidden;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    .pitch-player-card:hover {{
        transform: translateY(-5px) scale(1.03);
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.52);
        border-color: var(--sidebar-accent);
        z-index: 10;
    }}

    /* Spillerbilde cutout - Zoomet inn og beskåret under sponsorlogo for store, tydelige ansikter */
    .card-photo-container {{
        height: 135px;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        margin: 4px auto 6px auto;
        position: relative;
        overflow: hidden;
        border-radius: 14px;
        width: 100%;
    }}
    .card-photo-container::before {{
        content: "";
        position: absolute;
        top: 8px;
        width: 115px;
        height: 115px;
        border-radius: 50%;
        background: radial-gradient(circle, var(--photo-halo, rgba(255, 255, 255, 0.14)) 0%, transparent 72%);
        pointer-events: none;
        z-index: 1;
    }}
    .card-photo-container.no-photo {{
        display: none;
    }}
    .card-player-photo {{
        height: 135px;
        width: auto;
        max-width: 100%;
        object-fit: contain;
        transform-origin: top center;
        transform: scale(1.42) translateY(4px);
        filter: drop-shadow(0 6px 14px rgba(0, 0, 0, 0.45));
        transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1), filter 0.25s ease;
        display: block;
        position: relative;
        z-index: 2;
    }}
    .pitch-player-card:hover .card-player-photo {{
        transform: scale(1.48) translateY(2px);
        filter: drop-shadow(0 10px 22px rgba(0, 0, 0, 0.65));
    }}

    .bench-card .card-photo-container {{
        height: 115px;
        margin: 4px auto 6px auto;
        overflow: hidden;
        border-radius: 12px;
        align-items: flex-start;
    }}
    .bench-card .card-photo-container::before {{
        top: 6px;
        width: 95px;
        height: 95px;
    }}
    .bench-card .card-player-photo {{
        height: 115px;
        transform-origin: top center;
        transform: scale(1.42) translateY(3px);
    }}
    .bench-card:hover .card-player-photo {{
        transform: scale(1.47) translateY(1px);
    }}

    /* Topplinje med klubbfarger */
    .card-club-bar {{
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 5px;
        z-index: 5;
    }}

    /* Topprad med klubbmerke og kapteinsmerke */
    .card-top-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin-top: 4px;
        margin-bottom: 5px;
        min-height: 24px;
    }}
    .card-club-group {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }}
    .club-badge-img {{
        width: 22px;
        height: 22px;
        object-fit: contain;
        filter: drop-shadow(0 1px 3px rgba(0, 0, 0, 0.35));
        display: inline-block;
        vertical-align: middle;
        flex-shrink: 0;
    }}
    .club-pill {{
        display: inline-flex;
        align-items: center;
        font-size: 11px;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 6px;
        letter-spacing: 0.6px;
        text-transform: uppercase;
        box-shadow: 0 1px 3px rgba(0,0,0,0.18);
    }}
    .captain-pill {{
        display: inline-flex;
        align-items: center;
        font-size: 11px;
        font-weight: 800;
        padding: 2px 7px;
        border-radius: 6px;
        color: #ffffff !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }}

    /* Typografi på kortet */
    .pitch-card-name {{
        font-weight: 800;
        font-size: 16px;
        color: var(--pitch-card-text);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        line-height: 1.25;
        margin: 4px 0 3px 0;
    }}
    .pitch-card-fixture {{
        font-size: 12px;
        font-weight: 800;
        border-radius: 7px;
        padding: 3px 8px;
        margin: 4px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .pitch-card-events {{
        font-size: 11px;
        font-weight: 700;
        color: var(--pitch-card-text);
        min-height: 17px;
        margin: 2px 0;
    }}
    .pitch-card-info-text {{
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
        margin: 2px 0;
        min-height: 17px;
    }}

    /* Hovedboks for Poeng og xP */
    .pitch-pts-box {{
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        background: var(--pitch-pts-box-bg);
        border: 1px solid var(--pitch-pts-box-border);
        border-radius: 10px;
        padding: 4px 10px;
        margin-top: 6px;
    }}
    .pitch-pts-label {{
        font-size: 10px;
        font-weight: 800;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }}
    .pitch-pts-val {{
        font-size: 22px;
        font-weight: 900;
        letter-spacing: -0.5px;
        line-height: 1;
    }}

    /* Spesialtilstander for kort */
    .pitch-player-card.is-captain {{
        border: 2px solid #f59e0b !important;
        box-shadow: 0 0 14px rgba(245, 158, 11, 0.35), var(--pitch-card-shadow) !important;
    }}
    .pitch-player-card.is-haul {{
        border: 2px solid #10b981 !important;
    }}
    .pitch-player-card.is-live-match {{
        border: 2px solid #ef4444 !important;
        animation: liveGlow 2s infinite ease-in-out;
    }}
    @keyframes liveGlow {{
        0%, 100% {{ box-shadow: 0 0 10px rgba(239, 68, 68, 0.35), var(--pitch-card-shadow); }}
        50% {{ box-shadow: 0 0 20px rgba(239, 68, 68, 0.7), var(--pitch-card-shadow); }}
    }}

    /* Mørkere / dempet tilstand for kamper som ikke har startet ennå */
    .pitch-player-card.is-upcoming-match {{
        opacity: 0.65;
        filter: saturate(65%);
        background: var(--pitch-card-dim-bg) !important;
        border: 2px dashed var(--pitch-card-dim-border) !important;
    }}
    .pitch-player-card.is-upcoming-match:hover {{
        opacity: 0.95;
        filter: saturate(100%);
        transform: translateY(-4px) scale(1.02);
    }}

    /* Dempet tilstand for skadde/suspenderte spillere i neste runde */
    .pitch-player-card.is-player-out {{
        opacity: 0.65;
        filter: saturate(60%);
        background: var(--pitch-card-dim-bg) !important;
        border: 2px dashed #f87171 !important;
    }}

    /* Benkkort */
    .bench-card {{
        position: relative;
        overflow: hidden;
        background: var(--card-bg);
        border: 1.5px solid var(--card-border);
        border-radius: 18px;
        padding: 12px 14px 14px 14px;
        text-align: center;
        box-shadow: var(--card-shadow);
        transition: transform 0.15s ease;
    }}
    .bench-card:hover {{
        transform: translateY(-2px);
    }}

    /* FDR Farger */
    .fdr-1 {{ background-color: #22c55e; color: white !important; }}
    .fdr-2 {{ background-color: #86efac; color: #064e3b !important; }}
    .fdr-3 {{ background-color: var(--card-border); color: var(--text-primary) !important; }}
    .fdr-4 {{ background-color: #fdba74; color: #7c2d12 !important; }}
    .fdr-5 {{ background-color: #ef4444; color: white !important; }}

    /* Status-merker for live-runden */
    .pitch-card-fixture.is-finished-badge {{
        background: var(--pitch-status-done-bg) !important;
        color: var(--pitch-status-done-text) !important;
        border: 1.5px solid var(--pitch-status-done-border) !important;
        font-weight: 800 !important;
        letter-spacing: 0.3px;
    }}
    .pitch-card-fixture.is-live-badge {{
        background: rgba(239, 68, 68, 0.18) !important;
        color: #ef4444 !important;
        border: 1.5px solid #ef4444 !important;
        font-weight: 800 !important;
        letter-spacing: 0.3px;
    }}
    .pitch-card-fixture.is-upcoming-badge {{
        background: var(--pitch-status-sched-bg) !important;
        color: var(--pitch-status-sched-text) !important;
        border: 1px solid var(--card-border) !important;
        font-weight: 700 !important;
    }}
    .pitch-card-match-sub {{
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        min-height: 17px;
        margin: 2px 0;
    }}

    /* Klikkbare spillerkort & hover-effekt */
    .pitch-card-link {{
        text-decoration: none !important;
        color: inherit !important;
        display: inline-block;
        transition: transform 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: pointer;
        user-select: none;
        outline: none;
    }}
    .pitch-card-link:focus-visible {{
        outline: 2px solid var(--brand-blue);
        border-radius: 12px;
    }}
    .pitch-card-link:hover .pitch-player-card,
    .pitch-card-link:hover .bench-card,
    .pitch-player-card:hover,
    .bench-card:hover {{
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 16px 32px rgba(0, 0, 0, 0.28), 0 0 0 2px var(--brand-blue) !important;
    }}
    .pitch-card-link:active .pitch-player-card,
    .pitch-card-link:active .bench-card {{
        transform: translateY(-2px) scale(0.99);
    }}
    .pitch-card-click-hint {{
        display: block;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: var(--brand-blue);
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 6px;
        padding: 2px 6px;
        margin-top: 5px;
        text-align: center;
        transition: all 0.15s ease;
    }}
    .pitch-card-link:hover .pitch-card-click-hint,
    .pitch-player-card:hover .pitch-card-click-hint,
    .bench-card:hover .pitch-card-click-hint {{
        background: var(--brand-blue);
        color: #ffffff !important;
    }}
    div[class*="st-key-btn_trigger_tab"] {{
        position: fixed !important;
        top: -9999px !important;
        left: -9999px !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        pointer-events: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* =========================================================================
       RESPONSIVT MOBILDESIGN (@media)
       ========================================================================= */
    @media (max-width: 768px) {{
        .block-container {{
            padding-top: 0.75rem !important;
            padding-left: 0.4rem !important;
            padding-right: 0.4rem !important;
            padding-bottom: 2rem !important;
        }}
        .hero-banner {{
            padding: 12px 14px !important;
            border-radius: 14px !important;
        }}
        .hero-title {{
            font-size: 1.3rem !important;
            margin-bottom: 8px !important;
        }}
        .hero-pills {{
            gap: 6px !important;
        }}
        .hero-pill {{
            font-size: 0.74rem !important;
            padding: 4px 8px !important;
        }}
        div[data-testid="stMetric"] {{
            padding: 8px 10px !important;
        }}
        div[data-testid="stMetricValue"] div {{
            font-size: 1.25rem !important;
        }}
        div[data-testid="stMetricLabel"] p {{
            font-size: 0.68rem !important;
        }}
        .pitch-container {{
            padding: 12px 4px !important;
            border-radius: 14px !important;
            border-width: 2px !important;
            min-height: 800px !important;
            margin: 10px 0 !important;
        }}
        .pitch-row {{
            gap: 5px !important;
            margin: 6px 0 !important;
        }}
        .pitch-player-card {{
            width: 120px !important;
            min-width: 90px !important;
            max-width: 130px !important;
            padding: 6px 5px 8px 5px !important;
            margin: 2px 1px !important;
            border-radius: 12px !important;
        }}
        .card-photo-container {{
            height: 70px !important;
            margin: 2px auto 4px auto !important;
        }}
        .card-photo-container::before {{
            width: 60px !important;
            height: 60px !important;
        }}
        .card-player-photo {{
            height: 70px !important;
            transform: scale(1.35) translateY(2px) !important;
        }}
        .pitch-card-name {{
            font-size: 12px !important;
            margin: 2px 0 !important;
        }}
        .club-badge-img {{
            width: 16px !important;
            height: 16px !important;
        }}
        .club-pill {{
            font-size: 8px !important;
            padding: 1px 4px !important;
        }}
        .captain-pill {{
            font-size: 8px !important;
            padding: 1px 4px !important;
        }}
        .pitch-card-fixture {{
            font-size: 8.5px !important;
            padding: 2px 4px !important;
        }}
        .pitch-card-info-text {{
            font-size: 8.5px !important;
        }}
        .pitch-pts-box {{
            padding: 2px 6px !important;
            margin-top: 3px !important;
        }}
        .pitch-pts-val {{
            font-size: 13px !important;
        }}
        .pitch-pts-label {{
            font-size: 7.5px !important;
        }}
        .pitch-card-click-hint {{
            font-size: 7.5px !important;
            padding: 1px 3px !important;
        }}
        .bench-card {{
            padding: 6px 5px 8px 5px !important;
            border-radius: 12px !important;
        }}
        .bench-card .card-photo-container {{
            height: 60px !important;
        }}
        .bench-card .card-player-photo {{
            height: 60px !important;
        }}
        div[data-testid="stTabs"] button {{
            padding: 6px 8px !important;
            font-size: 0.78rem !important;
        }}
    }}

    @media (max-width: 480px) {{
        .pitch-row {{
            gap: 3px !important;
        }}
        .pitch-player-card {{
            width: calc(25% - 4px) !important;
            min-width: 70px !important;
            max-width: 90px !important;
            padding: 4px 3px 6px 3px !important;
            margin: 2px 1px !important;
        }}
        .card-photo-container {{
            height: 52px !important;
        }}
        .card-player-photo {{
            height: 52px !important;
        }}
        .pitch-card-name {{
            font-size: 9.5px !important;
        }}
        .pitch-card-fixture {{
            font-size: 7.5px !important;
            line-height: 1.1 !important;
            padding: 1px 2px !important;
        }}
        .pitch-pts-val {{
            font-size: 11px !important;
        }}
    }}
    </style>
    """
    return css

# Klubb-farger og offisielle logo-URLs for alle lag i Eliteserien og OBOS-ligaen
CLUB_COLORS = {
    "B/G": {"name": "Bodø/Glimt", "primary": "#eab308", "secondary": "#000000", "text": "#000000", "border": "#ca8a04", "gradient": "linear-gradient(90deg, #eab308, #ca8a04)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_4.png"},
    "BRA": {"name": "Brann", "primary": "#dc2626", "secondary": "#ffffff", "text": "#ffffff", "border": "#b91c1c", "gradient": "linear-gradient(90deg, #dc2626, #991b1b)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_298.png"},
    "VIK": {"name": "Viking", "primary": "#0f172a", "secondary": "#38bdf8", "text": "#ffffff", "border": "#0284c7", "gradient": "linear-gradient(90deg, #091e3a, #1e3a8a)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_198.png"},
    "RBK": {"name": "Rosenborg", "primary": "#18181b", "secondary": "#ffffff", "text": "#ffffff", "border": "#52525b", "gradient": "linear-gradient(90deg, #18181b, #27272a)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_145.png"},
    "MOL": {"name": "Molde", "primary": "#2563eb", "secondary": "#ffffff", "text": "#ffffff", "border": "#1d4ed8", "gradient": "linear-gradient(90deg, #2563eb, #1d4ed8)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_3.png"},
    "VIF": {"name": "Vålerenga", "primary": "#1d4ed8", "secondary": "#ef4444", "text": "#ffffff", "border": "#dc2626", "gradient": "linear-gradient(90deg, #1d4ed8, #dc2626)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_169882.png"},
    "FFK": {"name": "Fredrikstad", "primary": "#b91c1c", "secondary": "#ffffff", "text": "#ffffff", "border": "#991b1b", "gradient": "linear-gradient(90deg, #b91c1c, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_117.png"},
    "LSK": {"name": "Lillestrøm", "primary": "#facc15", "secondary": "#111827", "text": "#111827", "border": "#ca8a04", "gradient": "linear-gradient(90deg, #facc15, #000000)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_177.png"},
    "TIL": {"name": "Tromsø", "primary": "#ef4444", "secondary": "#ffffff", "text": "#ffffff", "border": "#b91c1c", "gradient": "linear-gradient(90deg, #ef4444, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_2.png"},
    "S08": {"name": "Sarpsborg 08", "primary": "#0284c7", "secondary": "#ffffff", "text": "#ffffff", "border": "#0369a1", "gradient": "linear-gradient(90deg, #0284c7, #0369a1)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_797.png"},
    "HAM": {"name": "HamKam", "primary": "#16a34a", "secondary": "#ffffff", "text": "#ffffff", "border": "#15803d", "gradient": "linear-gradient(90deg, #16a34a, #15803d)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_63.png"},
    "KBK": {"name": "Kristiansund", "primary": "#1e40af", "secondary": "#ffffff", "text": "#ffffff", "border": "#1d4ed8", "gradient": "linear-gradient(90deg, #1e40af, #1d4ed8)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_785.png"},
    "SAN": {"name": "Sandefjord", "primary": "#1e3a8a", "secondary": "#38bdf8", "text": "#ffffff", "border": "#38bdf8", "gradient": "linear-gradient(90deg, #1e3a8a, #0284c7)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_195.png"},
    "KFUM": {"name": "KFUM Oslo", "primary": "#dc2626", "secondary": "#ffffff", "text": "#ffffff", "border": "#b91c1c", "gradient": "linear-gradient(90deg, #dc2626, #2563eb)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_306.png"},
    "Haugesund": {"name": "Haugesund", "primary": "#1d4ed8", "secondary": "#ffffff", "text": "#ffffff", "border": "#1e40af", "gradient": "linear-gradient(90deg, #1d4ed8, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_202.png"},
    "FKH": {"name": "Haugesund", "primary": "#1d4ed8", "secondary": "#ffffff", "text": "#ffffff", "border": "#1e40af", "gradient": "linear-gradient(90deg, #1d4ed8, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_202.png"},
    "Odd": {"name": "Odd", "primary": "#0f172a", "secondary": "#ffffff", "text": "#ffffff", "border": "#334155", "gradient": "linear-gradient(90deg, #0f172a, #334155)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_270.png"},
    "Strømsgodset": {"name": "Strømsgodset", "primary": "#0c4a6e", "secondary": "#ffffff", "text": "#ffffff", "border": "#0369a1", "gradient": "linear-gradient(90deg, #0c4a6e, #0284c7)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_816.png"},
    "SIF": {"name": "Strømsgodset", "primary": "#0c4a6e", "secondary": "#ffffff", "text": "#ffffff", "border": "#0369a1", "gradient": "linear-gradient(90deg, #0c4a6e, #0284c7)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_816.png"},
    "GOD": {"name": "Strømsgodset", "primary": "#0c4a6e", "secondary": "#ffffff", "text": "#ffffff", "border": "#0369a1", "gradient": "linear-gradient(90deg, #0c4a6e, #0284c7)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_816.png"},
    "AaFK": {"name": "Aalesund", "primary": "#ea580c", "secondary": "#1e40af", "text": "#ffffff", "border": "#c2410c", "gradient": "linear-gradient(90deg, #ea580c, #1e40af)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_15.png"},
    "STA": {"name": "Start", "primary": "#eab308", "secondary": "#18181b", "text": "#18181b", "border": "#ca8a04", "gradient": "linear-gradient(90deg, #eab308, #18181b)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_261.png"},
    "Bryne": {"name": "Bryne", "primary": "#b91c1c", "secondary": "#ffffff", "text": "#ffffff", "border": "#991b1b", "gradient": "linear-gradient(90deg, #b91c1c, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_205.png"},
    "Lyn": {"name": "Lyn", "primary": "#dc2626", "secondary": "#1d4ed8", "text": "#ffffff", "border": "#b91c1c", "gradient": "linear-gradient(90deg, #dc2626, #1d4ed8)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_161785.png"},
    "Moss": {"name": "Moss", "primary": "#eab308", "secondary": "#18181b", "text": "#18181b", "border": "#ca8a04", "gradient": "linear-gradient(90deg, #eab308, #18181b)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_120.png"},
    "Sogndal": {"name": "Sogndal", "primary": "#18181b", "secondary": "#ffffff", "text": "#ffffff", "border": "#334155", "gradient": "linear-gradient(90deg, #18181b, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_107.png"},
    "Stabæk": {"name": "Stabæk", "primary": "#1e3a8a", "secondary": "#38bdf8", "text": "#ffffff", "border": "#0284c7", "gradient": "linear-gradient(90deg, #1e3a8a, #0284c7)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_269.png"},
    "STB": {"name": "Stabæk", "primary": "#1e3a8a", "secondary": "#38bdf8", "text": "#ffffff", "border": "#0284c7", "gradient": "linear-gradient(90deg, #1e3a8a, #0284c7)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_269.png"},
    "Kongsvinger": {"name": "Kongsvinger", "primary": "#dc2626", "secondary": "#ffffff", "text": "#ffffff", "border": "#b91c1c", "gradient": "linear-gradient(90deg, #dc2626, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_5.png"},
    "KIL": {"name": "Kongsvinger", "primary": "#dc2626", "secondary": "#ffffff", "text": "#ffffff", "border": "#b91c1c", "gradient": "linear-gradient(90deg, #dc2626, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_5.png"},
    "Raufoss": {"name": "Raufoss", "primary": "#eab308", "secondary": "#18181b", "text": "#18181b", "border": "#ca8a04", "gradient": "linear-gradient(90deg, #eab308, #ca8a04)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_59.png"},
    "Ranheim TF": {"name": "Ranheim TF", "primary": "#1d4ed8", "secondary": "#ffffff", "text": "#ffffff", "border": "#1e40af", "gradient": "linear-gradient(90deg, #1d4ed8, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_133568.png"},
    "Ranheim": {"name": "Ranheim TF", "primary": "#1d4ed8", "secondary": "#ffffff", "text": "#ffffff", "border": "#1e40af", "gradient": "linear-gradient(90deg, #1d4ed8, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_133568.png"},
    "Sandnes Ulf": {"name": "Sandnes Ulf", "primary": "#0284c7", "secondary": "#ffffff", "text": "#ffffff", "border": "#0369a1", "gradient": "linear-gradient(90deg, #0284c7, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_208313.png"},
    "Egersund": {"name": "Egersund", "primary": "#eab308", "secondary": "#18181b", "text": "#18181b", "border": "#ca8a04", "gradient": "linear-gradient(90deg, #eab308, #18181b)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_475.png"},
    "EIK": {"name": "Egersund", "primary": "#eab308", "secondary": "#18181b", "text": "#18181b", "border": "#ca8a04", "gradient": "linear-gradient(90deg, #eab308, #18181b)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_475.png"},
    "Hødd": {"name": "Hødd", "primary": "#1d4ed8", "secondary": "#ffffff", "text": "#ffffff", "border": "#1e40af", "gradient": "linear-gradient(90deg, #1d4ed8, #ffffff)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_171.png"},
    "Åsane": {"name": "Åsane", "primary": "#ea580c", "secondary": "#18181b", "text": "#18181b", "border": "#c2410c", "gradient": "linear-gradient(90deg, #ea580c, #18181b)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_126.png"},
    "Strømmen": {"name": "Strømmen", "primary": "#ca8a04", "secondary": "#18181b", "text": "#18181b", "border": "#a16207", "gradient": "linear-gradient(90deg, #ca8a04, #18181b)", "logo": "https://fantasy.eliteserien.no/img/badges/badge_184.png"},
}

def get_club_style(team_key: str) -> dict:
    """Returnerer klubbfarger, logo og stil for gitt kortnavn eller klubbnavn."""
    default_style = {
        "name": team_key or "Ukjent",
        "primary": "#3b82f6",
        "secondary": "#ffffff",
        "text": "#ffffff",
        "border": "#2563eb",
        "gradient": "linear-gradient(90deg, #3b82f6, #1d4ed8)",
        "logo": "https://fantasy.eliteserien.no/img/badges/badge_4.png"
    }
    if not team_key:
        return default_style
    # Sjekk direkte treff på kortnavn eller navn
    if team_key in CLUB_COLORS:
        return CLUB_COLORS[team_key]
    for k, v in CLUB_COLORS.items():
        if k.lower() == team_key.lower() or v['name'].lower() == team_key.lower():
            return v
    return default_style

def get_event_display_name(event_id: Optional[int], events: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Returnerer det offisielle visningsnavnet for en gameweek fra events-listen (f.eks. 'Runde 21' i stedet for intern id 20).
    Sikrer at event-forskyvninger (f.eks. utsatte kamper) ikke forvirrer brukeren.
    """
    if not event_id:
        return "Ukjent runde"
    if events:
        for ev in events:
            if ev.get('id') == event_id:
                return ev.get('name', f"Runde {event_id}")
    return f"Runde {event_id}"

def get_pitch_card_script() -> str:
    """
    Returnerer et klientskript som fanger opp klikk på spillerkort (bane & benk)
    og trigger samme umiddelbare modal-popout helt uten full siderefresh.
    """
    return """
    <script>
    (function() {
        window.triggerCardInspect = function(pid, pname, pteam, tab) {
            if (!pid) return false;
            // 1. Primær metode: Trigger dedikert usynlig Streamlit-knapp direkte (lynraskt & 100% pålitelig)
            var selList = [
                '.st-key-btn_trigger_tab' + tab + '_' + pid + ' button',
                '.st-key-btn_trigger_tab0_' + pid + ' button',
                '.st-key-btn_trigger_tab1_' + pid + ' button',
                '.st-key-btn_bench_live_' + pid + ' button',
                '.st-key-btn_bench_squad_' + pid + ' button'
            ];
            for (var i = 0; i < selList.length; i++) {
                var btn = document.querySelector(selList[i]);
                if (btn) {
                    btn.click();
                    return true;
                }
            }
            // 2. Sekundær metode: Trigger st.pills med fulle Pointer- og MouseEvents
            var pillKey = (tab === '1') ? 'pills_squad_starters' : 'pills_live_starters';
            var container = document.querySelector('.st-key-' + pillKey) || 
                            document.querySelector('.st-key-pills_live_starters') || 
                            document.querySelector('.st-key-pills_squad_starters');
            if (container) {
                var buttons = container.querySelectorAll('button, [role="button"], [data-variant="pills"]');
                var targetStr = (pname && pteam) ? (pname + ' (' + pteam + ')') : '';
                var matched = null;
                for (var j = 0; j < buttons.length; j++) {
                    var txt = buttons[j].textContent || buttons[j].innerText || '';
                    if (targetStr && txt.indexOf(targetStr) !== -1) {
                        matched = buttons[j];
                        break;
                    }
                    if (!matched && pname && txt.indexOf(pname) !== -1) {
                        matched = buttons[j];
                    }
                }
                if (matched) {
                    var rect = matched.getBoundingClientRect();
                    var opts = { bubbles: true, cancelable: true, view: window, clientX: rect.left + 5, clientY: rect.top + 5 };
                    matched.dispatchEvent(new PointerEvent('pointerdown', opts));
                    matched.dispatchEvent(new MouseEvent('mousedown', opts));
                    matched.dispatchEvent(new PointerEvent('pointerup', opts));
                    matched.dispatchEvent(new MouseEvent('mouseup', opts));
                    matched.dispatchEvent(new MouseEvent('click', opts));
                    matched.click();
                    return true;
                }
            }
            return false;
        };

        function onCardClick(e) {
            var card = e.target.closest('[data-player-id]');
            if (!card) return;
            
            e.preventDefault();
            e.stopPropagation();
            
            var pid = card.getAttribute('data-player-id');
            var pname = card.getAttribute('data-player-name') || '';
            var pteam = card.getAttribute('data-player-team') || '';
            var tab = card.getAttribute('data-tab') || '0';
            
            window.triggerCardInspect(pid, pname, pteam, tab);
        }

        if (!window._pitchCardClickListenerActive) {
            window._pitchCardClickListenerActive = true;
            document.addEventListener('click', onCardClick, true);
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' || e.key === ' ') {
                    var card = e.target.closest('[data-player-id]');
                    if (card) {
                        onCardClick(e);
                    }
                }
            }, true);
        }
    })();
    </script>
    """

# Injiser standard stil ved oppstart basert på aktiv fargemodus i session_state
initial_theme = st.session_state.get('app_theme_mode', '🖥️ Auto (Følg system)')
st.markdown(build_custom_css(initial_theme), unsafe_allow_html=True)


# ==============================================================================
# 2. DATAINNHENTING MED CACHING & ROBUST FEILHÅNDTERING
# ==============================================================================
BASE_URL = "https://fantasy.eliteserien.no/api/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate"
}

def fetch_api_json(endpoint: str) -> Optional[Dict[str, Any]]:
    """Gjør en GET-forespørsel mot Eliteserien Fantasy API med gzip-støtte."""
    url = f"{BASE_URL}{endpoint.lstrip('/')}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=12)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            st.warning(f"API-kall til {endpoint} returnerte statuskode {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Kunne ikke hente data fra {endpoint}: {str(e)}")
        return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_fixtures() -> Optional[List[Dict[str, Any]]]:
    """Henter terminlisten og oppdaterte kampstatuser (60 sek cache for live resultater)."""
    return fetch_api_json("fixtures/")

@st.cache_data(ttl=300, show_spinner=False)
def load_bootstrap_data():
    """Henter spillere, lag og runder fra Eliteserien Fantasy API (5 min cache)."""
    return fetch_api_json("bootstrap-static/")

def load_all_core_data():
    """Henter bootstrap-static og terminlisten (fixtures)."""
    bootstrap = load_bootstrap_data()
    fixtures = fetch_fixtures()
    return bootstrap, fixtures

@st.cache_data(ttl=120, show_spinner=False)
def load_manager_data(team_id: int, current_gw: int):
    """Henter manager-informasjon og aktive picks for gitt lag og gameweek."""
    entry = fetch_api_json(f"entry/{team_id}/")
    picks = None
    if current_gw and current_gw > 0:
        picks = fetch_api_json(f"entry/{team_id}/event/{current_gw}/picks/")
        # Dersom inneværende runde ikke har picks ennå (f.eks. pre-deadline), prøv forrige runde
        if not picks and current_gw > 1:
            picks = fetch_api_json(f"entry/{team_id}/event/{current_gw - 1}/picks/")
    return entry, picks

@st.cache_data(ttl=300, show_spinner=False)
def fetch_league_standings(league_id: int) -> Optional[Dict[str, Any]]:
    """Henter full tabell for en gitt classic-liga fra API-et."""
    return fetch_api_json(f"leagues-classic/{league_id}/standings/")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_manager_history(team_id: int) -> Optional[Dict[str, Any]]:
    """Henter sesonghistorikk for en manager (poeng, rank og verdi per runde)."""
    return fetch_api_json(f"entry/{team_id}/history/")

@st.cache_data(ttl=120, show_spinner=False)
def fetch_manager_transfers(team_id: int) -> Optional[List[Dict[str, Any]]]:
    """Henter offisiell overgangshistorikk for en manager fra Eliteserien API."""
    return fetch_api_json(f"entry/{team_id}/transfers/")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_event_live(event_id: int) -> Optional[Dict[str, Any]]:
    """Henter live hendelses- og poengdata for en gitt gameweek (60 sek cache)."""
    return fetch_api_json(f"event/{event_id}/live/")

@st.cache_data(ttl=60, show_spinner=False)
def fetch_manager_gw_picks(team_id: int, event_id: int) -> Optional[Dict[str, Any]]:
    """Henter managerens laguttak og poeng for en spesifikk gameweek (60 sek cache)."""
    return fetch_api_json(f"entry/{team_id}/event/{event_id}/picks/")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_player_summary(player_id: int) -> Optional[Dict[str, Any]]:
    """Henter full sesonghistorikk, prishistorikk og kamper for en gitt spiller (element-summary)."""
    return fetch_api_json(f"element-summary/{player_id}/")


# ==============================================================================
# 3. MOTSTANDERSTYRKE & FIXTURE DIFFICULTY RATING (FDR)
# ==============================================================================
def calculate_team_strengths_and_fdr(teams_data: List[Dict], fixtures_data: List[Dict]) -> Tuple[Dict[int, float], Dict[int, Dict[int, List[Dict]]]]:
    """
    Beregner motstanderstyrke (1.0 - 5.0) og terminliste-oppsett med FDR.
    Eliteserien API gir ikke alltid native FDR i fixtures; vi kalkulerer
    derfor dynamisk styrke basert på tabellform, poengsnitt og målforskjell.
    """
    teams_dict = {t['id']: t for t in teams_data}
    team_stats = defaultdict(lambda: {'pts': 0, 'gf': 0, 'ga': 0, 'played': 0})
    
    # Summer opp ferdigspilte kamper
    for f in fixtures_data:
        is_fin = f.get('finished', False) or f.get('finished_provisional', False) or (f.get('started', False) and f.get('minutes', 0) >= 90)
        if is_fin and f.get('team_h_score') is not None:
            th, ta = f['team_h'], f['team_a']
            hs, ascore = f['team_h_score'], f['team_a_score']
            team_stats[th]['played'] += 1
            team_stats[ta]['played'] += 1
            team_stats[th]['gf'] += hs
            team_stats[th]['ga'] += ascore
            team_stats[ta]['gf'] += ascore
            team_stats[ta]['ga'] += hs
            if hs > ascore:
                team_stats[th]['pts'] += 3
            elif hs < ascore:
                team_stats[ta]['pts'] += 3
            else:
                team_stats[th]['pts'] += 1
                team_stats[ta]['pts'] += 1
                
    # Beregn poeng per kamp (PPG)
    ppg_dict = {}
    for tid in teams_dict.keys():
        s = team_stats[tid]
        ppg_dict[tid] = (s['pts'] / s['played']) if s['played'] > 0 else 1.2
        
    min_ppg = min(ppg_dict.values()) if ppg_dict else 0.5
    max_ppg = max(ppg_dict.values()) if ppg_dict else 2.5
    range_ppg = max(max_ppg - min_ppg, 0.1)
    
    team_strength = {}
    for tid, ppg in ppg_dict.items():
        # Normaliser mellom 1.5 og 4.8
        score = 1.5 + 3.3 * ((ppg - min_ppg) / range_ppg)
        team_strength[tid] = round(score, 1)

    # Bygg oversikt over kommende kamper per lag og runde
    team_fixtures = defaultdict(lambda: defaultdict(list))
    for f in fixtures_data:
        ev = f.get('event')
        is_fin = f.get('finished', False) or f.get('finished_provisional', False) or (f.get('started', False) and f.get('minutes', 0) >= 90)
        if ev is not None and not is_fin:
            th, ta = f['team_h'], f['team_a']
            # Sjekk om det finnes eksplisitt difficulty i api, ellers bruk vår dynamiske beregning
            fdr_h_raw = f.get('team_h_difficulty')
            fdr_a_raw = f.get('team_a_difficulty')
            
            if fdr_h_raw is not None:
                fdr_h = max(1, min(5, int(fdr_h_raw)))
            else:
                # Hjemmelag møter Team A: Motstanderens styrke minus hjemmefordel (-0.4)
                fdr_h = max(1, min(5, int(round(team_strength.get(ta, 3.0) - 0.4))))
                
            if fdr_a_raw is not None:
                fdr_a = max(1, min(5, int(fdr_a_raw)))
            else:
                # Bortelag møter Team H: Motstanderens styrke pluss borteulempe (+0.4)
                fdr_a = max(1, min(5, int(round(team_strength.get(th, 3.0) + 0.4))))
                
            opp_h = teams_dict.get(ta, {}).get('short_name', f"T{ta}")
            opp_a = teams_dict.get(th, {}).get('short_name', f"T{th}")
            
            team_fixtures[th][ev].append({
                'opp_id': ta, 'opp_name': opp_h, 'is_home': True, 'fdr': fdr_h
            })
            team_fixtures[ta][ev].append({
                'opp_id': th, 'opp_name': opp_a, 'is_home': False, 'fdr': fdr_a
            })
            
    return team_strength, team_fixtures


# ==============================================================================
# 4. PREDIKSJONSMODELL FOR FORVENTEDE POENG (xP)
# ==============================================================================
def compute_player_xp_model(
    bootstrap: Dict,
    team_fixtures: Dict,
    next_gw: int,
    horizon: int
) -> pd.DataFrame:
    """
    Kalkulerer xP for hver spiller over de neste `horizon` rundene basert på:
    - Base form: Vektet snitt av form (65%) og points_per_game (35%)
    - Fixture Difficulty Rating (FDR): Justering med +/- 12% per vanskelighetsgrad
    - Hjemmebanefordel: +12% ved hjemmekamper, -8% borte
    - Offensive underliggende tall: Bonus for målfarlige midtbaner/angripere
    - Skadestatus/spilletid: Skalering ved usikkerhet, 0 ved skade/suspensjon
    """
    teams_dict = {t['id']: t for t in bootstrap['teams']}
    elements = bootstrap['elements']
    pos_baselines = {1: 3.5, 2: 3.8, 3: 4.2, 4: 4.5}
    pos_names = {1: 'KEEPER', 2: 'FORSVAR', 3: 'MIDTBANE', 4: 'ANGREP'}
    pos_short = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}
    
    gws_in_horizon = [next_gw + i for i in range(horizon)]
    player_rows = []
    
    for el in elements:
        pid = el['id']
        tid = el['team']
        pos = el['element_type']
        
        try:
            form_val = float(el.get('form') or 0.0)
        except (ValueError, TypeError):
            form_val = 0.0
            
        try:
            ppg_val = float(el.get('points_per_game') or 0.0)
        except (ValueError, TypeError):
            ppg_val = 0.0
            
        # Base xP
        if form_val <= 0 and ppg_val <= 0:
            base_xp = pos_baselines.get(pos, 3.8)
        elif form_val <= 0:
            base_xp = ppg_val
        else:
            base_xp = 0.65 * form_val + 0.35 * ppg_val
            
        # Skadestatus og tilgjengelighet
        status = el.get('status', 'a')
        chance_next = el.get('chance_of_playing_next_round')
        news_text = (el.get('news') or '').strip()
        
        if status in ['u', 'i', 's'] or chance_next == 0:
            play_chance = 0.0
        elif chance_next is not None:
            play_chance = float(chance_next) / 100.0
        elif status == 'd':
            play_chance = 0.70
        else:
            play_chance = 1.0

        # Menneskelesbar status-beskrivelse for UI
        if status == 'i':
            status_desc = "🔴 Skadet (0%)"
        elif status == 's':
            status_desc = "🟥 Suspendert (0%)"
        elif status == 'u':
            status_desc = "⚪ Utilgjengelig (0%)"
        elif status == 'd':
            pct = int(chance_next) if chance_next is not None else 70
            status_desc = f"🟡 Usikker ({pct}%)"
        else:
            status_desc = "🟢 Frisk"
            
        # Beregn runde-for-runde xP
        gw_xps = []
        fixture_strs = []
        fixture_fdrs = []
        
        for g in gws_in_horizon:
            fixtures_in_gw = team_fixtures[tid].get(g, [])
            if not fixtures_in_gw:
                # Blank gameweek
                gw_xps.append(0.0)
                fixture_strs.append("-")
                fixture_fdrs.append(3)
            else:
                gw_sum = 0.0
                gw_labels = []
                gw_fdr_sum = 0
                for fix in fixtures_in_gw:
                    fdr = fix['fdr']
                    is_h = fix['is_home']
                    gw_fdr_sum += fdr
                    
                    # FDR koeffisient: 3 er nøytral. Hvert trinn gir +/- 12%
                    fdr_mult = 1.0 + (3.0 - fdr) * 0.12
                    # Hjemmebanefordel (+12% vs -8%)
                    venue_mult = 1.12 if is_h else 0.92
                    
                    # Offensiv faktor for angripende spillere
                    goals = float(el.get('goals_scored') or 0)
                    assists = float(el.get('assists') or 0)
                    off_factor = 1.0
                    if pos in [3, 4] and (goals + assists) >= 5:
                        off_factor = 1.06
                        
                    calc_xp = base_xp * fdr_mult * venue_mult * off_factor * play_chance
                    gw_sum += max(0.0, calc_xp)
                    gw_labels.append(f"{fix['opp_name']} ({'H' if is_h else 'B'})")
                    
                gw_xps.append(round(gw_sum, 2))
                fixture_strs.append(" + ".join(gw_labels))
                fixture_fdrs.append(round(gw_fdr_sum / len(fixtures_in_gw)))
                
        next_gw_xp = gw_xps[0] if gw_xps else 0.0
        total_xp = round(sum(gw_xps), 2)
        avg_fdr = round(sum(fixture_fdrs) / len(fixture_fdrs), 1) if fixture_fdrs else 3.0
        
        player_rows.append({
            'id': pid,
            'web_name': el.get('web_name', 'Ukjent'),
            'first_name': el.get('first_name', ''),
            'second_name': el.get('second_name', ''),
            'team_id': tid,
            'team_name': teams_dict.get(tid, {}).get('name', 'Ukjent'),
            'team_short': teams_dict.get(tid, {}).get('short_name', 'LAG'),
            'logo': get_club_style(teams_dict.get(tid, {}).get('short_name', 'LAG')).get('logo', ''),
            'photo_url': f"https://beta.toppfotball.no/Fantasy/players/{el.get('fiks_id')}.png" if el.get('fiks_id') else "",
            'element_type': pos,
            'pos_name': pos_names.get(pos, 'MIDTBANE'),
            'pos_short': pos_short.get(pos, 'MID'),
            'now_cost': float(el.get('now_cost', 50)) / 10.0,
            'cost_raw': int(el.get('now_cost', 50)),
            'selected_by_percent': float(el.get('selected_by_percent') or 0.0),
            'form': form_val,
            'ppg': ppg_val,
            'total_points': int(el.get('total_points', 0)),
            'minutes': int(el.get('minutes', 0)),
            'goals_scored': int(el.get('goals_scored', 0)),
            'assists': int(el.get('assists', 0)),
            'clean_sheets': int(el.get('clean_sheets', 0)),
            'goals_conceded': int(el.get('goals_conceded', 0)),
            'yellow_cards': int(el.get('yellow_cards', 0)),
            'red_cards': int(el.get('red_cards', 0)),
            'bonus': int(el.get('bonus', 0)),
            'dreamteam_count': int(el.get('dreamteam_count', 0)),
            'cost_change_start': int(el.get('cost_change_start', 0)),
            'cost_change_event': int(el.get('cost_change_event', 0)),
            'transfers_in_event': int(el.get('transfers_in_event', 0)),
            'transfers_out_event': int(el.get('transfers_out_event', 0)),
            'transfers_in': int(el.get('transfers_in', 0)),
            'transfers_out': int(el.get('transfers_out', 0)),
            'value_season': float(el.get('value_season') or 0.0),
            'value_form': float(el.get('value_form') or 0.0),
            'status': status,
            'status_desc': status_desc,
            'news': news_text,
            'chance_of_playing': chance_next,
            'play_chance': play_chance,
            'next_gw_xp': next_gw_xp,
            'total_xp': total_xp,
            'gw_xps': gw_xps,
            'next_fixture': fixture_strs[0] if fixture_strs else "-",
            'next_fdr': fixture_fdrs[0] if fixture_fdrs else 3,
            'fixtures_str': " | ".join(fixture_strs),
            'avg_fdr': avg_fdr
        })
        
    return pd.DataFrame(player_rows)


# ==============================================================================
# 5. PuLP OPTIMERINGSMOTOR: LAGUTTAK & OVERGANGER (MILP)
# ==============================================================================
def solve_optimal_starting_xi(squad_df: pd.DataFrame) -> Tuple[List[int], List[int], int, int]:
    """
    Finner optimal startellever og benk fra de 15 spillerne i troppen ved hjelp av PuLP (MILP).
    Inkluderer robust fallback-heuristikk dersom solver-miljøet mangler eller feiler.
    Begrensninger:
    - Nøyaktig 11 i startoppstillingen
    - Nøyaktig 1 Keeper
    - Mellom 3 og 5 Forsvarsspillere
    - Mellom 2 og 5 Midtbanespillere
    - Mellom 1 og 3 Angrepsspillere
    Returnerer: (starter_ids, bench_ids, captain_id, vice_captain_id)
    """
    ids = squad_df['id'].tolist()
    gkps = squad_df[squad_df['element_type'] == 1]['id'].tolist()
    defs = squad_df[squad_df['element_type'] == 2]['id'].tolist()
    mids = squad_df[squad_df['element_type'] == 3]['id'].tolist()
    fwds = squad_df[squad_df['element_type'] == 4]['id'].tolist()

    starters = []
    try:
        prob = pulp.LpProblem("Optimize_Starting_XI", pulp.LpMaximize)
        is_starter = {pid: pulp.LpVariable(f"start_{pid}", cat="Binary") for pid in ids}

        xp_dict = dict(zip(squad_df['id'], squad_df['next_gw_xp']))
        prob += pulp.lpSum(is_starter[pid] * xp_dict.get(pid, 0.0) for pid in ids)

        # Begrensninger
        prob += pulp.lpSum(is_starter[pid] for pid in ids) == 11
        if gkps:
            prob += pulp.lpSum(is_starter[pid] for pid in gkps) == 1
        if defs:
            prob += pulp.lpSum(is_starter[pid] for pid in defs) >= min(3, len(defs))
            prob += pulp.lpSum(is_starter[pid] for pid in defs) <= min(5, len(defs))
        if mids:
            prob += pulp.lpSum(is_starter[pid] for pid in mids) >= min(2, len(mids))
            prob += pulp.lpSum(is_starter[pid] for pid in mids) <= min(5, len(mids))
        if fwds:
            prob += pulp.lpSum(is_starter[pid] for pid in fwds) >= min(1, len(fwds))
            prob += pulp.lpSum(is_starter[pid] for pid in fwds) <= min(3, len(fwds))

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        solved_starters = [pid for pid in ids if is_starter[pid].varValue is not None and is_starter[pid].varValue > 0.5]
        if len(solved_starters) == 11:
            starters = solved_starters
    except Exception:
        starters = []

    # Heuristisk fallback hvis PuLP feiler eller ikke returnerte nøyaktig 11 spillere
    if len(starters) != 11:
        squad_sorted = squad_df.sort_values(by='next_gw_xp', ascending=False)
        gk_sorted = squad_sorted[squad_sorted['element_type'] == 1]['id'].tolist()
        def_sorted = squad_sorted[squad_sorted['element_type'] == 2]['id'].tolist()
        mid_sorted = squad_sorted[squad_sorted['element_type'] == 3]['id'].tolist()
        fwd_sorted = squad_sorted[squad_sorted['element_type'] == 4]['id'].tolist()

        starters = []
        if gk_sorted:
            starters.append(gk_sorted[0])
        starters.extend(def_sorted[:3])
        starters.extend(mid_sorted[:2])
        starters.extend(fwd_sorted[:1])

        rem_candidates = []
        for pid in def_sorted[3:]:
            rem_candidates.append((pid, 2))
        for pid in mid_sorted[2:]:
            rem_candidates.append((pid, 3))
        for pid in fwd_sorted[1:]:
            rem_candidates.append((pid, 4))

        xp_map = dict(zip(squad_df['id'], squad_df['next_gw_xp']))
        rem_candidates.sort(key=lambda x: xp_map.get(x[0], 0.0), reverse=True)

        d_count = len([p for p in starters if p in defs])
        m_count = len([p for p in starters if p in mids])
        f_count = len([p for p in starters if p in fwds])

        for pid, pos in rem_candidates:
            if len(starters) == 11:
                break
            if pos == 2 and d_count < 5:
                starters.append(pid)
                d_count += 1
            elif pos == 3 and m_count < 5:
                starters.append(pid)
                m_count += 1
            elif pos == 4 and f_count < 3:
                starters.append(pid)
                f_count += 1

        # Nødutfylling dersom troppen har uvanlige posisjonsavvik
        for pid in ids:
            if len(starters) == 11:
                break
            if pid not in starters:
                starters.append(pid)

    bench = [pid for pid in ids if pid not in starters]

    # Rangér kaptein og visekaptein blant startere etter neste rundes xP
    starters_df = squad_df[squad_df['id'].isin(starters)].sort_values(by='next_gw_xp', ascending=False)
    captain_id = starters_df.iloc[0]['id'] if len(starters_df) > 0 else (ids[0] if ids else 0)
    vice_captain_id = starters_df.iloc[1]['id'] if len(starters_df) > 1 else captain_id

    # Sorter benken: Reservekeeper først, deretter utespillere etter synkende xP
    bench_gk = [pid for pid in bench if pid in gkps]
    bench_outfield = [pid for pid in bench if pid not in gkps]
    bench_outfield_sorted = squad_df[squad_df['id'].isin(bench_outfield)].sort_values(by='next_gw_xp', ascending=False)['id'].tolist()
    ordered_bench = bench_gk + bench_outfield_sorted

    return starters, ordered_bench, captain_id, vice_captain_id


def solve_transfers_pulp(
    all_players_df: pd.DataFrame,
    current_squad_ids: List[int],
    bank_m: float,
    k_transfers: int,
    teams_dict: Dict[int, Any]
) -> Optional[Dict[str, Any]]:
    """
    Finner de nøyaktig beste K overgangene ved hjelp av PuLP Mixed-Integer Linear Programming.
    Følger alle Eliteserien Fantasy-regler:
    - Troppstørrelse = 15
    - Posisjonskvoter: 2 GKP, 5 DEF, 5 MID, 3 FWD
    - Klubbkvote: Maks 3 spillere fra samme klubb
    - Budsjettbegrensning: Kjøpesum <= Salgssum (ut) + penger i banken
    - Eksakt k spillere byttet ut
    """
    all_ids = all_players_df['id'].tolist()
    current_set = set(current_squad_ids)
    
    current_squad_df = all_players_df[all_players_df['id'].isin(current_squad_ids)]
    current_budget_raw = current_squad_df['cost_raw'].sum() + int(round(bank_m * 10))
    
    try:
        prob = pulp.LpProblem(f"Transfer_Optimizer_{k_transfers}", pulp.LpMaximize)
        
        # Binær variabel y_i: Er spiller i med i den nye troppen?
        y = {pid: pulp.LpVariable(f"squad_{pid}", cat="Binary") for pid in all_ids}
        
        # Målfunksjon: Maksimer total xP over den valgte horisonten
        xp_dict = dict(zip(all_players_df['id'], all_players_df['total_xp']))
        prob += pulp.lpSum(y[pid] * xp_dict[pid] for pid in all_ids)
        
        # Tropp på 15 spillere
        prob += pulp.lpSum(y[pid] for pid in all_ids) == 15
        
        # Posisjonskvoter
        for pos_code, quota in [(1, 2), (2, 5), (3, 5), (4, 3)]:
            pos_ids = all_players_df[all_players_df['element_type'] == pos_code]['id'].tolist()
            prob += pulp.lpSum(y[pid] for pid in pos_ids) == quota
            
        # Maks 3 spillere per Eliteserien-lag
        for tid in teams_dict.keys():
            club_ids = all_players_df[all_players_df['team_id'] == tid]['id'].tolist()
            if club_ids:
                prob += pulp.lpSum(y[pid] for pid in club_ids) <= 3
                
        # Budsjettregel
        cost_dict = dict(zip(all_players_df['id'], all_players_df['cost_raw']))
        prob += pulp.lpSum(y[pid] * cost_dict[pid] for pid in all_ids) <= current_budget_raw
        
        # Nøyaktig k spillere byttet ut fra nåværende tropp
        prob += pulp.lpSum(1 - y[pid] for pid in current_squad_ids) == k_transfers
        
        # Løs problemet
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
        if pulp.LpStatus[status] != 'Optimal':
            return None
            
        new_squad_ids = [pid for pid in all_ids if y[pid].varValue is not None and y[pid].varValue > 0.5]
        transfers_out = [pid for pid in current_squad_ids if pid not in new_squad_ids]
        transfers_in = [pid for pid in new_squad_ids if pid not in current_set]
        
        out_df = all_players_df[all_players_df['id'].isin(transfers_out)]
        in_df = all_players_df[all_players_df['id'].isin(transfers_in)]
        
        xp_out = float(out_df['total_xp'].sum())
        xp_in = float(in_df['total_xp'].sum())
        gross_gain = round(xp_in - xp_out, 2)
        
        new_cost_raw = sum(cost_dict[pid] for pid in new_squad_ids)
        remaining_bank = round((current_budget_raw - new_cost_raw) / 10.0, 2)
        
        return {
            'k': k_transfers,
            'transfers_out': out_df,
            'transfers_in': in_df,
            'xp_out': xp_out,
            'xp_in': xp_in,
            'gross_gain': gross_gain,
            'remaining_bank': remaining_bank,
            'new_squad_ids': new_squad_ids
        }
    except Exception:
        return None


# ==============================================================================
# 5.4 OVERGANGSTRACKER & OPTIMIZER-EVALUERING
# ==============================================================================
TRACKER_FILE = Path(__file__).parent / "transfer_tracker.json"

def load_tracked_transfers() -> List[Dict[str, Any]]:
    """Laster lagrede overganger fra disk for historisk evaluering."""
    try:
        if TRACKER_FILE.exists():
            with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_tracked_transfers(transfers: List[Dict[str, Any]]) -> None:
    """Lagrer overgangslogg til disk."""
    try:
        with open(TRACKER_FILE, "w", encoding="utf-8") as f:
            json.dump(transfers, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def add_tracked_transfer(gw: int, out_id: int, out_name: str, out_team: str,
                         in_id: int, in_name: str, in_team: str,
                         exp_gain: float, hit_cost: int = 0) -> None:
    """Legger til en ny gjennomført overgang i loggen."""
    transfers = load_tracked_transfers()
    tx_id = f"tx_{int(time.time())}_{out_id}_{in_id}"
    rec = {
        "id": tx_id,
        "gw": int(gw),
        "date": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "out_id": int(out_id),
        "out_name": str(out_name),
        "out_team": str(out_team),
        "in_id": int(in_id),
        "in_name": str(in_name),
        "in_team": str(in_team),
        "exp_gain": round(float(exp_gain), 2),
        "hit_cost": int(hit_cost)
    }
    # Unngå duplikat-logging av nøyaktig samme bytte i samme runde
    if not any(t.get('gw') == rec['gw'] and t.get('out_id') == rec['out_id'] and t.get('in_id') == rec['in_id'] for t in transfers):
        transfers.append(rec)
        save_tracked_transfers(transfers)

def delete_tracked_transfer(tx_id: str) -> None:
    """Fjerner en overgang fra loggen."""
    transfers = load_tracked_transfers()
    transfers = [t for t in transfers if t.get('id') != tx_id]
    save_tracked_transfers(transfers)

def get_player_transfer_history(player_id: int, manager_id: Optional[int] = None) -> Dict[str, List[int]]:
    """
    Returnerer hvilke runder en spiller ble kjøpt (grønn) eller solgt (rød),
    både fra offisielle API-overganger og den lokale overgangsloggen.
    """
    bought_gws = set()
    sold_gws = set()

    # 1. Offisielle overganger fra API
    if manager_id and manager_id > 0:
        try:
            api_txs = fetch_manager_transfers(manager_id) or []
            for tx in api_txs:
                gw = tx.get('event')
                if gw:
                    if tx.get('element_in') == player_id:
                        bought_gws.add(int(gw))
                    if tx.get('element_out') == player_id:
                        sold_gws.add(int(gw))
        except Exception:
            pass

    # 2. Lokale overganger fra overgangslogg
    local_txs = load_tracked_transfers()
    for tx in local_txs:
        gw = tx.get('gw')
        if gw:
            if tx.get('in_id') == player_id:
                bought_gws.add(int(gw))
            if tx.get('out_id') == player_id:
                sold_gws.add(int(gw))

    return {
        'bought': sorted(list(bought_gws)),
        'sold': sorted(list(sold_gws))
    }

def evaluate_transfer_record(rec: Dict[str, Any], bootstrap: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluerer det faktiske utfallet av en overgang etter at runden er spilt:
    - Henter faktiske poeng for både spiller UT og spiller INN fra Eliteserien API
    - Beregner faktisk netto gevinst (INN - UT - Hit)
    - Sammenligner med forventet xP-gevinst fra optimizeren
    """
    events = bootstrap.get('events', [])
    gw = int(rec.get('gw', 1))
    ev_obj = next((e for e in events if e.get('id') == gw), None)
    is_finished = ev_obj.get('finished', False) or ev_obj.get('finished_provisional', False) if ev_obj else False
    is_current = ev_obj.get('is_current', False) if ev_obj else False

    in_sum = fetch_player_summary(rec['in_id']) or {}
    out_sum = fetch_player_summary(rec['out_id']) or {}

    in_hist = in_sum.get('history', [])
    out_hist = out_sum.get('history', [])

    in_match = next((h for h in in_hist if h.get('round') == gw), None)
    out_match = next((h for h in out_hist if h.get('round') == gw), None)

    in_pts = in_match.get('total_points') if in_match else (0 if is_finished else None)
    out_pts = out_match.get('total_points') if out_match else (0 if is_finished else None)
    hit = rec.get('hit_cost', 0)

    res = dict(rec)
    res['in_pts'] = in_pts
    res['out_pts'] = out_pts
    res['is_finished'] = is_finished
    res['is_live'] = is_current and not is_finished

    if in_pts is not None and out_pts is not None:
        actual_gain = in_pts - out_pts - hit
        diff = actual_gain - rec.get('exp_gain', 0.0)
        res['actual_gain'] = actual_gain
        res['diff_vs_exp'] = round(diff, 2)
        res['played'] = True
        if actual_gain > 0:
            res['status'] = f"✅ Suksess (+{actual_gain} pts)"
            res['status_icon'] = "🏆"
            res['status_tag'] = "Suksess"
        elif actual_gain == 0:
            res['status'] = "⚖️ Uavgjort (0 pts)"
            res['status_icon'] = "⚖️"
            res['status_tag'] = "Uavgjort"
        else:
            res['status'] = f"❌ Tap ({actual_gain} pts)"
            res['status_icon'] = "⚠️"
            res['status_tag'] = "Tap"
    else:
        res['actual_gain'] = None
        res['diff_vs_exp'] = None
        res['played'] = False
        res['status'] = f"⏳ Venter på runde {gw}"
        res['status_icon'] = "⏳"
        res['status_tag'] = "Kommende"

    return res


# ==============================================================================
# 5.5 INTERAKTIV SPILLERPROFIL & SESONGANALYSE (MODAL DIALOG)
# ==============================================================================
@st.dialog("⚽ Spillerprofil & Dybdeanalyse", width="medium")
def render_player_dialog(player_id: int, df_players: pd.DataFrame, bootstrap: Dict[str, Any], eval_horizon: int, events: List[Dict], team_fixtures: Any, manager_id: Optional[int] = None):
    """
    Rendrer en detaljert spillerprofil-modal med:
    - Bilde, klubb, posisjon, draktfarger og skadestatus
    - Kjøps- og salgsrunder markert med grønn (kjøpt) og rød (solgt) dot
    - Prisutvikling, eierandel og overgangstrend
    - Interaktive Altair-grafer for poeng per runde og pris/eiere over tid
    - Hjemmebane vs. Bortebane sammenligning og effektivitetsstatistikk
    - Kamp-for-kamp historikk med optimaliserte, kompakte kolonnebredder
    - Kommende terminliste med FDR og historiske sesonger
    """
    matching = df_players[df_players['id'] == player_id]
    if matching.empty:
        st.warning(f"Fant ikke spillerdata for spiller-ID {player_id}.")
        return
    p = matching.iloc[0]
    teams_dict = {t['id']: t for t in bootstrap.get('teams', [])}
    club = get_club_style(p['team_short'])

    # Hent kjøps- og salgshistorikk for denne spilleren
    active_mid = manager_id or st.session_state.get('current_manager_id')
    tx_history = get_player_transfer_history(player_id, active_mid)
    bought_gws = tx_history.get('bought', [])
    sold_gws = tx_history.get('sold', [])

    # Hent full sesonghistorikk fra element-summary
    summary = fetch_player_summary(player_id) or {}
    history = summary.get('history', [])
    past_seasons = summary.get('history_past', [])
    upcoming_fixtures = summary.get('fixtures', [])

    # Kompakt integrert spillerkort i toppen av dialogen
    full_name = f"{p.get('first_name', '')} {p.get('second_name', '')}".strip()
    team_name = p.get('team_name', p.get('team_short', ''))
    pos_desc = p.get('pos_desc', p.get('pos_short', ''))

    photo_url = p.get('photo_url', '')
    photo_tag = f'<img src="{photo_url}" style="width:100%; height:100%; object-fit:cover; object-position:top center;" alt="{p["web_name"]}" onerror="this.style.display=\'none\';" />' if photo_url else ''
    logo_tag = f'<img src="{club.get("logo", "")}" style="height:18px; width:18px; object-fit:contain; vertical-align:middle; margin-right:4px;" alt="{p["team_short"]}" />' if club.get("logo") else ''

    status = p.get('status', 'a')
    status_badge = ""
    if status in ['i', 's', 'u']:
        status_badge = f'<span style="background:#fee2e2; color:#991b1b; font-size:11px; font-weight:800; border-radius:6px; padding:2px 7px; border:1px solid #f87171;" title="{p.get("news", "")}">🔴 {p.get("status_desc", "Ute")}: {p.get("news", "")}</span>'
    elif status == 'd':
        status_badge = f'<span style="background:#fef3c7; color:#92400e; font-size:11px; font-weight:800; border-radius:6px; padding:2px 7px; border:1px solid #fcd34d;" title="{p.get("news", "")}">🟡 {p.get("status_desc", "Usikker")}: {p.get("news", "")}</span>'

    dt_count = int(p.get('dreamteam_count', 0))
    dt_badge = f'<span style="background:linear-gradient(135deg,#f59e0b,#d97706); color:white; font-size:11px; font-weight:800; border-radius:6px; padding:2px 7px;">⭐ Ukens Lag ({dt_count}x)</span>' if dt_count > 0 else ""

    tx_badges = []
    for g in bought_gws:
        tx_badges.append(f'<span style="background:#dcfce7; color:#15803d; font-size:11px; font-weight:800; border-radius:6px; padding:2px 7px; border:1px solid #86efac;" title="Kjøpt i Runde {g}">🟢 Kjøpt i GW {g}</span>')
    for g in sold_gws:
        tx_badges.append(f'<span style="background:#fee2e2; color:#b91c1c; font-size:11px; font-weight:800; border-radius:6px; padding:2px 7px; border:1px solid #fca5a5;" title="Solgt i Runde {g}">🔴 Solgt i GW {g}</span>')
    tx_badge_str = " ".join(tx_badges)

    fdr_labels = {
        1: "1/5 (Svært lett)",
        2: "2/5 (Lett)",
        3: "3/5 (Middels)",
        4: "4/5 (Vanskelig)",
        5: "5/5 (Svært tøff)"
    }
    fdr_val = int(p.get('next_fdr', 3))
    fdr_desc = fdr_labels.get(fdr_val, f"{fdr_val}/5")

    header_card_html = f'''
    <div style="background:var(--pitch-card-bg, #1e293b); border:1px solid var(--border-color, #334155); border-radius:14px; padding:10px 14px; margin-bottom:8px; box-shadow:0 4px 14px rgba(0,0,0,0.18);">
        <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px;">
            <div style="display:flex; align-items:center; gap:12px;">
                <div style="width:62px; height:62px; border-radius:50%; overflow:hidden; background:{club['gradient']}; border:2px solid {club['border']}; flex-shrink:0; display:flex; align-items:center; justify-content:center; box-shadow:0 2px 8px rgba(0,0,0,0.25);">
                    {photo_tag}
                </div>
                <div>
                    <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                        <span style="font-size:19px; font-weight:800; color:var(--text-primary);">{p['web_name']}</span>
                        <span style="background:{club['primary']}; color:{club['text']}; border:1px solid {club['border']}; font-size:11px; font-weight:800; border-radius:5px; padding:2px 7px;">{p['team_short']}</span>
                        <span style="background:rgba(128,128,128,0.18); color:var(--text-muted); font-size:11px; font-weight:700; border-radius:5px; padding:2px 7px;">{pos_desc}</span>
                    </div>
                    <div style="font-size:12px; color:var(--text-muted); margin-top:2px;">
                        {logo_tag}<b>{team_name}</b> • {full_name}
                    </div>
                    <div style="display:flex; gap:6px; margin-top:4px; flex-wrap:wrap;">
                        {status_badge}
                        {dt_badge}
                        {tx_badge_str}
                    </div>
                </div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:11px; color:var(--text-muted); text-transform:uppercase; font-weight:700;">Neste Kamp</div>
                <div style="font-size:13px; font-weight:800; color:var(--text-primary);">{p.get('next_fixture', '-')} • <span style="font-size:11px; opacity:0.85;">FDR {fdr_val}/5</span></div>
            </div>
        </div>
    </div>
    '''
    if hasattr(st, "html"):
        st.html(header_card_html)
    else:
        st.markdown(header_card_html, unsafe_allow_html=True)

    # 4 Hovedmetrikker
    m1, m2, m3, m4 = st.columns(4)
    cost_chg = p.get('cost_change_start', 0) / 10.0
    cost_chg_str = f"+{cost_chg:.1f}m" if cost_chg > 0 else (f"{cost_chg:.1f}m" if cost_chg < 0 else "0.0m")
    with m1:
        st.metric("Pris", f"{p.get('now_cost', 0):.1f}m", cost_chg_str)
    with m2:
        st.metric("Totalpoeng", f"{p.get('total_points', 0)} pts", f"{p.get('ppg', 0):.1f} PPG")
    with m3:
        st.metric("Eierandel", f"{p.get('selected_by_percent', 0):.1f}%", f"Form: {p.get('form', '-')}")
    with m4:
        st.metric(f"xP Neste GW", f"{p.get('next_gw_xp', 0):.2f}", f"Motstand: {fdr_desc}")

    # Forbered kamphistorikk-data
    rows = []
    for h in history:
        opp_id = h.get('opponent_team')
        opp_team = teams_dict.get(opp_id, {})
        opp_short = opp_team.get('short_name', f"T{opp_id}")
        venue = "H" if h.get('was_home') else "B"
        score = f"{h.get('team_h_score', 0)}-{h.get('team_a_score', 0)}" if h.get('was_home') else f"{h.get('team_a_score', 0)}-{h.get('team_h_score', 0)}"
        pts = h.get('total_points', 0)
        gw_num = int(h.get('round', 0))

        tx_tag = ""
        if gw_num in bought_gws and gw_num in sold_gws:
            tx_tag = "🟢 Kjøpt / 🔴 Solgt"
        elif gw_num in bought_gws:
            tx_tag = "🟢 Kjøpt"
        elif gw_num in sold_gws:
            tx_tag = "🔴 Solgt"
        else:
            tx_tag = "-"

        if pts >= 8:
            cat = 'Topp (≥8)'
        elif pts >= 5:
            cat = 'God (5-7)'
        elif pts >= 2:
            cat = 'Middels (2-4)'
        else:
            cat = 'Blank (0-1)'

        cs_icon = "✅" if h.get('clean_sheets', 0) > 0 else "-"

        rows.append({
            'GW': f"GW {gw_num}",
            'gw_num': gw_num,
            'Status': tx_tag,
            'Kamp': f"vs {opp_short} ({venue}) [{score}]",
            'Min': h.get('minutes', 0),
            'Mål ⚽': h.get('goals_scored', 0),
            'Ass 👟': h.get('assists', 0),
            'CS 🧤': cs_icon,
            'Bonus ⭐': h.get('bonus', 0),
            '🟨': h.get('yellow_cards', 0),
            '🟥': h.get('red_cards', 0),
            'Pris': round(h.get('value', 0) / 10.0, 1),
            'Pts': pts,
            'Poeng': pts,
            'Eiere': h.get('selected', 0),
            'Kategori': cat,
            'Runde': f"GW {gw_num}"
        })
    df_hist = pd.DataFrame(rows)

    # Faner inne i dialogen
    sub_tab_graphs, sub_tab_split, sub_tab_hist, sub_tab_fix, sub_tab_past = st.tabs([
        "📈 Sesonggrafer",
        "⚔️ Hjemme vs Borte & Nøkkeltall",
        "📋 Kamp-for-Kamp Historikk",
        "📅 Kommende Kamper (FDR)",
        "📜 Tidligere Sesonger"
    ])

    with sub_tab_graphs:
        if not df_hist.empty:
            st.markdown("##### 📊 Poeng per Gameweek")
            bar_chart = alt.Chart(df_hist).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                x=alt.X('GW:O', sort=alt.SortField('gw_num', order='ascending'), title='Gameweek (Runde)'),
                y=alt.Y('Pts:Q', title='Poeng i Runden'),
                color=alt.Color(
                    'Kategori:N',
                    scale=alt.Scale(
                        domain=['Topp (≥8)', 'God (5-7)', 'Middels (2-4)', 'Blank (0-1)'],
                        range=['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
                    ),
                    legend=alt.Legend(title="Ytelse", orient="top")
                ),
                tooltip=[
                    alt.Tooltip('GW:N', title='Runde'),
                    alt.Tooltip('Status:N', title='Status / Overgang'),
                    alt.Tooltip('Kamp:N', title='Kamp'),
                    alt.Tooltip('Pts:Q', title='Poeng'),
                    alt.Tooltip('Min:Q', title='Minutter spilt'),
                    alt.Tooltip('Mål ⚽:Q', title='Mål'),
                    alt.Tooltip('Ass 👟:Q', title='Målgivende'),
                    alt.Tooltip('CS 🧤:N', title='Clean Sheet'),
                    alt.Tooltip('Bonus ⭐:Q', title='Bonuspoeng')
                ]
            ).properties(height=260)

            ppg_val = float(p.get('ppg', 0))
            if ppg_val > 0:
                rule = alt.Chart(pd.DataFrame({'ppg': [ppg_val]})).mark_rule(
                    color='#f97316', strokeDash=[5, 5], size=2
                ).encode(
                    y='ppg:Q',
                    tooltip=[alt.Tooltip('ppg:Q', title='Sesongsnitt (PPG)')]
                )
                bar_chart = bar_chart + rule

            st.altair_chart(bar_chart, width="stretch")
            st.caption(f"💡 Oransje stiplet linje viser spillerens sesongsnitt på **{ppg_val:.1f} poeng per kamp**.")

            st.markdown("---")
            st.markdown("##### 📈 Prisutvikling & Eierandel over Sesongen")
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                # Prisgraf med grønn dot for kjøpt og rød dot for solgt
                price_base = alt.Chart(df_hist).encode(
                    x=alt.X('GW:O', sort=alt.SortField('gw_num', order='ascending'), title='Runde')
                )
                price_line = price_base.mark_line(
                    color='#3b82f6', strokeWidth=3
                ).encode(
                    y=alt.Y('Pris:Q', scale=alt.Scale(zero=False), title='Pris (m kr)')
                )
                price_points = price_base.mark_circle(
                    size=55, color='#1d4ed8'
                ).encode(
                    y=alt.Y('Pris:Q', scale=alt.Scale(zero=False)),
                    tooltip=[
                        alt.Tooltip('GW:N', title='Runde'),
                        alt.Tooltip('Status:N', title='Status / Overgang'),
                        alt.Tooltip('Pris:Q', title='Pris (m)'),
                        alt.Tooltip('Kamp:N', title='Kamp')
                    ]
                )

                df_tx_dots = df_hist[df_hist['Status'] != "-"].copy()
                if not df_tx_dots.empty:
                    df_tx_dots['DotColor'] = df_tx_dots['Status'].apply(
                        lambda s: '#10b981' if 'Kjøpt' in s and 'Solgt' not in s else ('#ef4444' if 'Solgt' in s and 'Kjøpt' not in s else '#f59e0b')
                    )
                    df_tx_dots['DotLabel'] = df_tx_dots['Status'].apply(
                        lambda s: '🟢 Kjøpt' if 'Kjøpt' in s and 'Solgt' not in s else ('🔴 Solgt' if 'Solgt' in s and 'Kjøpt' not in s else '🟢🔴')
                    )

                    tx_dots_chart = alt.Chart(df_tx_dots).mark_circle(
                        size=160, stroke='#ffffff', strokeWidth=2.5, opacity=1
                    ).encode(
                        x=alt.X('GW:O', sort=alt.SortField('gw_num', order='ascending')),
                        y=alt.Y('Pris:Q', scale=alt.Scale(zero=False)),
                        color=alt.Color('DotColor:N', scale=None),
                        tooltip=[
                            alt.Tooltip('GW:N', title='Runde'),
                            alt.Tooltip('Status:N', title='Overgang'),
                            alt.Tooltip('Pris:Q', title='Pris ved overgang (m)'),
                            alt.Tooltip('Kamp:N', title='Kamp')
                        ]
                    )

                    tx_text_chart = alt.Chart(df_tx_dots).mark_text(
                        dy=-14, fontSize=11, fontWeight='bold'
                    ).encode(
                        x=alt.X('GW:O', sort=alt.SortField('gw_num', order='ascending')),
                        y=alt.Y('Pris:Q', scale=alt.Scale(zero=False)),
                        text='DotLabel:N',
                        color=alt.Color('DotColor:N', scale=None),
                        tooltip=[
                            alt.Tooltip('GW:N', title='Runde'),
                            alt.Tooltip('Status:N', title='Overgang'),
                            alt.Tooltip('Pris:Q', title='Pris ved overgang (m)')
                        ]
                    )
                    price_chart = (price_line + price_points + tx_dots_chart + tx_text_chart).properties(height=220)
                else:
                    price_chart = (price_line + price_points).properties(height=220)

                st.altair_chart(price_chart, width="stretch")
                if not df_tx_dots.empty:
                    st.caption("💡 🟢 Grønn dot markerer runden du kjøpte, 🔴 rød dot markerer runden du solgte.")
            with c_p2:
                sel_chart = alt.Chart(df_hist).mark_area(
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='#10b981', offset=0), alt.GradientStop(color='rgba(16, 185, 129, 0.05)', offset=1)],
                        x1=1, x2=1, y1=1, y2=0
                    ),
                    line={'color': '#10b981', 'strokeWidth': 2}
                ).encode(
                    x=alt.X('GW:O', sort=alt.SortField('gw_num', order='ascending'), title='Runde'),
                    y=alt.Y('Eiere:Q', title='Antall Eiere'),
                    tooltip=[alt.Tooltip('GW:N', title='Runde'), alt.Tooltip('Eiere:Q', title='Managere', format=',')]
                ).properties(height=220)
                st.altair_chart(sel_chart, width="stretch")
        else:
            st.info("Ingen spilte kamper registrert for denne spilleren ennå.")

    with sub_tab_split:
        st.markdown("##### 🏠 Hjemmebane vs. ✈️ Bortebane Sammenligning")
        home_m = [h for h in history if h.get('was_home')]
        away_m = [h for h in history if not h.get('was_home')]
        h_pts = sum(h['total_points'] for h in home_m)
        a_pts = sum(h['total_points'] for h in away_m)
        h_avg = (h_pts / len(home_m)) if home_m else 0.0
        a_avg = (a_pts / len(away_m)) if away_m else 0.0
        h_g = sum(h['goals_scored'] for h in home_m)
        a_g = sum(h['goals_scored'] for h in away_m)
        h_a = sum(h['assists'] for h in home_m)
        a_a = sum(h['assists'] for h in away_m)
        h_cs = sum(h['clean_sheets'] for h in home_m)
        a_cs = sum(h['clean_sheets'] for h in away_m)
        h_bon = sum(h['bonus'] for h in home_m)
        a_bon = sum(h['bonus'] for h in away_m)

        split_df = pd.DataFrame([
            {'Arena': '🏠 Hjemmebane', 'Kamper': len(home_m), 'Poeng': h_pts, 'Snitt PPG': f"{h_avg:.2f}", 'Mål ⚽': h_g, 'Assists 👟': h_a, 'Clean Sheets 🧤': h_cs, 'Bonus ⭐': h_bon},
            {'Arena': '✈️ Bortebane', 'Kamper': len(away_m), 'Poeng': a_pts, 'Snitt PPG': f"{a_avg:.2f}", 'Mål ⚽': a_g, 'Assists 👟': a_a, 'Clean Sheets 🧤': a_cs, 'Bonus ⭐': a_bon}
        ])
        st.dataframe(split_df, hide_index=True, width="stretch")

        st.markdown("---")
        st.markdown("##### ⚡ Effektivitetsstatistikk & Nøkkeltall")
        mins = int(p.get('minutes', 0) or sum(h.get('minutes', 0) for h in history))
        tot_pts = int(p.get('total_points', 0) or sum(h.get('total_points', 0) for h in history))
        goals = int(p.get('goals_scored', 0) or sum(h.get('goals_scored', 0) for h in history))
        assists = int(p.get('assists', 0) or sum(h.get('assists', 0) for h in history))
        ga = goals + assists
        min_per_ga = f"{mins / ga:.1f} min" if ga > 0 else "N/A"
        min_per_pt = f"{mins / tot_pts:.1f} min" if tot_pts > 0 else "N/A"
        cs = int(p.get('clean_sheets', 0) or sum(h.get('clean_sheets', 0) for h in history))
        gc = int(p.get('goals_conceded', 0) or sum(h.get('goals_conceded', 0) for h in history))
        yc = int(p.get('yellow_cards', 0) or sum(h.get('yellow_cards', 0) for h in history))
        rc = int(p.get('red_cards', 0) or sum(h.get('red_cards', 0) for h in history))

        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        with s_c1:
            st.metric("Målpoeng (Mål + Assist)", f"{ga}", f"{goals} mål • {assists} ass")
            st.metric("Spilte Minutter", f"{mins}'")
        with s_c2:
            st.metric("Minutter per Målpoeng", min_per_ga)
            st.metric("Minutter per Poeng", min_per_pt)
        with s_c3:
            st.metric("Clean Sheets 🧤", f"{cs}")
            st.metric("Mål sluppet inn 🥅", f"{gc}")
        with s_c4:
            in_ev = int(p.get('transfers_in_event', 0))
            out_ev = int(p.get('transfers_out_event', 0))
            net_tr = in_ev - out_ev
            st.metric("Kjøpt/Solgt denne runden", f"{net_tr:+d}", f"+{in_ev} inn / -{out_ev} ut")
            st.metric("Kort", f"🟨 {yc}  🟥 {rc}")

    with sub_tab_hist:
        if not df_hist.empty:
            st.markdown("##### 📋 Full Kamphistorikk per Gameweek")
            table_cols = ['GW', 'Status', 'Kamp', 'Min', 'Mål ⚽', 'Ass 👟', 'CS 🧤', 'Bonus ⭐', '🟨', '🟥', 'Pris', 'Pts']
            st.dataframe(
                df_hist[table_cols],
                column_config={
                    "GW": st.column_config.TextColumn("GW", width=55),
                    "Status": st.column_config.TextColumn("Status", width=85),
                    "Kamp": st.column_config.TextColumn("Kamp", width=125),
                    "Min": st.column_config.NumberColumn("Min", width=45),
                    "Mål ⚽": st.column_config.NumberColumn("Mål ⚽", width=50),
                    "Ass 👟": st.column_config.NumberColumn("Ass 👟", width=50),
                    "CS 🧤": st.column_config.TextColumn("CS 🧤", width=50),
                    "Bonus ⭐": st.column_config.NumberColumn("Bonus ⭐", width=55),
                    "🟨": st.column_config.NumberColumn("🟨", width=40),
                    "🟥": st.column_config.NumberColumn("🟥", width=40),
                    "Pris": st.column_config.NumberColumn("Pris", format="%.1f m", width=55),
                    "Pts": st.column_config.NumberColumn("Pts", width=45),
                },
                hide_index=True
            )
        else:
            st.info("Ingen kamphistorikk tilgjengelig.")

    with sub_tab_fix:
        if upcoming_fixtures:
            st.markdown("##### 📅 Kommende Kamper & Motstandere")
            fix_rows = []
            for f in upcoming_fixtures[:8]:
                opp_id = f.get('team_a') if f.get('is_home') else f.get('team_h')
                opp_short = teams_dict.get(opp_id, {}).get('short_name', f"T{opp_id}")
                venue = "H" if f.get('is_home') else "B"
                k_date = f.get('kickoff_time', '')[:10]
                k_time = f.get('kickoff_time', '')[11:16]
                diff = f.get('difficulty', 3)
                ev_id = f.get('event')
                status_tx = "-"
                if ev_id in bought_gws and ev_id in sold_gws:
                    status_tx = "🟢 Kjøpt / 🔴 Solgt"
                elif ev_id in bought_gws:
                    status_tx = "🟢 Kjøpt"
                elif ev_id in sold_gws:
                    status_tx = "🔴 Solgt"
                fix_rows.append({
                    'Runde': f"GW {ev_id}",
                    'Status': status_tx,
                    'Motstander': f"{opp_short} ({venue})",
                    'Dato': k_date,
                    'Tid': k_time,
                    'FDR': diff
                })
            st.dataframe(pd.DataFrame(fix_rows), hide_index=True, width="stretch")
        else:
            st.info("Ingen kommende kamper registrert.")

    with sub_tab_past:
        if past_seasons:
            st.markdown("##### 📜 Tidligere Sesonger i Eliteserien")
            past_rows = []
            for ps in past_seasons:
                past_rows.append({
                    'Sesong': ps.get('season_name', '?'),
                    'Totalpoeng': ps.get('total_points', 0),
                    'Minutter': ps.get('minutes', 0),
                    'Mål': ps.get('goals_scored', 0),
                    'Assists': ps.get('assists', 0),
                    'Clean Sheets': ps.get('clean_sheets', 0)
                })
            st.dataframe(pd.DataFrame(past_rows), hide_index=True, width="stretch")
        else:
            st.info("Ingen tidligere sesongstatistikk tilgjengelig for denne spilleren.")


# ==============================================================================
# 6. HOVEDAPPLIKASJON (STREAMLIT APP)
# ==============================================================================
def main():
    # Last inn kjernedata
    with st.spinner("Henter data fra Eliteserien Fantasy API..."):
        bootstrap, fixtures = load_all_core_data()
        
    if not bootstrap or not fixtures:
        st.error("Kunne ikke koble til Eliteserien Fantasy API. Vennligst sjekk internettforbindelsen eller prøv igjen om et øyeblikk.")
        return

    # Finn nåværende og neste runde automatisk
    events = bootstrap.get('events', [])
    current_gw = next((ev['id'] for ev in events if ev.get('is_current')), None)
    next_gw = next((ev['id'] for ev in events if ev.get('is_next')), None)
    if not current_gw:
        # Hvis ingen runde er aktiv (f.eks. pre-season eller mellom runder)
        current_gw = 1
    if not next_gw:
        next_gw = (current_gw + 1) if current_gw < len(events) else current_gw

    # SIDEBAR: Brukerinput og innstillinger
    st.sidebar.markdown('<div class="sidebar-title">🎨 UTSEENDE & FARGER</div>', unsafe_allow_html=True)
    theme_options = ["🖥️ Auto (Følg system)", "☀️ Lys", "🌙 Mørk"]
    cur_theme = st.session_state.get('app_theme_mode', "🖥️ Auto (Følg system)")
    theme_choice = st.sidebar.radio(
        "Velg fargemodus",
        options=theme_options,
        index=theme_options.index(cur_theme) if cur_theme in theme_options else 0,
        horizontal=True,
        key="app_theme_mode",
        label_visibility="collapsed"
    )
    # Oppdater CSS dynamisk med valgt fargemodus
    st.markdown(build_custom_css(theme_choice), unsafe_allow_html=True)

    # Sjekk query parameters for manager_id og embed (f.eks. fra ranheimkraft.no iframe embed)
    query_id = st.query_params.get("id") or st.query_params.get("manager_id")
    init_manager_id = 7861
    if query_id:
        try:
            init_manager_id = int(query_id)
        except (ValueError, TypeError):
            pass
    elif "saved_manager_id" in st.session_state:
        init_manager_id = int(st.session_state["saved_manager_id"])

    # Embed-modus tilpasninger hvis appen vises i iframe på ranheimkraft.no
    is_embedded = st.query_params.get("embed") in ["true", "True", "1"]
    if is_embedded:
        st.markdown(
            """
            <style>
            header[data-testid="stHeader"] {
                display: none !important;
            }
            div.block-container {
                padding-top: 0.5rem !important;
                padding-bottom: 1.5rem !important;
            }
            footer {
                display: none !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

    st.sidebar.markdown('<div style="margin-top:16px;" class="sidebar-title">👤 DITT LAG & IDENTIFIKATOR</div>', unsafe_allow_html=True)
    manager_id = st.sidebar.number_input(
        "Manager ID (Team ID)",
        min_value=1,
        max_value=999999,
        value=init_manager_id,
        step=1,
        help="Skriv inn din Eliteserien Fantasy ID (standard: 7861)"
    )
    st.session_state["saved_manager_id"] = manager_id
    
    free_transfers = st.sidebar.selectbox(
        "Tilgjengelige gratisbytter",
        options=[1, 2],
        index=0,
        help="Hvor mange gratis overganger du har tilgjengelig før neste runde"
    )
    
    st.sidebar.markdown('<div style="margin-top:16px;" class="sidebar-title">⚙️ OPTIMERINGSPARAMETRE</div>', unsafe_allow_html=True)
    eval_horizon = st.sidebar.slider(
        "Tidshorisont for xP (Runder)",
        min_value=1,
        max_value=5,
        value=3,
        help="Antall kommende runder som legges til grunn for forventede poeng (xP)"
    )
    
    allow_hits = st.sidebar.checkbox(
        "Tillat -4 poengstraff (Hits)",
        value=False,
        help="Undersøk om ekstra overganger utover gratisbytter (-4 poengstraff) lønner seg matematisk"
    )
    
    if st.sidebar.button("🔄 Tving oppdatering av data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    st.sidebar.markdown('<div style="margin-top:16px;" class="sidebar-title">💡 INFORMASJON & REGLER</div>', unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **💡 Om optimeringsmotoren:**
        * **PuLP MILP:** Løser laguttak og spillerbytter matematisk til globalt optimum.
        * **xP-modell:** Inkluderer form, dynamisk FDR, hjemmebanefordel (+12%), offensive faktorer og skadesannsynlighet.
        * **Hit-evaluering:** Netto gevinst = xP(Inn) - xP(Ut) - 4.0.
        """
    )
    with st.sidebar.expander("ℹ️ Hvorfor vises ikke byttet mitt straks?"):
        st.markdown(
            """
            **Offisiell Eliteserien Fantasy-regel:**
            Eliteseriens åpne API publiserer **aldri** nye overganger før selve fristen (*deadline*) for runden har stengt. Dette gjøres for å hindre at konkurrenter i ligaene dine kan spionere på byttene dine på forhånd!
            
            👉 **Løsning:** 
            Bruk den grønne knappen **"✅ Registrer dette byttet i appen"** i *Transfer Hub*, så oppdateres banen, formasjonen og budsjettet ditt her umiddelbart!
            """
        )

    # Beregn team strengths, FDR og xP for alle spillere
    teams_data = bootstrap.get('teams', [])
    teams_dict = {t['id']: t for t in teams_data}
    team_strengths, team_fixtures = calculate_team_strengths_and_fdr(teams_data, fixtures)
    
    df_players = compute_player_xp_model(bootstrap, team_fixtures, next_gw, eval_horizon)

    # Hent managerdata
    entry, picks_data = load_manager_data(manager_id, current_gw)
    
    if not entry:
        st.error(f"Fant ikke manager med ID `{manager_id}`. Kontroller at ID-en er riktig.")
        return

    # Topp-header
    manager_name = f"{entry.get('player_first_name', '')} {entry.get('player_last_name', '')}"
    team_name = entry.get('name', 'Mitt Lag')
    overall_pts = entry.get('summary_overall_points', 0)
    overall_rank = entry.get('summary_overall_rank', 0)
    
    history = picks_data.get('entry_history', {}) if picks_data else {}
    bank_m = history.get('bank', entry.get('last_deadline_bank', 0)) / 10.0
    team_val_m = history.get('value', entry.get('last_deadline_value', 1000)) / 10.0

    hero_banner_html = f"""
    <div class="hero-banner">
        <div class="hero-badge">⚽ ELITESERIEN FANTASY 2026 • AI & MILP OPTIMIZER</div>
        <div class="hero-title">Fantasy Predictor & Transfer Optimizer</div>
        <div class="hero-pills">
            <div class="hero-pill">👤 Manager: <b>{manager_name}</b></div>
            <div class="hero-pill">🛡️ Lag: <b>{team_name}</b></div>
            <div class="hero-pill">⚡ Aktiv Runde: <b>GW {current_gw}</b></div>
            <div class="hero-pill">🎯 Neste Runde: <b>GW {next_gw}</b> ({eval_horizon} runder horisont)</div>
        </div>
    </div>
    """
    if hasattr(st, "html"):
        st.html(hero_banner_html)
    else:
        st.markdown(hero_banner_html, unsafe_allow_html=True)

    # Identifiser managerens tropp fra API
    if not picks_data or 'picks' not in picks_data:
        st.warning(f"Fant ingen registrerte spillere for GW {current_gw} for manager ID {manager_id}.")
        return

    def _get_active_transfers_from_tracker(gw_cutoff: int) -> List[Tuple[int, int]]:
        tracked = load_tracked_transfers()
        res = []
        for t in tracked:
            if t.get('gw', 0) >= gw_cutoff:
                tup = (int(t['out_id']), int(t['in_id']))
                if tup not in res:
                    res.append(tup)
        return res

    if 'applied_transfers' not in st.session_state:
        st.session_state['applied_transfers'] = _get_active_transfers_from_tracker(current_gw)

    if 'current_manager_id' not in st.session_state or st.session_state['current_manager_id'] != manager_id:
        st.session_state['current_manager_id'] = manager_id
        st.session_state['applied_transfers'] = _get_active_transfers_from_tracker(current_gw)

    # Start med tropp og bank fra API
    squad_pids = [p['element'] for p in picks_data['picks']]
    active_bank = bank_m

    # Påfør lokale overganger som brukeren har utført
    for out_id, in_id in st.session_state['applied_transfers']:
        if out_id in squad_pids:
            squad_pids.remove(out_id)
            squad_pids.append(in_id)
            p_out = df_players[df_players['id'] == out_id].iloc[0]
            p_in = df_players[df_players['id'] == in_id].iloc[0]
            active_bank += (p_out['now_cost'] - p_in['now_cost'])

    bank_m = max(0.0, round(active_bank, 2))
    squad_df = df_players[df_players['id'].isin(squad_pids)].copy()

    # Nøkkeltall
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    with col_m1:
        st.metric("Total Poeng", f"{overall_pts} pts")
    with col_m2:
        st.metric("Total Rangering", f"#{overall_rank:,}".replace(",", " "))
    with col_m3:
        st.metric("Penger i Bank", f"{bank_m:.1f} mill")
    with col_m4:
        st.metric("Total Lagverdi", f"{team_val_m:.1f} mill")
    with col_m5:
        # Trekk fra brukte bytter hvis noen er registrert
        rem_free = max(0, free_transfers - len(st.session_state['applied_transfers']))
        st.metric("Gratisbytter Igjen", f"{rem_free} stk")

    # Informasjonsbanner dersom bytter er utført lokalt
    if st.session_state['applied_transfers']:
        applied_names = []
        for out_id, in_id in st.session_state['applied_transfers']:
            o_name = df_players.loc[df_players['id'] == out_id, 'web_name'].values[0]
            i_name = df_players.loc[df_players['id'] == in_id, 'web_name'].values[0]
            applied_names.append(f"❌ {o_name} ➔ ✅ {i_name}")
            
        c_note, c_reset = st.columns([4, 1])
        with c_note:
            st.info(f"🔄 **Registrerte overganger i appen:** {' | '.join(applied_names)}  \n*(Eliteseriens offisielle API holder bytter hemmelige frem til fristen, så de oppdateres her lokalt i appen!)*")
        with c_reset:
            if st.button("Tilbakestill til API", width="stretch"):
                st.session_state['applied_transfers'] = []
                st.rerun()

    # Sjekk query parameters for åpning av spillerprofil
    if "inspect_player" in st.query_params:
        try:
            st.session_state["modal_player_id"] = int(st.query_params["inspect_player"])
        except Exception:
            pass
        if "tab" in st.query_params:
            try:
                t_idx = int(st.query_params["tab"])
                avail_tab_labels = [
                    "⚡ Denne Rundens Poeng (Live)",
                    "⚽ Mitt Lag & Neste Runde",
                    "🔄 Transfer Hub (Beste Kjøp & Salg)",
                    "📊 Spillerdatabase / Kalkulator",
                    "🏆 Mine Ligaer & Tabeller",
                    "📈 Grafer & Statistikk"
                ]
                if 0 <= t_idx < len(avail_tab_labels):
                    st.session_state["active_tab_label"] = avail_tab_labels[t_idx]
            except Exception:
                pass
        del st.query_params["inspect_player"]
        if "tab" in st.query_params:
            del st.query_params["tab"]

    # FANE-STRUKTUR
    tab_labels = [
        "⚡ Denne Rundens Poeng (Live)",
        "⚽ Mitt Lag & Neste Runde",
        "🔄 Transfer Hub (Beste Kjøp & Salg)",
        "📊 Spillerdatabase / Kalkulator",
        "🏆 Mine Ligaer & Tabeller",
        "📈 Grafer & Statistikk"
    ]
    cur_default_tab = st.session_state.get("active_tab_label", tab_labels[0])
    tab_live, tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_labels, default=cur_default_tab)

    # Åpne spillerprofil-dialog hvis en spiller er valgt
    if st.session_state.get("modal_player_id"):
        target_pid = st.session_state["modal_player_id"]
        st.session_state["modal_player_id"] = None
        render_player_dialog(target_pid, df_players, bootstrap, eval_horizon, events, team_fixtures, manager_id=manager_id)

    # ==========================================================================
    # FANE 0: DENNE RUNDENS POENG (LIVE & KAMPSTATUS)
    # ==========================================================================
    with tab_live:
        st.subheader("⚡ Denne Rundens Poeng (Live & Kampstatus)")
        st.caption("Følg dine faktiske poeng, kapteinsdobling, spilte minutter, mål, målgivende og gjenstående spillere i sanntid.")

        with st.container(border=True):
            col_gw_sel, col_live_ref = st.columns([3, 1])
            with col_gw_sel:
                gw_options = list(range(max(1, current_gw), 0, -1))
                selected_live_gw = st.selectbox(
                    "⚡ Velg Gameweek for Live Resultater:",
                    options=gw_options,
                    index=0,
                    format_func=lambda x: f"{get_event_display_name(x, events)} (Inneværende live-runde)" if x == current_gw else f"{get_event_display_name(x, events)} (Avsluttet)",
                    key="live_selected_gw"
                )
            with col_live_ref:
                st.write("")
                st.write("")
                if st.button("🔄 Oppdater live poeng nå", width="stretch"):
                    st.cache_data.clear()
                    st.rerun()

        live_gw_name = get_event_display_name(selected_live_gw, events)
        if selected_live_gw == current_gw:
            st.info(
                f"⚡ **Aktiv visning: {live_gw_name} (Inneværende live-runde).**  \n"
                f"Viser oppdaterte live-poeng, kapteinsdobling, spilte minutter og scoringer i sanntid.  \n"
                f"*(For å planlegge bytter, kaptein og lagoppstilling for neste runde ({get_event_display_name(next_gw, events)}), gå til fanen **'⚽ Mitt Lag & Neste Runde'**!)*"
            )
        else:
            st.success(
                f"📜 **Aktiv visning: {live_gw_name} (Ferdigspilt historisk runde).**  \n"
                f"Viser ditt faktiske laguttak og oppnådde poengsum for denne runden. For å se live-runden igjen, velg **{get_event_display_name(current_gw, events)}** i menyen over."
            )

        # Hent picks og live data for valgt gameweek
        with st.spinner(f"Henter data for {live_gw_name}..."):
            gw_picks_data = fetch_manager_gw_picks(manager_id, selected_live_gw)
            gw_live_data = fetch_event_live(selected_live_gw)

        if not gw_picks_data or not gw_picks_data.get('picks'):
            st.info(f"Fant ingen laguttak registrert for {live_gw_name}.")
        else:
            live_elements_map = {el['id']: el for el in gw_live_data.get('elements', [])} if gw_live_data else {}
            players_dict_all = {p['id']: p for p in bootstrap.get('elements', [])}
            teams_dict_all = {t['id']: t for t in bootstrap.get('teams', [])}

            # Kamper i denne runden (bruk ferske fixtures for umiddelbare live oppdateringer)
            fresh_fixtures = fetch_fixtures() or fixtures
            gw_fixtures_list = [f for f in fresh_fixtures if f.get('event') == selected_live_gw]
            team_live_matches = {}
            for f in gw_fixtures_list:
                th, ta = f['team_h'], f['team_a']
                started = f.get('started', False)
                mins = f.get('minutes', 0)
                finished = f.get('finished', False) or f.get('finished_provisional', False) or (started and mins >= 90)
                hs = f.get('team_h_score')
                ascore = f.get('team_a_score')
                
                th_short = teams_dict_all.get(th, {}).get('short_name', '?')
                ta_short = teams_dict_all.get(ta, {}).get('short_name', '?')
                
                if finished:
                    st_text = "Ferdig"
                    score_str = f"{hs} - {ascore}"
                elif started:
                    st_text = "I gang"
                    score_str = f"{hs} - {ascore}"
                else:
                    st_text = "Ikke startet"
                    score_str = "vs"
                    
                team_live_matches[th] = {'opp': ta_short, 'score': score_str, 'status': st_text, 'is_home': True, 'minutes': mins}
                team_live_matches[ta] = {'opp': th_short, 'score': score_str, 'status': st_text, 'is_home': False, 'minutes': mins}

            # Bygg spillerdata
            gw_history = gw_picks_data.get('entry_history', {})
            gw_chip = gw_picks_data.get('active_chip')
            
            # Formater chip-navn
            chip_names = {
                'wildcard': '🃏 Wildcard',
                'freehit': '⚡ Rikskringkasting / Free Hit',
                '3xc': '👑 Triple Kaptein (3x)',
                'bboost': '🪑 Benkespurt (Bench Boost)',
                'rich': '💰 Riksbonanza',
                'two_captains': '👑👑 To Kapteiner'
            }
            chip_label = chip_names.get(gw_chip, f"Aktiv Chip: {gw_chip}") if gw_chip else "Ingen chip aktiv"

            live_rows = []
            starters_calc_pts = 0
            bench_calc_pts = 0
            starters_done_count = 0
            starters_in_progress_count = 0
            starters_yet_to_play_count = 0

            for p in gw_picks_data.get('picks', []):
                pid = p['element']
                pos = p['position']
                mult = p['multiplier']
                is_c = p['is_captain']
                is_vc = p['is_vice_captain']

                p_inf = players_dict_all.get(pid, {})
                p_liv = live_elements_map.get(pid, {})
                st_data = p_liv.get('stats', {})

                tid = p_inf.get('team')
                t_short = teams_dict_all.get(tid, {}).get('short_name', '???')
                m_info = team_live_matches.get(tid, {'opp': '?', 'score': 'vs', 'status': 'Ikke startet', 'is_home': True})

                raw_pts = st_data.get('total_points', 0)
                effective_pts = raw_pts * mult
                mins = st_data.get('minutes', 0)
                is_bench_player = (mult == 0)

                if not is_bench_player:
                    starters_calc_pts += effective_pts
                    if m_info['status'] == 'Ferdig':
                        starters_done_count += 1
                    elif m_info['status'] == 'I gang':
                        starters_in_progress_count += 1
                    else:
                        starters_yet_to_play_count += 1
                else:
                    bench_calc_pts += raw_pts

                # Hendelser
                events_list = []
                if st_data.get('goals_scored', 0) > 0:
                    events_list.append(f"⚽ {st_data['goals_scored']}")
                if st_data.get('assists', 0) > 0:
                    events_list.append(f"👟 {st_data['assists']}")
                if st_data.get('clean_sheets', 0) > 0 and p_inf.get('element_type') in [1, 2]:
                    events_list.append("🧤 CS")
                if st_data.get('bonus', 0) > 0:
                    events_list.append(f"⭐ +{st_data['bonus']}")
                if st_data.get('yellow_cards', 0) > 0:
                    events_list.append("🟨")
                if st_data.get('red_cards', 0) > 0:
                    events_list.append("🟥")
                if st_data.get('penalties_saved', 0) > 0:
                    events_list.append("🥅 Str.redn.")
                if st_data.get('own_goals', 0) > 0:
                    events_list.append("⚠️ Selvmål")
                events_str = " ".join(events_list) if events_list else "-"

                pos_map = {1: 'GK', 2: 'DEF', 3: 'MID', 4: 'FWD'}
                fiks_id = p_inf.get('fiks_id')
                p_photo = f"https://beta.toppfotball.no/Fantasy/players/{fiks_id}.png" if fiks_id else ""
                live_rows.append({
                    'id': pid,
                    'pos_order': pos,
                    'web_name': p_inf.get('web_name', 'Ukjent'),
                    'team_short': t_short,
                    'pos_short': pos_map.get(p_inf.get('element_type'), '?'),
                    'element_type': p_inf.get('element_type', 1),
                    'multiplier': mult,
                    'is_cap': is_c,
                    'is_vc': is_vc,
                    'is_bench': is_bench_player,
                    'minutes': mins,
                    'raw_pts': raw_pts,
                    'effective_pts': effective_pts,
                    'events_str': events_str,
                    'goals': st_data.get('goals_scored', 0),
                    'assists': st_data.get('assists', 0),
                    'clean_sheets': st_data.get('clean_sheets', 0),
                    'bonus': st_data.get('bonus', 0),
                    'match_status': m_info['status'],
                    'match_desc': f"{m_info['opp']} ({'H' if m_info['is_home'] else 'B'})",
                    'match_score': m_info['score'],
                    'photo_url': p_photo
                })

            df_live_all = pd.DataFrame(live_rows)
            df_live_starters = df_live_all[~df_live_all['is_bench']].sort_values(by=['element_type', 'pos_order'])
            df_live_bench = df_live_all[df_live_all['is_bench']].sort_values(by='pos_order')

            hit_cost = gw_history.get('event_transfers_cost', 0)
            net_round_pts = starters_calc_pts - hit_cost

            # Ligasnitt for denne runden
            event_obj = next((ev for ev in bootstrap.get('events', []) if ev['id'] == selected_live_gw), {})
            gw_avg_score = event_obj.get('average_entry_score', 0)
            diff_from_avg = net_round_pts - gw_avg_score

            # Nøkkeltall-kort
            col_l1, col_l2, col_l3, col_l4, col_l5 = st.columns(5)
            with col_l1:
                hit_tag = f" (-{hit_cost} hit)" if hit_cost > 0 else ""
                st.metric("Rundens Poeng (Netto)", f"{net_round_pts} pts{hit_tag}", f"{starters_calc_pts} råpoeng")
            with col_l2:
                r_rank = gw_history.get('rank')
                r_rank_str = f"#{r_rank:,}".replace(",", " ") if r_rank else "Beregnes..."
                st.metric("Rundens Rangering", r_rank_str)
            with col_l3:
                st.metric(
                    "Spillere i Aksjon",
                    f"{starters_done_count + starters_in_progress_count} / 11",
                    f"{starters_yet_to_play_count} gjenstår ⏳" if starters_yet_to_play_count > 0 else "Alle har spilt! ✅"
                )
            with col_l4:
                diff_sym = "+" if diff_from_avg >= 0 else ""
                st.metric("Mot Ligasnittet", f"{diff_sym}{diff_from_avg} pts", f"Snitt: {gw_avg_score} pts")
            with col_l5:
                st.metric("Poeng på Benk", f"{bench_calc_pts} pts", chip_label)

            # Visningsveksler
            st.markdown("---")
            # Hurtigsøk etter spiller for profil & analyse
            with st.expander("🔍 Søk opp og analyser en hvilken som helst spiller i Eliteserien", expanded=False):
                c_sel_p0, c_btn_p0 = st.columns([3, 1])
                with c_sel_p0:
                    sorted_p0 = df_players.sort_values(by='web_name')
                    chosen_p0 = st.selectbox(
                        "Velg spiller for dybdeanalyse:",
                        options=sorted_p0['id'].tolist(),
                        format_func=lambda x: f"{df_players.loc[df_players['id']==x, 'web_name'].values[0]} ({df_players.loc[df_players['id']==x, 'team_short'].values[0]} - {df_players.loc[df_players['id']==x, 'pos_short'].values[0]})",
                        key="quick_inspect_player_tab0"
                    )
                with c_btn_p0:
                    st.write("")
                    st.write("")
                    if st.button("Åpne profil & grafer", key="btn_open_quick_inspect_tab0", width="stretch"):
                        render_player_dialog(chosen_p0, df_players, bootstrap, eval_horizon, events, team_fixtures, manager_id=manager_id)

            c_head, c_view = st.columns([2, 1])
            with c_head:
                st.markdown(f"### 🏟️ Live Startellever — {live_gw_name}")
            with c_view:
                live_view_mode = st.radio("Format:", ["🏟️ Fotballbane", "📋 Tabelloversikt"], horizontal=True, label_visibility="collapsed")

            # Umiddelbar spillerprofil-velger for startellever og benk (uten sidelasting)
            def _on_live_pill_select():
                sel = st.session_state.get("pills_live_starters")
                if sel:
                    st.session_state["modal_player_id"] = sel
                    st.session_state["active_tab_label"] = tab_labels[0]
                    st.session_state["pills_live_starters"] = None

            live_all_pids = df_live_starters['id'].tolist() + df_live_bench['id'].tolist()
            st.pills(
                "🔍 Klikk for umiddelbar profil & grafer (eller trykk på kortet):",
                options=live_all_pids,
                format_func=lambda pid: f"{df_live_all.loc[df_live_all['id']==pid, 'web_name'].values[0]} ({df_live_all.loc[df_live_all['id']==pid, 'team_short'].values[0]})",
                key="pills_live_starters",
                on_change=_on_live_pill_select
            )

            # Usynlige trigger-knapper for umiddelbar modal-åpning ved klikk på kortene
            for p_obj_id in live_all_pids:
                if st.button("inspect", key=f"btn_trigger_tab0_{p_obj_id}"):
                    st.session_state["modal_player_id"] = p_obj_id
                    st.session_state["active_tab_label"] = tab_labels[0]
                    st.rerun()

            def render_live_card_html(p: pd.Series) -> str:
                club = get_club_style(p['team_short'])
                badge_html = ""
                if p['is_cap']:
                    mult_str = f"x{p['multiplier']}"
                    badge_html = f'<span class="captain-pill" style="background:linear-gradient(135deg,#f59e0b,#d97706);">👑 C ({mult_str})</span>'
                elif p['is_vc']:
                    badge_html = '<span class="captain-pill" style="background:linear-gradient(135deg,#64748b,#475569);">🥈 VC</span>'
                elif p.get('dreamteam_count', 0) > 0:
                    dt_c = int(p.get('dreamteam_count', 0))
                    badge_html = f'<span class="captain-pill" style="background:linear-gradient(135deg,#eab308,#ca8a04);" title="Valgt til Ukens Lag {dt_c}x">⭐ {dt_c}x</span>'

                m_status = p['match_status']
                is_live = (m_status == 'I gang')
                is_finished = (m_status == 'Ferdig')
                is_upcoming = (not is_live and not is_finished)
                mins = int(p.get('minutes', 0))

                if is_live:
                    mins_str = f"{mins}'" if mins > 0 else ""
                    status_badge = f'<div class="pitch-card-fixture is-live-badge" title="Kamp pågår nå ({mins} minutter spilt)">🔴 LIVE {mins_str} • {p["match_score"]}</div>'
                    if p['events_str'] != "-":
                        details_html = f'<div class="pitch-card-events">{p["events_str"]} <span style="font-weight:600; opacity:0.8;">({p["match_desc"]})</span></div>'
                    else:
                        details_html = f'<div class="pitch-card-match-sub">🔴 I spill nå • vs {p["match_desc"]}</div>'

                elif is_finished:
                    mins_tag = f"{mins}'" if mins > 0 else "0'"
                    tooltip_fin = f"Ferdigspilt kamp: {mins} minutter spilt. Sluttresultat {p['match_score']} ({p['match_desc']})"
                    status_badge = f'<div class="pitch-card-fixture is-finished-badge" title="{tooltip_fin}">✅ FERDIG ({mins_tag}) • {p["match_score"]}</div>'
                    if p['events_str'] != "-":
                        details_html = f'<div class="pitch-card-events">{p["events_str"]} <span style="font-weight:600; opacity:0.8;">({p["match_desc"]})</span></div>'
                    elif mins > 0:
                        details_html = f'<div class="pitch-card-match-sub">⏱️ {mins} min spilt • vs {p["match_desc"]}</div>'
                    else:
                        details_html = f'<div class="pitch-card-match-sub" style="color:#f87171;">⏱️ 0 min (Ubenyttet) • vs {p["match_desc"]}</div>'

                else:
                    status_badge = f'<div class="pitch-card-fixture is-upcoming-badge" title="Kamp ikke startet ennå">⏳ {p["match_desc"]}</div>'
                    details_html = '<div class="pitch-card-match-sub">⏳ Ikke startet</div>'

                eff = p['effective_pts']
                extra_class = ""
                if is_live:
                    extra_class = " is-live-match"
                    pts_color = "#ef4444" if eff == 0 else ("var(--pitch-pts-green)" if eff >= 6 else "var(--text-primary)")
                elif is_upcoming:
                    extra_class = " is-upcoming-match"
                    pts_color = "var(--text-muted)"
                else:
                    # Ferdig
                    if eff >= 6:
                        pts_color = "var(--pitch-pts-green)"
                        extra_class = " is-haul"
                    elif eff >= 2:
                        pts_color = "var(--text-primary)"
                    else:
                        pts_color = "#ef4444"

                captain_class = " is-captain" if p['is_cap'] else ""
                logo_tag = f'<img src="{club.get("logo", "")}" class="club-badge-img" alt="{p["team_short"]}" onerror="this.style.display=\'none\';" />' if club.get("logo") else ""
                photo_html = f'<div class="card-photo-container"><img src="{p.get("photo_url", "")}" class="card-player-photo" alt="{p["web_name"]}" loading="lazy" onerror="this.style.display=\'none\'; this.parentElement.classList.add(\'no-photo\');" /></div>' if p.get("photo_url") else ""

                return (
                    f'<div class="pitch-card-link" role="button" tabindex="0" data-player-id="{p["id"]}" data-tab="0" data-player-name="{p["web_name"]}" data-player-team="{p["team_short"]}" onclick="if(window.triggerCardInspect){{window.triggerCardInspect(\'{p["id"]}\', \'{p["web_name"]}\', \'{p["team_short"]}\', \'0\');}} return false;" title="Trykk for full profil, analyse og grafer for {p["web_name"]}">'
                    f'<div class="pitch-player-card{captain_class}{extra_class}">'
                    f'<div class="card-club-bar" style="background: {club["gradient"]};"></div>'
                    f'<div class="card-top-row">'
                    f'<div class="card-club-group">'
                    f'{logo_tag}'
                    f'<span class="club-pill" style="background:{club["primary"]}; color:{club["text"]}; border:1px solid {club["border"]};">{p["team_short"]}</span>'
                    f'</div>'
                    f'{badge_html}'
                    f'</div>'
                    f'{photo_html}'
                    f'<div class="pitch-card-name" title="{p["web_name"]}">{p["web_name"]}</div>'
                    f'{status_badge}'
                    f'{details_html}'
                    f'<div class="pitch-pts-box">'
                    f'<span class="pitch-pts-label">Poeng</span>'
                    f'<span class="pitch-pts-val" style="color:{pts_color};">{eff}</span>'
                    f'</div>'
                    f'<div class="pitch-card-click-hint">🔍 Profil & Grafer</div>'
                    f'</div>'
                    f'</div>'
                )

            if live_view_mode == "🏟️ Fotballbane":
                live_gks = df_live_starters[df_live_starters['element_type'] == 1]
                live_defs = df_live_starters[df_live_starters['element_type'] == 2]
                live_mids = df_live_starters[df_live_starters['element_type'] == 3]
                live_fwds = df_live_starters[df_live_starters['element_type'] == 4]

                pitch_live_rows = [
                    '<div class="pitch-container">',
                    '<div class="pitch-midfield-line"></div>',
                    '<div class="pitch-center-circle"></div>'
                ]

                for r_group in [live_gks, live_defs, live_mids, live_fwds]:
                    pitch_live_rows.append('<div class="pitch-row">')
                    for _, p_item in r_group.iterrows():
                        pitch_live_rows.append(render_live_card_html(p_item))
                    pitch_live_rows.append('</div>')

                pitch_live_rows.append('</div>')
                live_pitch_html = "".join(pitch_live_rows) + get_pitch_card_script()

                if hasattr(st, "html"):
                    st.html(live_pitch_html, unsafe_allow_javascript=True)
                else:
                    st.markdown(live_pitch_html, unsafe_allow_html=True)

            else:
                # Tabellvisning
                tab_df = df_live_starters[['web_name', 'team_short', 'pos_short', 'is_cap', 'is_vc', 'minutes', 'goals', 'assists', 'clean_sheets', 'bonus', 'raw_pts', 'multiplier', 'effective_pts', 'match_status', 'match_desc', 'match_score']].copy()
                tab_df['Logo'] = tab_df['team_short'].apply(lambda s: get_club_style(s).get('logo', ''))
                tab_df['Rolle'] = tab_df.apply(lambda r: "👑 Kaptein" if r['is_cap'] else ("🥈 Visekaptein" if r['is_vc'] else "Starter"), axis=1)
                tab_df['Kamp'] = tab_df.apply(
                    lambda r: f"✅ Ferdig: {r['match_score']} ({r['match_desc']})" if r['match_status'] == 'Ferdig'
                    else (f"🔴 Live: {r['match_score']} ({r['match_desc']})" if r['match_status'] == 'I gang'
                    else f"⏳ Ikke startet: {r['match_desc']}"),
                    axis=1
                )
                tab_df = tab_df[['Logo', 'web_name', 'team_short', 'pos_short', 'Rolle', 'Kamp', 'minutes', 'goals', 'assists', 'clean_sheets', 'bonus', 'raw_pts', 'effective_pts']]
                tab_df.columns = ['Logo', 'Spiller', 'Klubb', 'Pos', 'Rolle', 'Kampstatus', 'Min', 'Mål ⚽', 'Ass 👟', 'CS 🧤', 'Bonus ⭐', 'Rå Pts', 'Eff. Poeng']
                st.dataframe(
                    tab_df,
                    column_config={"Logo": st.column_config.ImageColumn("Logo", width="small")},
                    width="stretch",
                    hide_index=True
                )

            # Benkvisning (Live)
            st.markdown("### 🪑 Live Benk & Innbyttere")
            bench_cols_live = st.columns(4)
            bench_lbls_live = ["Reservekeeper (GK)", "Benk 1 (Sub)", "Benk 2 (Sub)", "Benk 3 (Sub)"]

            if not df_live_bench.empty:
                for b_idx, (_, b_row) in enumerate(df_live_bench.iterrows()):
                    if b_idx >= 4:
                        break
                    club_b = get_club_style(b_row['team_short'])
                    lbl = bench_lbls_live[b_idx]
                    b_status = b_row['match_status']
                    b_mins = int(b_row.get('minutes', 0))

                    if b_status == 'Ferdig':
                        b_mins_tag = f"{b_mins}'" if b_mins > 0 else "0'"
                        tooltip_b = f"Ferdigspilt kamp: {b_mins} minutter spilt ({b_row['match_desc']})"
                        b_status_badge = f'<div class="pitch-card-fixture is-finished-badge" title="{tooltip_b}">✅ FERDIG ({b_mins_tag}) • {b_row["match_score"]}</div>'
                        if b_row['events_str'] != "-":
                            b_details_html = f'<div class="pitch-card-events">{b_row["events_str"]} <span style="font-weight:600; opacity:0.8;">({b_row["match_desc"]})</span></div>'
                        elif b_mins > 0:
                            b_details_html = f'<div class="pitch-card-match-sub">⏱️ {b_mins} min spilt • vs {b_row["match_desc"]}</div>'
                        else:
                            b_details_html = f'<div class="pitch-card-match-sub" style="color:#f87171;">⏱️ 0 min (Ubenyttet) • vs {b_row["match_desc"]}</div>'
                    elif b_status == 'I gang':
                        b_status_badge = f'<div class="pitch-card-fixture is-live-badge" title="Kamp pågår nå ({b_mins} min)">🔴 LIVE {b_mins}\' • {b_row["match_score"]}</div>'
                        b_details_html = f'<div class="pitch-card-events">{b_row["events_str"]} <span style="font-weight:600; opacity:0.8;">({b_row["match_desc"]})</span></div>' if b_row['events_str'] != "-" else f'<div class="pitch-card-match-sub">🔴 I spill • vs {b_row["match_desc"]}</div>'
                    else:
                        b_status_badge = f'<div class="pitch-card-fixture is-upcoming-badge" title="Kamp ikke startet ennå">⏳ {b_row["match_desc"]}</div>'
                        b_details_html = '<div class="pitch-card-match-sub">⏳ Ikke startet</div>'

                    b_logo = f'<img src="{club_b.get("logo", "")}" class="club-badge-img" alt="{b_row["team_short"]}" onerror="this.style.display=\'none\';" />' if club_b.get("logo") else ""
                    b_photo = f'<div class="card-photo-container"><img src="{b_row.get("photo_url", "")}" class="card-player-photo" alt="{b_row["web_name"]}" loading="lazy" onerror="this.style.display=\'none\'; this.parentElement.classList.add(\'no-photo\');" /></div>' if b_row.get("photo_url") else ""

                    live_bench_card_html = (
                        f'<div class="pitch-card-link" role="button" tabindex="0" data-player-id="{b_row["id"]}" data-tab="0" data-player-name="{b_row["web_name"]}" data-player-team="{b_row["team_short"]}" onclick="if(window.triggerCardInspect){{window.triggerCardInspect(\'{b_row["id"]}\', \'{b_row["web_name"]}\', \'{b_row["team_short"]}\', \'0\');}} return false;" style="width:100%; cursor:pointer;" title="Trykk for full profil, analyse og grafer for {b_row["web_name"]}">'
                        f'<div class="bench-card">'
                        f'<div class="card-club-bar" style="background: {club_b["gradient"]};"></div>'
                        f'<div class="card-top-row">'
                        f'<div class="card-club-group">'
                        f'{b_logo}'
                        f'<span class="club-pill" style="background:{club_b["primary"]}; color:{club_b["text"]}; border:1px solid {club_b["border"]};">{b_row["team_short"]}</span>'
                        f'</div>'
                        f'<span style="font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-muted);">{lbl}</span>'
                        f'</div>'
                        f'{b_photo}'
                        f'<div class="pitch-card-name" title="{b_row["web_name"]}">{b_row["web_name"]}</div>'
                        f'{b_status_badge}'
                        f'{b_details_html}'
                        f'<div class="pitch-pts-box">'
                        f'<span class="pitch-pts-label">Poeng</span>'
                        f'<span class="pitch-pts-val">{b_row["raw_pts"]}</span>'
                        f'</div>'
                        f'<div class="pitch-card-click-hint">🔍 Profil & Grafer</div>'
                        f'</div>'
                        f'</div>'
                    )
                    with bench_cols_live[b_idx]:
                        live_bench_full_html = live_bench_card_html + get_pitch_card_script()
                        if hasattr(st, "html"):
                            st.html(live_bench_full_html, unsafe_allow_javascript=True)
                        else:
                            st.markdown(live_bench_full_html, unsafe_allow_html=True)
                        if st.button(f"🔍 {b_row['web_name']}", key=f"btn_bench_live_{b_row['id']}", width="stretch"):
                            render_player_dialog(b_row['id'], df_players, bootstrap, eval_horizon, events, team_fixtures, manager_id=manager_id)

            # Rundens Kamper
            st.markdown("---")
            with st.expander(f"📅 Alle Kamper i {live_gw_name}", expanded=False):
                if gw_fixtures_list:
                    fx_rows = []
                    for f in gw_fixtures_list:
                        th_n = teams_dict_all.get(f['team_h'], {}).get('name', 'Hjemme')
                        ta_n = teams_dict_all.get(f['team_a'], {}).get('name', 'Borte')
                        hs = f.get('team_h_score')
                        ascore = f.get('team_a_score')
                        start = f.get('started', False)
                        mins = f.get('minutes', 0)
                        fin = f.get('finished', False) or f.get('finished_provisional', False) or (start and mins >= 90)
                        
                        if fin:
                            st_display = "✅ Ferdig"
                            res_display = f"{hs} - {ascore}"
                        elif start:
                            st_display = f"🔴 I gang ({mins}')" if mins > 0 else "🔴 I gang"
                            res_display = f"{hs} - {ascore}"
                        else:
                            st_display = "⏳ Ikke startet"
                            res_display = "vs"
                            
                        fx_rows.append({
                            'Status': st_display,
                            'Hjemmelag': th_n,
                            'Resultat': res_display,
                            'Bortelag': ta_n
                        })
                    st.dataframe(pd.DataFrame(fx_rows)[['Status', 'Hjemmelag', 'Resultat', 'Bortelag']], width="stretch", hide_index=True)
                else:
                    st.info("Ingen kamper funnet for denne runden.")

    # ==========================================================================
    # FANE 1: MITT LAG & NESTE RUNDE
    # ==========================================================================
    with tab1:
        next_gw_name = get_event_display_name(next_gw, events)
        st.subheader("📋 Optimal Startellever & Benk")
        st.caption(f"Beregnet med PuLP for å maksimere forventede poeng (xP) i {next_gw_name}.")
        
        starters, bench_ordered, captain_id, vice_captain_id = solve_optimal_starting_xi(squad_df)
        
        cap_player = df_players[df_players['id'] == captain_id].iloc[0]
        vc_player = df_players[df_players['id'] == vice_captain_id].iloc[0]
        
        # Kaptein-anbefaling Callout
        st.info(
            f"👑 **Anbefalt Kaptein:** **{cap_player['web_name']}** ({cap_player['team_short']}) — "
            f"**{cap_player['next_gw_xp']:.2f} xP** ({cap_player['next_fixture']}) | "
            f"🥈 **Visekaptein:** **{vc_player['web_name']}** ({vc_player['team_short']}) — "
            f"**{vc_player['next_gw_xp']:.2f} xP** ({vc_player['next_fixture']})"
        )
        st.caption("💡 **Tips til koden på kortene:** F.eks. **KBK (B) [FDR 2]** betyr motstander **Kristiansund Ballklubb**, **(B)** = Bortekamp (*(H)* = Hjemme), og **FDR 2** = *Fixture Difficulty Rating* på en skala fra 1 (svært lett/grønn) til 5 (ekstremt tøff/rød).")
        
        # Opprett fotballbane
        starters_df = df_players[df_players['id'].isin(starters)]
        gk_starters = starters_df[starters_df['element_type'] == 1]
        def_starters = starters_df[starters_df['element_type'] == 2]
        mid_starters = starters_df[starters_df['element_type'] == 3]
        fwd_starters = starters_df[starters_df['element_type'] == 4]
        
        formation_str = f"{len(def_starters)}-{len(mid_starters)}-{len(fwd_starters)}"
        
        # Hurtigsøk etter spiller for profil & analyse
        with st.expander("🔍 Søk opp og analyser en hvilken som helst spiller i Eliteserien", expanded=False):
            c_sel_p1, c_btn_p1 = st.columns([3, 1])
            with c_sel_p1:
                sorted_p1 = df_players.sort_values(by='web_name')
                chosen_p1 = st.selectbox(
                    "Velg spiller for dybdeanalyse:",
                    options=sorted_p1['id'].tolist(),
                    format_func=lambda x: f"{df_players.loc[df_players['id']==x, 'web_name'].values[0]} ({df_players.loc[df_players['id']==x, 'team_short'].values[0]} - {df_players.loc[df_players['id']==x, 'pos_short'].values[0]})",
                    key="quick_inspect_player_tab1"
                )
            with c_btn_p1:
                st.write("")
                st.write("")
                if st.button("Åpne profil & grafer", key="btn_open_quick_inspect_tab1", width="stretch"):
                    render_player_dialog(chosen_p1, df_players, bootstrap, eval_horizon, events, team_fixtures, manager_id=manager_id)

        col_form, col_toggle = st.columns([1, 1])
        with col_form:
            st.markdown(f"**Anbefalt Formasjon:** `{formation_str}`")
        with col_toggle:
            view_mode = st.radio("Visningsformat", ["🏟️ Fotballbane", "📋 Tabelloversikt"], horizontal=True, label_visibility="collapsed")

        # Umiddelbar spillerprofil-velger for startellever og benk (uten sidelasting)
        def _on_squad_pill_select():
            sel = st.session_state.get("pills_squad_starters")
            if sel:
                st.session_state["modal_player_id"] = sel
                st.session_state["active_tab_label"] = tab_labels[1]
                st.session_state["pills_squad_starters"] = None

        all_squad_pids = starters + bench_ordered
        st.pills(
            "🔍 Klikk for umiddelbar profil & grafer (eller trykk på kortet):",
            options=all_squad_pids,
            format_func=lambda pid: f"{df_players.loc[df_players['id']==pid, 'web_name'].values[0]} ({df_players.loc[df_players['id']==pid, 'team_short'].values[0]})",
            key="pills_squad_starters",
            on_change=_on_squad_pill_select
        )

        # Usynlige trigger-knapper for umiddelbar modal-åpning ved klikk på kortene
        for p_obj_id in all_squad_pids:
            if st.button("inspect", key=f"btn_trigger_tab1_{p_obj_id}"):
                st.session_state["modal_player_id"] = p_obj_id
                st.session_state["active_tab_label"] = tab_labels[1]
                st.rerun()

        def render_card_inline(p: pd.Series, is_c: bool, is_vc: bool) -> str:
            club = get_club_style(p['team_short'])
            badge_html = ""
            if is_c:
                badge_html = '<span class="captain-pill" style="background:linear-gradient(135deg,#f59e0b,#d97706);">👑 C (2x)</span>'
            elif is_vc:
                badge_html = '<span class="captain-pill" style="background:linear-gradient(135deg,#64748b,#475569);">🥈 VC</span>'
            elif p.get('dreamteam_count', 0) > 0:
                dt_c = int(p.get('dreamteam_count', 0))
                badge_html = f'<span class="captain-pill" style="background:linear-gradient(135deg,#eab308,#ca8a04);" title="Valgt til Ukens Lag {dt_c}x">⭐ {dt_c}x</span>'

            fdr = int(p.get('next_fdr', 3))
            fdr_styles = {
                1: "background:#16a34a; color:#ffffff;",
                2: "background:#86efac; color:#064e3b;",
                3: "background:var(--pitch-status-ft-bg); color:var(--pitch-status-ft-text);",
                4: "background:#fdba74; color:#7c2d12;",
                5: "background:#ef4444; color:#ffffff;"
            }
            fdr_style = fdr_styles.get(fdr, "background:var(--pitch-status-ft-bg); color:var(--pitch-status-ft-text);")
            captain_class = " is-captain" if is_c else ""
            xp_val = (p['next_gw_xp'] * 2) if is_c else p['next_gw_xp']

            # Skadestatus-indikator på kortet
            status = p.get('status', 'a')
            status_html = ""
            extra_class = ""
            if status in ['i', 's', 'u']:
                status_html = f'<div style="background:#fee2e2; color:#991b1b; font-size:11px; font-weight:800; border-radius:6px; padding:2px 4px; margin:2px 0;" title="{p.get("news", "")}">🔴 {p.get("status_desc", "Ute")}</div>'
                extra_class = " is-player-out"
            elif status == 'd':
                status_html = f'<div style="background:#fef3c7; color:#92400e; font-size:11px; font-weight:800; border-radius:6px; padding:2px 4px; margin:2px 0;" title="{p.get("news", "")}">🟡 {p.get("status_desc", "Usikker")}</div>'
            else:
                cost_chg = p.get('cost_change_start', 0) / 10.0
                chg_arrow = " ▲" if cost_chg > 0 else (" ▼" if cost_chg < 0 else "")
                status_html = f'<div class="pitch-card-info-text">{p.get("now_cost", 0)}m{chg_arrow} • Form {p.get("form", "-")}</div>'

            logo_tag = f'<img src="{club.get("logo", "")}" class="club-badge-img" alt="{p["team_short"]}" onerror="this.style.display=\'none\';" />' if club.get("logo") else ""
            photo_html = f'<div class="card-photo-container"><img src="{p.get("photo_url", "")}" class="card-player-photo" alt="{p["web_name"]}" loading="lazy" onerror="this.style.display=\'none\'; this.parentElement.classList.add(\'no-photo\');" /></div>' if p.get("photo_url") else ""

            return (
                f'<div class="pitch-card-link" role="button" tabindex="0" data-player-id="{p["id"]}" data-tab="1" data-player-name="{p["web_name"]}" data-player-team="{p["team_short"]}" onclick="if(window.triggerCardInspect){{window.triggerCardInspect(\'{p["id"]}\', \'{p["web_name"]}\', \'{p["team_short"]}\', \'1\');}} return false;" title="Trykk for full profil, analyse og grafer for {p["web_name"]}">'
                f'<div class="pitch-player-card{captain_class}{extra_class}">'
                f'<div class="card-club-bar" style="background: {club["gradient"]};"></div>'
                f'<div class="card-top-row">'
                f'<div class="card-club-group">'
                f'{logo_tag}'
                f'<span class="club-pill" style="background:{club["primary"]}; color:{club["text"]}; border:1px solid {club["border"]};">{p["team_short"]}</span>'
                f'</div>'
                f'{badge_html}'
                f'</div>'
                f'{photo_html}'
                f'<div class="pitch-card-name" title="{p["web_name"]}">{p["web_name"]}</div>'
                f'<div class="pitch-card-fixture" style="{fdr_style}" title="Neste motstander og vanskelighetsgrad (FDR 1-5)">{p["next_fixture"]}</div>'
                f'{status_html}'
                f'<div class="pitch-pts-box">'
                f'<span class="pitch-pts-label">xP {next_gw_name}</span>'
                f'<span class="pitch-pts-val" style="color:var(--pitch-pts-green);">{xp_val:.1f}</span>'
                f'</div>'
                f'<div class="pitch-card-click-hint">🔍 Profil & Grafer</div>'
                f'</div>'
                f'</div>'
            )

        if view_mode == "🏟️ Fotballbane":
            pitch_rows = [
                '<div class="pitch-container">',
                '<div class="pitch-midfield-line"></div>',
                '<div class="pitch-center-circle"></div>'
            ]
            
            for row_players in [gk_starters, def_starters, mid_starters, fwd_starters]:
                pitch_rows.append('<div class="pitch-row">')
                for _, p in row_players.iterrows():
                    pitch_rows.append(render_card_inline(p, p['id'] == captain_id, p['id'] == vice_captain_id))
                pitch_rows.append('</div>')
                
            pitch_rows.append('</div>')
            pitch_html = "".join(pitch_rows) + get_pitch_card_script()

            if hasattr(st, "html"):
                st.html(pitch_html, unsafe_allow_javascript=True)
            else:
                st.markdown(pitch_html, unsafe_allow_html=True)
        else:
            # Tabellvisning for startellever
            starters_table = starters_df[['web_name', 'team_short', 'pos_short', 'now_cost', 'next_fixture', 'next_fdr', 'next_gw_xp', 'total_xp', 'status_desc', 'news']].copy()
            starters_table['Logo'] = starters_df['team_short'].apply(lambda s: get_club_style(s).get('logo', ''))
            starters_table['Rolle'] = starters_table.index.map(
                lambda idx: "👑 Kaptein (2x)" if starters_df.loc[idx, 'id'] == captain_id else ("🥈 Visekaptein" if starters_df.loc[idx, 'id'] == vice_captain_id else "Starter")
            )
            starters_table = starters_table[['Logo', 'web_name', 'team_short', 'pos_short', 'now_cost', 'next_fixture', 'next_fdr', 'next_gw_xp', 'total_xp', 'status_desc', 'news', 'Rolle']]
            starters_table.columns = ['Logo', 'Spiller', 'Klubb', 'Posisjon', 'Pris (m)', 'Neste Kamp', 'FDR', f'{next_gw_name} xP', f'Total xP ({eval_horizon}r)', 'Status', 'Skademelding', 'Rolle']
            st.dataframe(
                starters_table,
                column_config={"Logo": st.column_config.ImageColumn("Logo", width="small")},
                width="stretch",
                hide_index=True
            )

        # Benk
        st.markdown("### 🪑 Anbefalt Benkerangering")
        bench_cols = st.columns(4)
        bench_labels = ["Reservekeeper (GK)", "Benk 1 (Sub)", "Benk 2 (Sub)", "Benk 3 (Sub)"]
        for idx, bpid in enumerate(bench_ordered):
            bp = df_players[df_players['id'] == bpid].iloc[0]
            club_bp = get_club_style(bp['team_short'])
            fdr_bp = int(bp.get('next_fdr', 3))
            fdr_bp_styles = {
                1: "background:#16a34a; color:#ffffff;",
                2: "background:#86efac; color:#064e3b;",
                3: "background:var(--pitch-status-ft-bg); color:var(--pitch-status-ft-text);",
                4: "background:#fdba74; color:#7c2d12;",
                5: "background:#ef4444; color:#ffffff;"
            }
            fdr_bp_style = fdr_bp_styles.get(fdr_bp, "background:var(--pitch-status-ft-bg); color:var(--pitch-status-ft-text);")
            
            # Status på benkkort
            bp_status = bp.get('status', 'a')
            bp_status_html = ""
            if bp_status in ['i', 's', 'u']:
                bp_status_html = f'<div style="background:#fee2e2; color:#991b1b; font-size:11px; font-weight:800; border-radius:6px; padding:2px 4px; margin:2px 0;">{bp.get("status_desc", "Ute")}</div>'
            elif bp_status == 'd':
                bp_status_html = f'<div style="background:#fef3c7; color:#92400e; font-size:11px; font-weight:800; border-radius:6px; padding:2px 4px; margin:2px 0;">{bp.get("status_desc", "Usikker")}</div>'
            else:
                bp_status_html = f'<div class="pitch-card-info-text">{bp.get("now_cost", 0)}m • Form {bp.get("form", "-")}</div>'

            bp_logo = f'<img src="{club_bp.get("logo", "")}" class="club-badge-img" alt="{bp["team_short"]}" onerror="this.style.display=\'none\';" />' if club_bp.get("logo") else ""
            bp_photo = f'<div class="card-photo-container"><img src="{bp.get("photo_url", "")}" class="card-player-photo" alt="{bp["web_name"]}" loading="lazy" onerror="this.style.display=\'none\'; this.parentElement.classList.add(\'no-photo\');" /></div>' if bp.get("photo_url") else ""

            bench_card_html = (
                f'<div class="pitch-card-link" role="button" tabindex="0" data-player-id="{bp["id"]}" data-tab="1" data-player-name="{bp["web_name"]}" data-player-team="{bp["team_short"]}" onclick="if(window.triggerCardInspect){{window.triggerCardInspect(\'{bp["id"]}\', \'{bp["web_name"]}\', \'{bp["team_short"]}\', \'1\');}} return false;" style="width:100%; cursor:pointer;" title="Trykk for full profil, analyse og grafer for {bp["web_name"]}">'
                f'<div class="bench-card">'
                f'<div class="card-club-bar" style="background: {club_bp["gradient"]};"></div>'
                f'<div class="card-top-row">'
                f'<div class="card-club-group">'
                f'{bp_logo}'
                f'<span class="club-pill" style="background:{club_bp["primary"]}; color:{club_bp["text"]}; border:1px solid {club_bp["border"]};">{bp["team_short"]}</span>'
                f'</div>'
                f'<span style="font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:0.5px; color:var(--text-muted);">{bench_labels[idx]}</span>'
                f'</div>'
                f'{bp_photo}'
                f'<div class="pitch-card-name" title="{bp["web_name"]}">{bp["web_name"]}</div>'
                f'{bp_status_html}'
                f'<div class="pitch-card-fixture" style="{fdr_bp_style}" title="Neste motstander og vanskelighetsgrad (FDR 1-5)">{bp["next_fixture"]}</div>'
                f'<div class="pitch-pts-box">'
                f'<span class="pitch-pts-label">xP {next_gw_name}</span>'
                f'<span class="pitch-pts-val" style="color:var(--pitch-pts-green);">{bp["next_gw_xp"]:.2f}</span>'
                f'</div>'
                f'<div class="pitch-card-click-hint">🔍 Profil & Grafer</div>'
                f'</div>'
                f'</div>'
            )
            with bench_cols[idx]:
                bench_full_html = bench_card_html + get_pitch_card_script()
                if hasattr(st, "html"):
                    st.html(bench_full_html, unsafe_allow_javascript=True)
                else:
                    st.markdown(bench_full_html, unsafe_allow_html=True)
                if st.button(f"🔍 {bp['web_name']}", key=f"btn_bench_squad_{bp['id']}", width="stretch"):
                    render_player_dialog(bp['id'], df_players, bootstrap, eval_horizon, events, team_fixtures, manager_id=manager_id)

    # ==========================================================================
    # FANE 2: TRANSFER HUB (BESTE KJØP & SALG)
    # ==========================================================================
    with tab2:
        st.subheader("🔄 PuLP Transfer Optimizer & Hit-Kalkulator")
        st.caption(f"Matematisk optimering som evaluerer overganger og minuspoeng (-4 hits) over {eval_horizon} runder.")

        # Løs for 1 og 2 bytter med PuLP
        with st.spinner("Kjører PuLP MILP-optimeringsmotor..."):
            opt_1 = solve_transfers_pulp(df_players, squad_pids, bank_m, 1, teams_dict)
            opt_2 = None
            opt_2_1gw = None
            if allow_hits or free_transfers >= 2:
                opt_2 = solve_transfers_pulp(df_players, squad_pids, bank_m, 2, teams_dict)
                # Beregn også 1-runders gevinst for sammenligning av hit over tid
                df_players_1gw = compute_player_xp_model(bootstrap, team_fixtures, next_gw, 1)
                opt_2_1gw = solve_transfers_pulp(df_players_1gw, squad_pids, bank_m, 2, teams_dict)

        col_trans1, col_trans2 = st.columns(2)

        # 1 Bytte (Gratis)
        with col_trans1:
            st.markdown("#### 🥇 Beste 1-Bytte (Gratis)")
            if opt_1:
                p_out = opt_1['transfers_out'].iloc[0]
                p_in = opt_1['transfers_in'].iloc[0]
                net_gain_1 = opt_1['gross_gain']
                
                box1_html = (
                    f'<div style="background:#eff6ff; border-left:6px solid #3b82f6; border-radius:10px; padding:16px; margin-bottom:1.2rem; color:#1e293b; box-shadow:0 2px 6px rgba(0,0,0,0.05);">'
                    f'<h4 style="margin:0 0 8px 0; color:#1d4ed8; font-size:16px;">Optimalt 1-Bytte (0 straffepoeng)</h4>'
                    f'<p style="margin:5px 0; font-size:14px;"><b>UT:</b> ❌ {p_out["web_name"]} ({p_out["team_short"]} | {p_out["pos_short"]} | {p_out["now_cost"]}m) — {p_out["total_xp"]:.2f} xP</p>'
                    f'<p style="margin:5px 0; font-size:14px;"><b>INN:</b> ✅ {p_in["web_name"]} ({p_in["team_short"]} | {p_in["pos_short"]} | {p_in["now_cost"]}m) — {p_in["total_xp"]:.2f} xP</p>'
                    f'<hr style="margin:10px 0; border:0; border-top:1px solid #bfdbfe;">'
                    f'<p style="margin:0; font-weight:700; font-size:14px;">Forventet Netto Gevinst ({eval_horizon} runder): <span style="color:#1d4ed8;">+{net_gain_1:.2f} poeng</span></p>'
                    f'<p style="margin:4px 0 0 0; font-size:12px; color:#64748b;">Gjenværende bank: {opt_1["remaining_bank"]:.1f}m</p>'
                    f'</div>'
                )
                if hasattr(st, "html"):
                    st.html(box1_html)
                else:
                    st.markdown(box1_html, unsafe_allow_html=True)
                
                if st.button(f"✅ Registrer dette byttet i appen (UT: {p_out['web_name']} ➔ INN: {p_in['web_name']})", key="apply_opt1", width="stretch"):
                    st.session_state['applied_transfers'].append((int(p_out['id']), int(p_in['id'])))
                    add_tracked_transfer(
                        gw=next_gw,
                        out_id=p_out['id'], out_name=p_out['web_name'], out_team=p_out['team_short'],
                        in_id=p_in['id'], in_name=p_in['web_name'], in_team=p_in['team_short'],
                        exp_gain=net_gain_1, hit_cost=0
                    )
                    st.rerun()
            else:
                st.warning("Fant ingen gyldig overgang innen budsjett.")

        # 2 Bytter (Evaluering av Hit)
        with col_trans2:
            st.markdown("#### 🥈 Beste 2-Bytter (Hit-evaluering)")
            if not allow_hits and free_transfers < 2:
                st.info(
                    "🔒 **2-bytte evaluering er deaktivert (-4 poengstraff er avslått).**  \n\n"
                    "Du har 1 gratisbytte tilgjengelig, og standardinnstillingen tillater ikke minuspoeng.  \n"
                    "Dersom du selv ønsker å vurdere 2 bytter med -4 poengstraff (hit), kan du hake av for **'Tillat -4 poengstraff (Hits)'** i sidemenyen til venstre."
                )
            elif opt_2:
                outs = opt_2['transfers_out']
                ins = opt_2['transfers_in']
                gross_gain_2 = opt_2['gross_gain']
                
                hit_penalty = 0.0 if free_transfers >= 2 else 4.0
                net_gain_2 = round(gross_gain_2 - hit_penalty, 2)
                
                # 1-runders gevinst ved 2 bytter
                net_gain_1gw = 0.0
                if opt_2_1gw:
                    net_gain_1gw = round(opt_2_1gw['gross_gain'] - hit_penalty, 2)

                # Anbefalingsflagg
                is_hit = hit_penalty > 0
                if is_hit:
                    if net_gain_2 >= 4.0:
                        hit_status = "🟢 ANBEFALT HIT"
                        box_style = "background:#f0fdf4; border-left:6px solid #22c55e; color:#14532d;"
                        expl = f"Byttet forventes å gi +{net_gain_2:.2f} poeng i netto gevinst over {eval_horizon} runder, godt over -4 straffen."
                    else:
                        hit_status = "🔴 IKKE VERDT -4"
                        box_style = "background:#fef2f2; border-left:6px solid #ef4444; color:#7f1d1d;"
                        expl = f"Gevinsten (+{gross_gain_2:.2f} brutto) forsvarer ikke minuspoenget (-4 hit). Netto gevinst er kun {net_gain_2:.2f} poeng."
                else:
                    hit_status = "🟢 2 GRATISBYTTER TILGJENGELIG"
                    box_style = "background:#eff6ff; border-left:6px solid #3b82f6; color:#1e3a8a;"
                    expl = f"Du har {free_transfers} gratisbytter! Ingen minuspoeng påløper. Netto gevinst: +{net_gain_2:.2f} poeng."

                out_text = "<br>".join([f"❌ {r['web_name']} ({r['team_short']} | {r['pos_short']}) — {r['total_xp']:.2f} xP" for _, r in outs.iterrows()])
                in_text = "<br>".join([f"✅ {r['web_name']} ({r['team_short']} | {r['pos_short']}) — {r['total_xp']:.2f} xP" for _, r in ins.iterrows()])

                box2_html = (
                    f'<div style="{box_style} border-radius:10px; padding:16px; margin-bottom:1.2rem; box-shadow:0 2px 6px rgba(0,0,0,0.05);">'
                    f'<h4 style="margin:0 0 6px 0; font-size:16px;">{hit_status}</h4>'
                    f'<div style="font-size:13px; margin-bottom:8px;"><b>UT:</b><br>{out_text}</div>'
                    f'<div style="font-size:13px; margin-bottom:8px;"><b>INN:</b><br>{in_text}</div>'
                    f'<hr style="margin:8px 0; border:0; border-top:1px solid rgba(0,0,0,0.1);">'
                    f'<p style="margin:3px 0; font-size:13px;"><b>Gevinst 1 Runde (Netto):</b> {net_gain_1gw:+.2f} pts</p>'
                    f'<p style="margin:3px 0; font-size:13px;"><b>Gevinst {eval_horizon} Runder (Netto):</b> <b>{net_gain_2:+.2f} pts</b> (Brutto: +{gross_gain_2:.2f}, Hit: -{hit_penalty:.0f})</p>'
                    f'<p style="margin:4px 0 0 0; font-size:12px; opacity:0.85;"><i>{expl}</i></p>'
                    f'</div>'
                )
                if hasattr(st, "html"):
                    st.html(box2_html)
                else:
                    st.markdown(box2_html, unsafe_allow_html=True)

                if st.button("✅ Registrer disse 2 byttene i appen", key="apply_opt2", width="stretch"):
                    for (o_idx, o_row), (i_idx, i_row) in zip(outs.iterrows(), ins.iterrows()):
                        st.session_state['applied_transfers'].append((int(o_row['id']), int(i_row['id'])))
                        add_tracked_transfer(
                            gw=next_gw,
                            out_id=o_row['id'], out_name=o_row['web_name'], out_team=o_row['team_short'],
                            in_id=i_row['id'], in_name=i_row['web_name'], in_team=i_row['team_short'],
                            exp_gain=round(i_row['total_xp'] - o_row['total_xp'], 2),
                            hit_cost=int(hit_penalty / 2)
                        )
                    st.rerun()
            else:
                st.warning("Fant ingen gyldig dobbeltovergang innen budsjett.")

        # Detaljert sammenligningstabell
        st.markdown("---")
        col_out5, col_in5 = st.columns(2)

        with col_out5:
            st.markdown("### 📉 Topp 5 Bytt-Ut forslag (Ditt lag)")
            st.caption("Spillere med lavest forventning, skaderisiko eller tøft kampprogram.")
            # Ranger spillere i troppen etter lavest total xP og skaderisiko
            worst_in_squad = squad_df.sort_values(by=['play_chance', 'total_xp'], ascending=[True, True]).head(5)
            
            out_display = worst_in_squad[['web_name', 'team_short', 'pos_short', 'now_cost', 'next_fixture', 'next_fdr', 'total_xp', 'status_desc', 'news']].copy()
            out_display['Logo'] = out_display['team_short'].apply(lambda s: get_club_style(s).get('logo', ''))
            out_display = out_display[['Logo', 'web_name', 'team_short', 'pos_short', 'now_cost', 'next_fixture', 'next_fdr', 'total_xp', 'status_desc', 'news']]
            out_display.columns = ['Logo', 'Spiller', 'Klubb', 'Pos', 'Pris (m)', 'Neste Kamp', 'FDR', f'xP ({eval_horizon}r)', 'Status', 'Skademelding']
            st.dataframe(
                out_display,
                column_config={"Logo": st.column_config.ImageColumn("Logo", width="small")},
                width="stretch",
                hide_index=True
            )

        with col_in5:
            st.markdown("### 📈 Topp 5 Bytt-Inn forslag (Hele ligaen)")
            st.caption("De beste spillerne innenfor tilgjengelig budsjett og frie plasser (kun 100% friske).")
            # Filtrer ut spillere man allerede eier
            avail_players = df_players[~df_players['id'].isin(squad_pids)]
            # Krev at anbefalte kjøp er 100% spilleklare
            fit_players = avail_players[avail_players['status'] == 'a']
            best_in = fit_players.sort_values(by='total_xp', ascending=False).head(5)
            
            in_display = best_in[['web_name', 'team_short', 'pos_short', 'now_cost', 'next_fixture', 'next_fdr', 'total_xp', 'selected_by_percent']].copy()
            in_display['Logo'] = in_display['team_short'].apply(lambda s: get_club_style(s).get('logo', ''))
            in_display = in_display[['Logo', 'web_name', 'team_short', 'pos_short', 'now_cost', 'next_fixture', 'next_fdr', 'total_xp', 'selected_by_percent']]
            in_display.columns = ['Logo', 'Spiller', 'Klubb', 'Pos', 'Pris (m)', 'Neste Kamp', 'FDR', f'xP ({eval_horizon}r)', 'Valgt %']
            st.dataframe(
                in_display,
                column_config={"Logo": st.column_config.ImageColumn("Logo", width="small")},
                width="stretch",
                hide_index=True
            )

        # Interaktiv Bytte-kalkulator
        st.markdown("---")
        st.markdown("### 🎯 Skreddersydd Byttekalkulator")
        st.caption("Velg en spesifikk spiller du vil selge, og se de beste erstatningene innen budsjett.")
        
        with st.container(border=True):
            st.markdown("##### ❌ SPILLER DU VIL SELGE (UT)")
            sel_out_id = st.selectbox(
                "Velg spiller fra din tropp:",
                options=squad_df['id'].tolist(),
                format_func=lambda x: f"{squad_df.loc[squad_df['id']==x, 'web_name'].values[0]} ({squad_df.loc[squad_df['id']==x, 'team_short'].values[0]} | {squad_df.loc[squad_df['id']==x, 'pos_short'].values[0]} | {squad_df.loc[squad_df['id']==x, 'now_cost'].values[0]}m)"
            )
        
        selected_out_player = squad_df[squad_df['id'] == sel_out_id].iloc[0]
        max_buy_budget = selected_out_player['now_cost'] + bank_m
        
        # Finn erstatninger i samme posisjon innen budsjett
        candidates = df_players[
            (df_players['element_type'] == selected_out_player['element_type']) &
            (~df_players['id'].isin(squad_pids)) &
            (df_players['now_cost'] <= round(max_buy_budget, 1))
        ].copy()
        
        candidates['netto_gevinst'] = candidates['total_xp'] - selected_out_player['total_xp']
        top_replacements = candidates.sort_values(by='netto_gevinst', ascending=False).head(5)
        
        if not top_replacements.empty:
            rep_view = top_replacements[['web_name', 'team_short', 'now_cost', 'next_fixture', 'total_xp', 'netto_gevinst']].copy()
            rep_view['Logo'] = rep_view['team_short'].apply(lambda s: get_club_style(s).get('logo', ''))
            rep_view['Rest-bank (m)'] = (max_buy_budget - rep_view['now_cost']).round(1)
            rep_view = rep_view[['Logo', 'web_name', 'team_short', 'now_cost', 'next_fixture', 'total_xp', 'netto_gevinst', 'Rest-bank (m)']]
            rep_view.columns = ['Logo', 'Erstatning INN', 'Klubb', 'Pris (m)', 'Neste Kamp', f'xP ({eval_horizon}r)', 'Netto Gevinst', 'Rest-bank (m)']
            st.dataframe(
                rep_view,
                column_config={"Logo": st.column_config.ImageColumn("Logo", width="small")},
                width="stretch",
                hide_index=True
            )

            with st.container(border=True):
                st.markdown("##### ✅ VELG NY SPILLER Å HENTE INN (KJØP)")
                c_pick_rep, c_act_rep = st.columns([3, 1])
                with c_pick_rep:
                    rep_pids = top_replacements['id'].tolist()
                    chosen_rep_id = st.selectbox(
                        "Velg erstatning blant de beste kandidatene:",
                        options=rep_pids,
                        format_func=lambda x: f"{top_replacements.loc[top_replacements['id']==x, 'web_name'].values[0]} ({top_replacements.loc[top_replacements['id']==x, 'team_short'].values[0]} | {top_replacements.loc[top_replacements['id']==x, 'now_cost'].values[0]}m | +{top_replacements.loc[top_replacements['id']==x, 'netto_gevinst'].values[0]:.2f} xP)",
                        key="custom_rep_choice"
                    )
                with c_act_rep:
                    st.write("")
                    st.write("")
                    if st.button("Gjennomfør bytte", key="btn_apply_custom_rep", width="stretch"):
                        st.session_state['applied_transfers'].append((int(selected_out_player['id']), int(chosen_rep_id)))
                        chosen_rep_row = top_replacements[top_replacements['id'] == chosen_rep_id].iloc[0]
                        add_tracked_transfer(
                            gw=next_gw,
                            out_id=selected_out_player['id'], out_name=selected_out_player['web_name'], out_team=selected_out_player['team_short'],
                            in_id=chosen_rep_row['id'], in_name=chosen_rep_row['web_name'], in_team=chosen_rep_row['team_short'],
                            exp_gain=chosen_rep_row['netto_gevinst'], hit_cost=0
                        )
                        st.rerun()
        else:
            st.info("Ingen erstatninger funnet innenfor makspris.")

        # ======================================================================
        # 📈 OVERGANGSLOGG & OPTIMIZER RESULTATER (TRACK RECORD)
        # ======================================================================
        st.markdown("---")
        st.subheader("📈 Overgangslogg & Optimizer Resultater (Track Record)")
        st.caption("Følg nøyaktig hvor mange poeng overgangene genererer gjennom sesongen, og se om optimeringene gir suksess!")

        raw_tracked = load_tracked_transfers()
        if raw_tracked:
            evaluated_list = [evaluate_transfer_record(t, bootstrap) for t in raw_tracked]
            
            finished_transfers = [t for t in evaluated_list if t.get('played')]
            total_logged = len(evaluated_list)
            total_finished = len(finished_transfers)
            
            total_net_gain = sum(t['actual_gain'] for t in finished_transfers) if finished_transfers else 0
            wins = sum(1 for t in finished_transfers if t['actual_gain'] > 0)
            losses = sum(1 for t in finished_transfers if t['actual_gain'] < 0)
            draws = sum(1 for t in finished_transfers if t['actual_gain'] == 0)
            win_rate = (wins / total_finished * 100) if total_finished > 0 else 0.0
            avg_gain = (total_net_gain / total_finished) if total_finished > 0 else 0.0
            avg_exp = (sum(t.get('exp_gain', 0.0) for t in finished_transfers) / total_finished) if total_finished > 0 else 0.0

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("Bytter Logget", f"{total_logged} stk", f"{total_finished} ferdigspilt")
            with sc2:
                gain_sign = "+" if total_net_gain > 0 else ""
                st.metric("Netto Poenggevinst", f"{gain_sign}{total_net_gain} pts", f"Snitt: {gain_sign}{avg_gain:.1f} pts/bytte")
            with sc3:
                st.metric("Suksessrate", f"{win_rate:.0f}%", f"{wins} vinn • {draws} uavgjort • {losses} tap")
            with sc4:
                st.metric("Forventet vs. Faktisk", f"{gain_sign}{avg_gain:.1f} pts", f"Forventet xP: +{avg_exp:.1f}")

            st.markdown("##### 📋 Loggførte Overganger")
            tracker_rows = []
            for t in evaluated_list:
                in_p_str = f"{t['in_name']} ({t['in_team']})"
                out_p_str = f"{t['out_name']} ({t['out_team']})"
                in_pts_disp = f"{t['in_pts']} pts" if t['in_pts'] is not None else "⏳"
                out_pts_disp = f"{t['out_pts']} pts" if t['out_pts'] is not None else "⏳"
                gain_disp = f"{t['actual_gain']:+d} pts" if t['actual_gain'] is not None else "⏳ Venter"
                hit_disp = f"-{t['hit_cost']}" if t.get('hit_cost', 0) > 0 else "0"
                
                tracker_rows.append({
                    "ID": t['id'],
                    "Runde": f"GW {t['gw']}",
                    "Dato": t.get('date', '-'),
                    "UT (Solgt)": f"❌ {out_p_str} [{out_pts_disp}]",
                    "INN (Kjøpt)": f"✅ {in_p_str} [{in_pts_disp}]",
                    "Hit": hit_disp,
                    "xP Forventet": f"+{t.get('exp_gain', 0.0):.2f}",
                    "Faktisk Gevinst": gain_disp,
                    "Status": t['status']
                })
            
            df_track_show = pd.DataFrame(tracker_rows)
            st.dataframe(
                df_track_show.drop(columns=['ID']),
                width="stretch",
                hide_index=True
            )

            with st.expander("⚙️ Administrer / Slett loggførte overganger", expanded=False):
                c_del_sel, c_del_btn = st.columns([3, 1])
                with c_del_sel:
                    del_opts = [t['id'] for t in evaluated_list]
                    chosen_del_id = st.selectbox(
                        "Velg overgang som skal slettes:",
                        options=del_opts,
                        format_func=lambda x: next(f"GW {t['gw']}: ❌ {t['out_name']} ➔ ✅ {t['in_name']} ({t['status']})" for t in evaluated_list if t['id'] == x),
                        key="sel_del_transfer"
                    )
                with c_del_btn:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Slett overgang", key="btn_del_transfer", width="stretch"):
                        del_obj = next((t for t in evaluated_list if t['id'] == chosen_del_id), None)
                        delete_tracked_transfer(chosen_del_id)
                        if del_obj and 'applied_transfers' in st.session_state:
                            tup = (int(del_obj['out_id']), int(del_obj['in_id']))
                            if tup in st.session_state['applied_transfers']:
                                st.session_state['applied_transfers'].remove(tup)
                        st.success("Overgang slettet fra loggen!")
                        st.rerun()

        else:
            st.info(
                "ℹ️ **Ingen overganger er logget ennå.**  \n"
                "Når du trykker på **«✅ Registrer dette byttet i appen»** på en av anbefalingene ovenfor, "
                "blir overgangen automatisk lagret her og evaluert mot faktiske poeng etter runden!"
            )

        with st.expander("➕ Loggfør en overgang manuelt (f.eks. fra tidligere runder)", expanded=False):
            c_man1, c_man2, c_man3 = st.columns(3)
            with c_man1:
                man_gw = st.number_input("Gameweek / Runde:", min_value=1, max_value=30, value=int(next_gw), step=1, key="man_tx_gw")
            with c_man2:
                sorted_all_p = df_players.sort_values(by='web_name')
                man_out_id = st.selectbox(
                    "Spiller UT (Solgt):",
                    options=sorted_all_p['id'].tolist(),
                    format_func=lambda x: f"{df_players.loc[df_players['id']==x, 'web_name'].values[0]} ({df_players.loc[df_players['id']==x, 'team_short'].values[0]})",
                    key="man_tx_out"
                )
            with c_man3:
                man_in_id = st.selectbox(
                    "Spiller INN (Kjøpt):",
                    options=sorted_all_p['id'].tolist(),
                    format_func=lambda x: f"{df_players.loc[df_players['id']==x, 'web_name'].values[0]} ({df_players.loc[df_players['id']==x, 'team_short'].values[0]})",
                    key="man_tx_in"
                )
            c_man_hit, c_man_btn = st.columns([1, 2])
            with c_man_hit:
                man_hit = st.selectbox("Minuspoeng:", [0, 4, 8], format_func=lambda x: f"-{x} hit" if x > 0 else "0 (Gratis)", key="man_tx_hit")
            with c_man_btn:
                st.write("")
                st.write("")
                if st.button("➕ Legg til i overgangslogg", key="btn_add_manual_tx", width="stretch"):
                    out_row = df_players[df_players['id'] == man_out_id].iloc[0]
                    in_row = df_players[df_players['id'] == man_in_id].iloc[0]
                    exp_diff = round(in_row['total_xp'] - out_row['total_xp'] - man_hit, 2)
                    add_tracked_transfer(
                        gw=man_gw,
                        out_id=out_row['id'], out_name=out_row['web_name'], out_team=out_row['team_short'],
                        in_id=in_row['id'], in_name=in_row['web_name'], in_team=in_row['team_short'],
                        exp_gain=exp_diff, hit_cost=man_hit
                    )
                    tx_tuple = (int(out_row['id']), int(in_row['id']))
                    if 'applied_transfers' not in st.session_state:
                        st.session_state['applied_transfers'] = []
                    if tx_tuple not in st.session_state['applied_transfers']:
                        st.session_state['applied_transfers'].append(tx_tuple)
                    st.success(f"Overgang logget for GW {man_gw}: {out_row['web_name']} ➔ {in_row['web_name']}! Mitt Lag er nå oppdatert.")
                    st.rerun()

    # ==========================================================================
    # FANE 3: SPILLERDATABASE / KALKULATOR
    # ==========================================================================
    with tab3:
        st.subheader("📊 Komplett Spillerdatabase med xP & FDR")
        st.caption("Søk og filtrer blant alle Eliteserien Fantasy-spillere for å planlegge fremover.")

        # Hurtigvelger for spillerprofil
        with st.expander("🔍 Åpne full profil, sesonggrafer og historikk for en spiller", expanded=False):
            c_sel_p3, c_btn_p3 = st.columns([3, 1])
            with c_sel_p3:
                sorted_p3 = df_players.sort_values(by='web_name')
                chosen_p3 = st.selectbox(
                    "Velg spiller for detaljanalyse:",
                    options=sorted_p3['id'].tolist(),
                    format_func=lambda x: f"{df_players.loc[df_players['id']==x, 'web_name'].values[0]} ({df_players.loc[df_players['id']==x, 'team_short'].values[0]} - {df_players.loc[df_players['id']==x, 'pos_short'].values[0]} | {df_players.loc[df_players['id']==x, 'now_cost'].values[0]}m)",
                    key="quick_inspect_player_tab3"
                )
            with c_btn_p3:
                st.write("")
                st.write("")
                if st.button("Åpne profil & grafer", key="btn_open_quick_inspect_tab3", width="stretch"):
                    render_player_dialog(chosen_p3, df_players, bootstrap, eval_horizon, events, team_fixtures, manager_id=manager_id)
        
        # Filtre
        with st.container(border=True):
            st.markdown("##### 🎛️ Søk- og filterpanel")
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                pos_filter = st.selectbox("Posisjon:", ["Alle", "Keeper", "Forsvar", "Midtbane", "Angrep"])
            with f_col2:
                team_names = ["Alle"] + sorted(list(set(df_players['team_name'])))
                team_filter = st.selectbox("Klubb:", team_names)
            with f_col3:
                max_p = float(df_players['now_cost'].max())
                price_filter = st.slider("Maks pris (mill):", 4.0, max_p, max_p, 0.5)
            with f_col4:
                search_query = st.text_input("Søk etter navn:", "")

            f_col5, f_col6 = st.columns(2)
            with f_col5:
                only_available = st.checkbox("Vis kun skadefrie spillere (100% spilletid)", value=False)
            with f_col6:
                only_my_team = st.checkbox("Vis kun spillere i mitt lag", value=False)

        # Filtrering
        filtered_df = df_players.copy()
        
        if pos_filter != "Alle":
            pos_map = {"Keeper": 1, "Forsvar": 2, "Midtbane": 3, "Angrep": 4}
            filtered_df = filtered_df[filtered_df['element_type'] == pos_map[pos_filter]]
            
        if team_filter != "Alle":
            filtered_df = filtered_df[filtered_df['team_name'] == team_filter]
            
        filtered_df = filtered_df[filtered_df['now_cost'] <= price_filter]
        
        if search_query:
            q = search_query.lower()
            filtered_df = filtered_df[
                filtered_df['web_name'].str.lower().str.contains(q) |
                filtered_df['first_name'].str.lower().str.contains(q) |
                filtered_df['second_name'].str.lower().str.contains(q)
            ]
            
        if only_available:
            filtered_df = filtered_df[filtered_df['status'] == 'a']
            
        if only_my_team:
            filtered_df = filtered_df[filtered_df['id'].isin(squad_pids)]

        # Legg til indikator for om spiller er i mitt lag
        filtered_df['Mitt Lag'] = filtered_df['id'].apply(lambda x: "⭐ Ja" if x in squad_pids else "")

        next_gw_name = get_event_display_name(next_gw, events)
        display_cols = [
            'logo', 'web_name', 'team_short', 'pos_short', 'now_cost', 'selected_by_percent',
            'form', 'ppg', 'next_fixture', 'next_fdr', 'next_gw_xp', 'total_xp', 'status_desc', 'news', 'Mitt Lag'
        ]
        
        table_data = filtered_df[display_cols].sort_values(by='total_xp', ascending=False)
        table_data.columns = [
            'Logo', 'Spiller', 'Klubb', 'Pos', 'Pris (m)', 'Eierandel %',
            'Form', 'PPG', 'Neste Kamp', 'FDR', f'xP {next_gw_name}', f'xP ({eval_horizon} Runder)', 'Status', 'Skademelding', 'I Mitt Lag'
        ]

        st.dataframe(
            table_data,
            width="stretch",
            hide_index=True,
            column_config={
                "Logo": st.column_config.ImageColumn("Logo", width="small"),
                "Pris (m)": st.column_config.NumberColumn(format="%.1f m"),
                "Eierandel %": st.column_config.NumberColumn(format="%.1f %%"),
                "Form": st.column_config.NumberColumn(format="%.1f"),
                "PPG": st.column_config.NumberColumn(format="%.1f"),
                f"xP {next_gw_name}": st.column_config.NumberColumn(format="%.2f"),
                f"xP ({eval_horizon} Runder)": st.column_config.NumberColumn(format="%.2f"),
            }
        )

        st.caption(f"Viser {len(table_data)} av {len(df_players)} spillere.")

    # ==========================================================================
    # FANE 4: MINE LIGAER & TABELLER
    # ==========================================================================
    with tab4:
        st.subheader("🏆 Mine Ligaer & Rangering")
        st.caption(f"Full oversikt over stillingen i dine private venneligaer og offisielle ligaer for {team_name} (ID: {manager_id}).")

        leagues_data = entry.get('leagues', {})
        classic_leagues = leagues_data.get('classic', [])

        if not classic_leagues:
            st.info("Fant ingen registrerte ligaer for dette laget.")
        else:
            # Bygg oversikt over alle ligaer
            league_summary_rows = []
            for l in classic_leagues:
                is_private = (l.get('league_type') == 'x')
                type_label = "🔒 Privat Venneliga" if is_private else "🌐 Offisiell Liga"
                
                cur_rank = l.get('entry_rank', 0)
                last_rank = l.get('entry_last_rank', cur_rank)
                total_entries = l.get('rank_count', 0)
                
                # Trend
                if last_rank is None or last_rank == 0 or last_rank == cur_rank:
                    trend = "➖ Uendret"
                elif cur_rank < last_rank:
                    diff = last_rank - cur_rank
                    trend = f"🟢 +{diff}"
                else:
                    diff = cur_rank - last_rank
                    trend = f"🔴 -{diff}"
                    
                pct = l.get('entry_percentile_rank')
                pct_str = f"Topp {pct} %" if pct is not None else "-"
                
                league_summary_rows.append({
                    'id': l['id'],
                    'Liganavn': l.get('name', 'Ukjent'),
                    'Type': type_label,
                    'Plassering': f"#{cur_rank:,}".replace(",", " "),
                    'Totalt Lag': f"{total_entries:,}".replace(",", " "),
                    'Endring': trend,
                    'Toppsjikt': pct_str,
                    'raw_rank': cur_rank,
                    'raw_total': total_entries,
                    'is_private': is_private
                })

            df_l_summary = pd.DataFrame(league_summary_rows)

            # Vis sammendrag
            st.markdown("### 📋 Alle Registrerte Ligaer")
            st.dataframe(
                df_l_summary[['Liganavn', 'Type', 'Plassering', 'Totalt Lag', 'Endring', 'Toppsjikt']],
                width="stretch",
                hide_index=True
            )

            # Interaktiv Ligatabell
            st.markdown("---")
            st.markdown("### 🔍 Live Ligatabell")
            
            # Prioriter private venneligaer først i dropdown
            df_sorted_opts = df_l_summary.sort_values(by=['is_private', 'raw_total'], ascending=[False, True])
            league_options = {
                row['id']: f"{row['Liganavn']} ({row['Type']}) — Plassering: {row['Plassering']} av {row['Totalt Lag']}"
                for _, row in df_sorted_opts.iterrows()
            }
            
            with st.container(border=True):
                st.markdown("##### 🏆 Velg liga for live tabell")
                selected_lid = st.selectbox(
                    "Velg liga for å åpne tabell og se avstand til konkurrentene:",
                    options=list(league_options.keys()),
                    format_func=lambda x: league_options[x]
                )

            if selected_lid:
                with st.spinner("Henter tabell for valgt liga..."):
                    standings_data = fetch_league_standings(selected_lid)

                if standings_data and 'standings' in standings_data:
                    results = standings_data['standings'].get('results', [])
                    if results:
                        st_rows = []
                        my_row = None
                        leader_pts = results[0].get('total', 0) if results else 0
                        podium_pts = results[min(2, len(results) - 1)].get('total', 0) if len(results) >= 3 else leader_pts

                        for r in results:
                            is_me = (r.get('entry') == manager_id)
                            tot = r.get('total', 0)
                            diff_leader = tot - leader_pts
                            diff_str = f"{diff_leader:+d} pts" if diff_leader != 0 else "Leder 🥇"
                            
                            row_dict = {
                                'Rank': f"#{r.get('rank')}",
                                'Lag': f"⭐ {r.get('entry_name')}" if is_me else r.get('entry_name'),
                                'Manager': r.get('player_name'),
                                f'GW {current_gw} pts': r.get('event_total', 0),
                                'Totalpoeng': tot,
                                'Diff Leder': diff_str,
                                'is_me': is_me
                            }
                            st_rows.append(row_dict)
                            if is_me:
                                my_row = r

                        # Hvis mitt lag finnes i resultatene, vis nøkkelmetrikker
                        if my_row:
                            my_pts = my_row.get('total', 0)
                            my_rank = my_row.get('rank', 0)
                            gap_leader = leader_pts - my_pts
                            gap_podium = podium_pts - my_pts
                            
                            m_col1, m_col2, m_col3 = st.columns(3)
                            with m_col1:
                                st.metric("Din Plassering", f"#{my_rank}")
                            with m_col2:
                                st.metric("Avstand til Leder (1. plass)", f"-{gap_leader} pts" if gap_leader > 0 else "Du leder! 🏆")
                            with m_col3:
                                if my_rank <= 3:
                                    st.metric("Status Pallplass", "Du er på pallen! 🥉")
                                else:
                                    st.metric("Avstand til Pallplass (3. plass)", f"-{gap_podium} pts")

                        df_standings = pd.DataFrame(st_rows)
                        st.dataframe(
                            df_standings[['Rank', 'Lag', 'Manager', f'GW {current_gw} pts', 'Totalpoeng', 'Diff Leder']],
                            width="stretch",
                            hide_index=True
                        )
                        st.caption(f"Viser de {len(results)} beste lagene i {standings_data.get('league', {}).get('name', 'ligaen')}.")
                    else:
                        st.info("Ingen deltakere registrert i tabellen for denne ligaen.")
                else:
                    st.warning("Kunne ikke hente tabell for denne ligaen fra Eliteserien API.")

    # ==========================================================================
    # FANE 5: GRAFER & STATISTIKK
    # ==========================================================================
    with tab5:
        st.subheader("📈 Avansert Statistikk & Grafisk Analyse")
        st.caption("Visuelle grafer over sesongutvikling, forsprang mot ligasnitt, FDR-terminliste for alle 16 klubber, verdimatrise og troppens budsjettfordeling.")

        with st.spinner("Henter historiske data og genererer analyser..."):
            history_data = fetch_manager_history(manager_id)

        history_current = history_data.get('current', []) if history_data else []
        
        if not history_current:
            st.info("Ingen historiske data tilgjengelig for dette laget ennå (f.eks. før første runde er ferdigspilt).")
        else:
            events_data = bootstrap.get('events', [])
            event_avgs = {ev['id']: ev.get('average_entry_score', 0) for ev in events_data}

            h_rows = []
            cum_diff = 0
            for h in history_current:
                gw_idx = h['event']
                avg_s = event_avgs.get(gw_idx, 0)
                pts = h['points']
                diff = pts - avg_s
                cum_diff += diff
                h_rows.append({
                    'gw': gw_idx,
                    'Runde': f"GW {gw_idx}",
                    'Dine Poeng': pts,
                    'Snittpoeng': avg_s,
                    'Diff Snitt': diff,
                    'Akkumulert Forsprang': cum_diff,
                    'Totalpoeng': h['total_points'],
                    'Overall Rangering': h['overall_rank'],
                    'Lagverdi (m)': h['value'] / 10.0,
                    'Benkepoeng': h['points_on_bench'],
                    'Minuspoeng': h.get('event_transfers_cost', 0)
                })
            df_hist = pd.DataFrame(h_rows)

            best_round = df_hist.loc[df_hist['Dine Poeng'].idxmax()]
            worst_round = df_hist.loc[df_hist['Dine Poeng'].idxmin()]
            tot_diff_pts = df_hist['Diff Snitt'].sum()
            tot_hit_pts = df_hist['Minuspoeng'].sum()
            tot_bench_pts = df_hist['Benkepoeng'].sum()
            avg_my_pts = df_hist['Dine Poeng'].mean()
            avg_league_pts = df_hist['Snittpoeng'].mean()

            # Nøkkeltall-kort
            st.markdown("### 🏆 Sesongstatistikk i Tall")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
            with kpi_col1:
                st.metric("Beste Runde", f"{best_round['Dine Poeng']} pts", f"GW {best_round['gw']} 🚀")
            with kpi_col2:
                st.metric("Laveste Runde", f"{worst_round['Dine Poeng']} pts", f"GW {worst_round['gw']}")
            with kpi_col3:
                diff_sign = "+" if tot_diff_pts >= 0 else ""
                st.metric("Forsprang vs. Snitt", f"{diff_sign}{tot_diff_pts} pts", f"Snitt {avg_my_pts:.1f} vs {avg_league_pts:.1f}")
            with kpi_col4:
                hits_taken = int(tot_hit_pts / 4) if tot_hit_pts > 0 else 0
                st.metric("Tapt på Hits (-4)", f"-{tot_hit_pts} pts", f"{hits_taken} ekstrabytter")
            with kpi_col5:
                st.metric("Poeng på Benken", f"{tot_bench_pts} pts", f"Snitt {(tot_bench_pts / len(df_hist)):.1f} / runde")

            st.markdown("---")

            # ------------------------------------------------------------------
            # SEKSJON 1: INTERAKTIVE SESONGGRAFER
            # ------------------------------------------------------------------
            with st.container(border=True):
                st.markdown("##### 📊 Velg visualisering")
                graph_choice = st.radio(
                    "Visningsmodus:",
                    [
                        "📊 Dine Poeng vs. Gjennomsnitt per Runde",
                        "📈 Sammenlagtrangering over Tid (Overall Rank)",
                        "🏔️ Akkumulert Poengforsprang til Snittet"
                    ],
                    horizontal=True
                )

            if graph_choice == "📊 Dine Poeng vs. Gjennomsnitt per Runde":
                df_melt = df_hist.melt(
                    id_vars=['gw', 'Runde', 'Diff Snitt'],
                    value_vars=['Dine Poeng', 'Snittpoeng'],
                    var_name='Kategori',
                    value_name='Poeng'
                )
                chart_bars = alt.Chart(df_melt).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X('gw:O', title='Gameweek (Runde)'),
                    y=alt.Y('Poeng:Q', title='Poeng i Runden'),
                    color=alt.Color(
                        'Kategori:N',
                        scale=alt.Scale(domain=['Dine Poeng', 'Snittpoeng'], range=['#1b84ff', '#94a3b8']),
                        legend=alt.Legend(title="Kategori", orient="top")
                    ),
                    xOffset='Kategori:N',
                    tooltip=[
                        alt.Tooltip('Runde:N', title='Runde'),
                        alt.Tooltip('Kategori:N', title='Kategori'),
                        alt.Tooltip('Poeng:Q', title='Poeng'),
                        alt.Tooltip('Diff Snitt:Q', title='Differanse mot Snitt')
                    ]
                ).properties(height=340)
                st.altair_chart(chart_bars, width="stretch")

            elif graph_choice == "📈 Sammenlagtrangering over Tid (Overall Rank)":
                chart_rank = alt.Chart(df_hist).mark_line(
                    point=alt.OverlayMarkDef(filled=True, size=60, color='#10b981'),
                    color='#10b981',
                    strokeWidth=3
                ).encode(
                    x=alt.X('gw:O', title='Gameweek (Runde)'),
                    y=alt.Y('Overall Rangering:Q', scale=alt.Scale(reverse=True), title='Sammenlagtplassering (høyere på grafen = bedre rank)'),
                    tooltip=[
                        alt.Tooltip('Runde:N', title='Runde'),
                        alt.Tooltip('Overall Rangering:Q', title='Plassering', format=','),
                        alt.Tooltip('Totalpoeng:Q', title='Totalpoeng'),
                        alt.Tooltip('Lagverdi (m):Q', title='Lagverdi (m)')
                    ]
                ).properties(height=340)
                st.altair_chart(chart_rank, width="stretch")

            else:
                chart_cum = alt.Chart(df_hist).mark_area(
                    line={'color': '#6366f1', 'strokeWidth': 2.5},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[
                            alt.GradientStop(color='#6366f1', offset=0),
                            alt.GradientStop(color='rgba(99, 102, 241, 0.05)', offset=1)
                        ],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    x=alt.X('gw:O', title='Gameweek (Runde)'),
                    y=alt.Y('Akkumulert Forsprang:Q', title='Akkumulerte Poeng Foran/Bak Ligasnitt'),
                    tooltip=[
                        alt.Tooltip('Runde:N', title='Runde'),
                        alt.Tooltip('Akkumulert Forsprang:Q', title='Netto forsprang'),
                        alt.Tooltip('Totalpoeng:Q', title='Totalpoeng')
                    ]
                ).properties(height=340)
                st.altair_chart(chart_cum, width="stretch")

            with st.expander("🔎 Vis komplett sesongtabell runde for runde"):
                st.dataframe(
                    df_hist[['Runde', 'Dine Poeng', 'Snittpoeng', 'Diff Snitt', 'Akkumulert Forsprang', 'Totalpoeng', 'Overall Rangering', 'Minuspoeng', 'Benkepoeng', 'Lagverdi (m)']],
                    width="stretch",
                    hide_index=True
                )

        # ------------------------------------------------------------------
        # SEKSJON 2: TERMINLISTE-TICKER & FDR HEATMAP (NESTE 5 RUNDER)
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🗓️ Terminliste-Ticker & Kampstyrke (Neste 5 Runder)")
        st.caption("FDR (Fixture Difficulty Rating 1-5): 🟢 1-2 (Lett) | ⚪ 3 (Middels) | 🟠 4 (Tøff) | 🔴 5 (Veldig tøff). Sortert etter enklest gjennomsnittlig kampprogram.")

        teams_dict_all = {t['id']: t for t in bootstrap.get('teams', []) if t['id'] <= 16}
        ticker_data_rows = []
        horizon_gws = [next_gw + i for i in range(5)]

        for tid, t in teams_dict_all.items():
            t_club = get_club_style(t['short_name'])
            row_t = {
                'Logo': t_club.get('logo', ''),
                'Klubb': t['name'],
                'Kortnavn': t['short_name'],
                'team_id': tid
            }
            team_fdrs = []
            for gw_target in horizon_gws:
                gw_col_name = get_event_display_name(gw_target, events)
                gw_fixs = team_fixtures.get(tid, {}).get(gw_target, [])
                if not gw_fixs:
                    row_t[gw_col_name] = "Blank ❌"
                    team_fdrs.append(3.0)
                else:
                    fix_strings = []
                    for fx in gw_fixs:
                        side_str = "H" if fx['is_home'] else "B"
                        fix_strings.append(f"{fx['opp_name']} ({side_str}) [FDR {fx['fdr']}]")
                        team_fdrs.append(fx['fdr'])
                    row_t[gw_col_name] = " / ".join(fix_strings)

            row_t['Snitt FDR'] = round(sum(team_fdrs) / len(team_fdrs), 2) if team_fdrs else 3.0
            ticker_data_rows.append(row_t)

        df_full_ticker = pd.DataFrame(ticker_data_rows).sort_values(by='Snitt FDR')

        # Filter for egne klubber
        squad_team_ids = set(squad_df['team_id'].unique())
        c_tick1, c_tick2 = st.columns([2, 1])
        with c_tick1:
            filter_my_clubs = st.checkbox("Vis kun klubber representert i min tropp", value=False)
        
        display_ticker = df_full_ticker[df_full_ticker['team_id'].isin(squad_team_ids)] if filter_my_clubs else df_full_ticker

        # Callouts for beste og tøffeste kampprogram
        best_3_teams = df_full_ticker.iloc[:3]['Klubb'].tolist()
        worst_3_teams = df_full_ticker.iloc[-3:]['Klubb'].tolist()

        c_call1, c_call2 = st.columns(2)
        with c_call1:
            st.success(f"🟢 **Beste kampprogram neste 5 runder (Kjøpsanbefaling):**  \n**{', '.join(best_3_teams)}**")
        with c_call2:
            st.error(f"🔴 **Tøffeste kampprogram neste 5 runder (Styr unna/selg):**  \n**{', '.join(worst_3_teams)}**")

        ticker_cols = ['Logo', 'Klubb', 'Snitt FDR'] + [get_event_display_name(gw_target, events) for gw_target in horizon_gws]
        st.dataframe(
            display_ticker[ticker_cols],
            column_config={
                "Logo": st.column_config.ImageColumn("Logo", width="small")
            },
            width="stretch",
            hide_index=True
        )

        # ------------------------------------------------------------------
        # SEKSJON 3: VERDI VS. FORVENTEDE POENG (xP MATRISE)
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 💎 Verdi vs. Forventede Poeng (xP Matrise)")
        st.caption(f"Sammenlign spillerpris mot forventede poeng (xP) over neste {eval_horizon} runder. Se hvilke spillere som gir mest 'valuta for pengene', og hvor dine spillere plasserer seg!")

        df_scatter_all = df_players[df_players['status'] != 'i'].copy()
        squad_id_set = set(squad_pids)
        df_scatter_all['Eierskap'] = df_scatter_all['id'].apply(lambda x: "Ditt Lag ⭐" if x in squad_id_set else "Andre Spillere")

        with st.container(border=True):
            st.markdown("##### 💎 Filter for verdimatrise")
            c_scat_f1, c_scat_f2 = st.columns([1, 1])
            with c_scat_f1:
                pos_filter = st.selectbox(
                    "Filtrer posisjon:",
                    ["Alle Posisjoner", "Keepere (GK)", "Forsvar (DEF)", "Midtbane (MID)", "Angrep (FWD)"]
                )
            with c_scat_f2:
                min_xp_filter = st.slider("Minste forventede poeng (xP):", min_value=0.0, max_value=20.0, value=3.0, step=0.5)

        filtered_scatter = df_scatter_all[df_scatter_all['total_xp'] >= min_xp_filter].copy()
        if pos_filter == "Keepere (GK)":
            filtered_scatter = filtered_scatter[filtered_scatter['element_type'] == 1]
        elif pos_filter == "Forsvar (DEF)":
            filtered_scatter = filtered_scatter[filtered_scatter['element_type'] == 2]
        elif pos_filter == "Midtbane (MID)":
            filtered_scatter = filtered_scatter[filtered_scatter['element_type'] == 3]
        elif pos_filter == "Angrep (FWD)":
            filtered_scatter = filtered_scatter[filtered_scatter['element_type'] == 4]

        scatter_chart = alt.Chart(filtered_scatter).mark_circle().encode(
            x=alt.X('now_cost:Q', title='Pris (millioner kr)', scale=alt.Scale(zero=False)),
            y=alt.Y('total_xp:Q', title=f'Forventede Poeng (xP Neste {eval_horizon} Runder)'),
            color=alt.Color(
                'pos_short:N',
                title='Posisjon',
                scale=alt.Scale(domain=['GK', 'DEF', 'MID', 'FWD'], range=['#f59e0b', '#3b82f6', '#10b981', '#ec4899'])
            ),
            size=alt.Size(
                'Eierskap:N',
                scale=alt.Scale(domain=['Andre Spillere', 'Ditt Lag ⭐'], range=[60, 260]),
                legend=alt.Legend(title="Dine Spillere", orient="top")
            ),
            tooltip=[
                alt.Tooltip('web_name:N', title='Spiller'),
                alt.Tooltip('team_short:N', title='Klubb'),
                alt.Tooltip('pos_short:N', title='Posisjon'),
                alt.Tooltip('now_cost:Q', title='Pris (m)'),
                alt.Tooltip('total_xp:Q', title='Total xP'),
                alt.Tooltip('next_fixture:N', title='Neste Kamp'),
                alt.Tooltip('Eierskap:N', title='Eierskap')
            ]
        ).properties(height=400)

        st.altair_chart(scatter_chart, width="stretch")

        # Topp 5 Verdispillere i utvalget
        filtered_scatter['xp_per_m'] = (filtered_scatter['total_xp'] / filtered_scatter['now_cost']).round(2)
        top_value_picks = filtered_scatter.sort_values(by='xp_per_m', ascending=False).head(5)
        st.markdown("##### 💡 Topp 5 'Mest xP per million' i dette utvalget:")
        st.dataframe(
            top_value_picks[['web_name', 'team_short', 'pos_short', 'now_cost', 'total_xp', 'xp_per_m', 'next_fixture', 'Eierskap']].rename(
                columns={'web_name': 'Spiller', 'team_short': 'Klubb', 'pos_short': 'Pos', 'now_cost': 'Pris (m)', 'total_xp': f'xP ({eval_horizon}r)', 'xp_per_m': 'xP / Mill'}
            ),
            width="stretch",
            hide_index=True
        )

        # ------------------------------------------------------------------
        # SEKSJON 4: TROPPENS KAPITALFORDELING & ROI PER POSISJON
        # ------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### ⚖️ Troppens Budsjettfordeling & Posisjonseffektivitet")
        st.caption("Hvordan fordeler lagverdien din seg over de ulike lagdelene, og hvilken posisjon leverer høyest avkastning (xP per million)?")

        pos_group = squad_df.groupby('pos_short').agg(
            Antall=('id', 'count'),
            Total_Pris=('now_cost', 'sum'),
            Total_xP=('total_xp', 'sum')
        ).reset_index()
        pos_group['xP_per_Mill'] = (pos_group['Total_xP'] / pos_group['Total_Pris']).round(2)
        pos_group['Total_Pris'] = pos_group['Total_Pris'].round(1)
        pos_group['Total_xP'] = pos_group['Total_xP'].round(1)

        c_roi1, c_roi2 = st.columns([1, 1])
        with c_roi1:
            st.markdown("##### 💰 Investert Kapital per Lagdel (mill. kr)")
            chart_cost = alt.Chart(pos_group).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color='#3b82f6').encode(
                x=alt.X('pos_short:N', title='Posisjon', sort=['GK', 'DEF', 'MID', 'FWD']),
                y=alt.Y('Total_Pris:Q', title='Investert Kapital (mill)'),
                tooltip=['pos_short:N', 'Antall:Q', 'Total_Pris:Q']
            ).properties(height=260)
            st.altair_chart(chart_cost, width="stretch")

        with c_roi2:
            st.markdown("##### ⚡ Forventet Avkastning (xP per Investert Million)")
            chart_roi = alt.Chart(pos_group).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color='#10b981').encode(
                x=alt.X('pos_short:N', title='Posisjon', sort=['GK', 'DEF', 'MID', 'FWD']),
                y=alt.Y('xP_per_Mill:Q', title='xP per Million'),
                tooltip=['pos_short:N', 'Total_xP:Q', 'xP_per_Mill:Q']
            ).properties(height=260)
            st.altair_chart(chart_roi, width="stretch")


if __name__ == "__main__":
    main()
