# 🇳🇴 Eliteserien Fantasy Predictor & Transfer Optimizer

En komplett web-applikasjon i Streamlit for **Eliteserien Fantasy** med:
- **Prediksjonsmodell for forventede poeng (xP):** Basert på form, tabellstyrke, dynamisk Fixture Difficulty Rating (FDR 1-5), hjemmebanefordel (+12 %), offensive faktorer og skadestatus.
- **PuLP Mixed-Integer Linear Programming (MILP):** Matematisk optimering av startellever, benk, og overganger innenfor Eliteseriens offisielle regler (budsjett, maks 3 spillere per klubb, formasjoner).
- **Hit-kalkulator (-4):** Matematisk evaluering over 1 runde vs. 3+ runder med tydelig anbefaling (**ANBEFALT HIT** vs. **IKKE VERDT -4**).
- **Interaktiv fotballbane:** Visuell startellever med kapteins- og visekapteinsmerker, FDR-fargekoder og prioritert benk.
- **Komplett spillerdatabase:** Sortering og søk på tvers av hele ligaen.

---

## 🚀 Installasjon og Oppstart

### 1. Installer nødvendige biblioteker
Kjør følgende kommando i terminalen:
```bash
pip install streamlit requests pandas numpy pulp
```

### 2. Start applikasjonen
Naviger til mappen og kjør:
```bash
streamlit run app.py
```
Eller dobbeltklikk på `run.bat` (på Windows).

Nettleseren åpner automatisk appen på `http://localhost:8501`.

---

## 📁 Struktur og Filer
- `app.py`: Komplett applikasjon med datahenting, caching, xP-modell, PuLP-motor og Streamlit UI.
- `run.bat`: Windows oppstartsskript for enkel kjøring.
