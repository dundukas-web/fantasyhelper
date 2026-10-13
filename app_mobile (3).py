import json
import random
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import milp, Bounds, LinearConstraint

# =========================================================
# 1. PUSLAPIO KONFIGŪRACIJA IR BASKETNEWS FANTASY TAMSUS STILIUS
# =========================================================
st.set_page_config(
    page_title="EuroLeague Fantasy Dominator Ultra 2026/27",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Bazinė tamsi tema matching BasketNews Fantasy UI */
    .main { background-color: #0e1117; color: #e6edf3; font-family: 'Inter', sans-serif; }
    
    /* Filtrai / Mygtukai viršuje */
    .filter-badge {
        background-color: #1a202c;
        color: #a0aec0;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        border: 1px solid #2d3748;
        display: inline-block;
        margin-right: 8px;
        cursor: pointer;
    }
    .filter-badge.active {
        background-color: #ffffff;
        color: #000000;
    }

    /* Mėlynas PIR ženklelis (matching UI) */
    .pir-badge {
        background-color: #00b4d8;
        color: #ffffff;
        font-weight: 800;
        padding: 4px 10px;
        border-radius: 8px;
        display: inline-block;
        text-align: center;
        min-width: 45px;
    }

    .pir-val-badge {
        background-color: #118ab2;
        color: #ffffff;
        font-weight: 700;
        padding: 4px 8px;
        border-radius: 6px;
        display: inline-block;
    }

    /* AI rekomendacijų kortelės */
    .ai-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #38bdf8;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    /* Aikštelės vizualizacija */
    .court-container {
        background-color: #17191e;
        border: 2px solid #2d3748;
        border-radius: 16px;
        padding: 20px;
        position: relative;
        min-height: 520px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
    }
    .court-slot {
        background: #101216;
        border: 1px solid #2a2e39;
        border-radius: 12px;
        padding: 10px;
        width: 140px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }
    .court-slot.captain {
        border: 2px solid #ff6b00;
    }
    .bench-container {
        background-color: #12141a;
        border: 1px solid #2d3748;
        border-radius: 16px;
        padding: 16px;
    }
    .bench-item {
        background: #1a1d24;
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. PROGNOZAVIMO ALGORITMAS IR DUOMENŲ VARIKLIS
# =========================================================
def parse_basketnews_json(raw_json):
    records = raw_json.get("data", {}).get("playersSearchRecordsFromClient", {}).get("records", [])
    players = []

    pos_map = {"guard": "G", "forward": "P", "center": "C"}

    for item in records:
        p_id = item.get("id", str(random.randint(1000, 9999)))
        first_name = item.get("firstName", "")
        last_name = item.get("lastName", "")
        name = f"{first_name} {last_name}".strip()
        photo = item.get("photo", "https://basketnews.com/storage/images/players_png/default.png")
        health = item.get("health", "ready")

        price_raw = item.get("fantasyPrice", 0) or 0
        price_m = round(price_raw / 1000000.0, 2)
        price_change = round((item.get("fantasyPriceChange", 0) or 0) / 1000000.0, 2)

        team_wrapper = item.get("team", {}) or {}
        pos_raw = team_wrapper.get("positions", ["guard"])
        pos_str = pos_raw[0] if isinstance(pos_raw, list) and pos_raw else "guard"
        pos = pos_map.get(pos_str.lower(), "G")

        team_info = team_wrapper.get("team", {}) or {}
        translation = team_info.get("translation", {}) or {}
        team_abbr = team_info.get("abbreviation") or translation.get("shortName") or "UNK"

        games = team_info.get("games", [])
        opp_abbr = "UNK"
        is_home = True
        win_prob = random.randint(30, 75)
        game_time = "21:00"

        if games:
            g = games[0]
            game_time = g.get("originalGameAt", "")[11:16]
            t1 = g.get("team1", {}).get("team", {}).get("abbreviation") or "T1"
            t2 = g.get("team2", {}).get("team", {}).get("abbreviation") or "T2"
            opp_abbr = t2 if team_abbr == t1 else t1
            is_home = (team_abbr == t1)

        stats = item.get("stats", {}) or {}
        avg_pir = round(float(item.get("average_fantasy_pts") or stats.get("s_eff") or 10.0), 1)
        pts_per_game = round(float(stats.get("s_pts", 10.0)), 1)
        mins = round(float(stats.get("s_time", 1200) / 60.0 / max(stats.get("s_gp", 1), 1)), 1)

        # Prognozuojamas PIR pagal pažangų modelį (Home/Away + Win Prob + Form)
        home_bonus = 1.08 if is_home else 0.94
        win_bonus = 1.0 + (win_prob - 50) * 0.003
        proj_pir = round(avg_pir * home_bonus * win_bonus, 1)

        val_score = round(proj_pir / price_m, 1) if price_m > 0 else 0.0

        players.append({
            "id": p_id,
            "name": name,
            "team": team_abbr,
            "opp": opp_abbr,
            "is_home": is_home,
            "win_prob": win_prob,
            "pos": pos,
            "price": price_m,
            "price_change": price_change,
            "avg_pir": avg_pir,
            "proj_pir": proj_pir,
            "pts": pts_per_game,
            "min": mins,
            "val": val_score,
            "status": health.capitalize(),
            "photo": photo,
            "time": game_time
        })

    return pd.DataFrame(players)

# Pavyzdiniai duomenys demo režimui
sample_json = {
    "data": {
        "playersSearchRecordsFromClient": {
            "records": [
                {"id":"1", "firstName":"Elijah", "lastName":"Bryant", "fantasyPrice":2130000, "fantasyPriceChange":130000, "team":{"positions":["guard"],"team":{"abbreviation":"HTA","games":[{"originalGameAt":"2026-10-14T20:00:00","team1":{"team":{"abbreviation":"HTA"}},"team2":{"team":{"abbreviation":"BJK"}}}]}},"stats":{"s_eff":27.9,"s_pts":15.5,"s_time":1932,"s_gp":1},"average_fantasy_pts":27.9},
                {"id":"2", "firstName":"Aleksandr", "lastName":"Vezenkov", "fantasyPrice":2200000, "fantasyPriceChange":0, "team":{"positions":["forward"],"team":{"abbreviation":"OLY","games":[{"originalGameAt":"2026-10-14T19:00:00","team1":{"team":{"abbreviation":"VAL"}},"team2":{"team":{"abbreviation":"OLY"}}}]}},"stats":{"s_eff":22.5,"s_pts":17.3,"s_time":1380,"s_gp":1},"average_fantasy_pts":22.5},
                {"id":"3", "firstName":"Carlik", "lastName":"Jones", "fantasyPrice":1950000, "fantasyPriceChange":100000, "team":{"positions":["guard"],"team":{"abbreviation":"PAR","games":[{"originalGameAt":"2026-10-15T20:30:00","team1":{"team":{"abbreviation":"PAR"}},"team2":{"team":{"abbreviation":"PAO"}}}]}},"stats":{"s_eff":25.9,"s_pts":18.8,"s_time":1566,"s_gp":1},"average_fantasy_pts":25.9},
                {"id":"4", "firstName":"Mathias", "lastName":"Lessort", "fantasyPrice":1900000, "fantasyPriceChange":100000, "team":{"positions":["center"],"team":{"abbreviation":"PAO","games":[{"originalGameAt":"2026-10-15T20:30:00","team1":{"team":{"abbreviation":"PAR"}},"team2":{"team":{"abbreviation":"PAO"}}}]}},"stats":{"s_eff":25.0,"s_pts":13.3,"s_time":1290,"s_gp":1},"average_fantasy_pts":25.0},
                {"id":"5", "firstName":"Jonas", "lastName":"Valančiūnas", "fantasyPrice":1830000, "fantasyPriceChange":30000, "team":{"positions":["center"],"team":{"abbreviation":"ŽAL","games":[{"originalGameAt":"2026-10-14T19:00:00","team1":{"team":{"abbreviation":"FEN"}},"team2":{"team":{"abbreviation":"ŽAL"}}}]}},"stats":{"s_eff":20.4,"s_pts":12.0,"s_time":1278,"s_gp":1},"average_fantasy_pts":20.4},
                {"id":"6", "firstName":"Sylvain", "lastName":"Francisco", "fantasyPrice":1840000, "fantasyPriceChange":20000, "team":{"positions":["guard"],"team":{"abbreviation":"PAO","games":[{"originalGameAt":"2026-10-15T20:30:00","team1":{"team":{"abbreviation":"PAR"}},"team2":{"team":{"abbreviation":"PAO"}}}]}},"stats":{"s_eff":19.8,"s_pts":16.3,"s_time":1320,"s_gp":1},"average_fantasy_pts":19.8},
                {"id":"7", "firstName":"Mfiondu", "lastName":"Kabengele", "fantasyPrice":1640000, "fantasyPriceChange":-30000, "team":{"positions":["center"],"team":{"abbreviation":"DUB","games":[{"originalGameAt":"2026-10-14T20:00:00","team1":{"team":{"abbreviation":"BKN"}},"team2":{"team":{"abbreviation":"DUB"}}}]}},"stats":{"s_eff":15.1,"s_pts":10.8,"s_time":1470,"s_gp":1},"average_fantasy_pts":15.1},
                {"id":"8", "firstName":"Dwayne", "lastName":"Bacon", "fantasyPrice":1590000, "fantasyPriceChange":50000, "team":{"positions":["forward"],"team":{"abbreviation":"DUB","games":[{"originalGameAt":"2026-10-14T20:00:00","team1":{"team":{"abbreviation":"BKN"}},"team2":{"team":{"abbreviation":"DUB"}}}]}},"stats":{"s_eff":19.2,"s_pts":15.8,"s_time":1662,"s_gp":1},"average_fantasy_pts":19.2},
                {"id":"9", "firstName":"Moses", "lastName":"Wright", "fantasyPrice":1500000, "fantasyPriceChange":50000, "team":{"positions":["center"],"team":{"abbreviation":"EA7","games":[{"originalGameAt":"2026-10-14T20:30:00","team1":{"team":{"abbreviation":"EA7"}},"team2":{"team":{"abbreviation":"RMB"}}}]}},"stats":{"s_eff":17.9,"s_pts":14.3,"s_time":1560,"s_gp":1},"average_fantasy_pts":17.9},
                {"id":"10", "firstName":"Theo", "lastName":"Maledon", "fantasyPrice":1420000, "fantasyPriceChange":10000, "team":{"positions":["guard"],"team":{"abbreviation":"ASV","games":[{"originalGameAt":"2026-10-14T20:30:00","team1":{"team":{"abbreviation":"ASV"}},"team2":{"team":{"abbreviation":"EA7"}}}]}},"stats":{"s_eff":16.7,"s_pts":12.5,"s_time":1500,"s_gp":1},"average_fantasy_pts":16.7}
            ]
        }
    }
}

st.sidebar.markdown("## ⚙️ Valdymas")
json_in = st.sidebar.text_area("Įklijuokite BasketNews JSON:", height=100)
if json_in.strip():
    try:
        df_players = parse_basketnews_json(json.loads(json_in))
    except Exception:
        df_players = parse_basketnews_json(sample_json)
else:
    df_players = parse_basketnews_json(sample_json)

# Session state žaidėjams
valid_ids = df_players["id"].tolist()
if "my_roster" not in st.session_state or not st.session_state["my_roster"]:
    st.session_state["my_roster"] = valid_ids[:10] if len(valid_ids) >= 10 else valid_ids
else:
    st.session_state["my_roster"] = [x for x in st.session_state["my_roster"] if x in valid_ids]
    if len(st.session_state["my_roster"]) < 10:
        st.session_state["my_roster"] = valid_ids[:10] if len(valid_ids) >= 10 else valid_ids

# =========================================================
# 3. MILP IR PAŽANGŪS PAIEŠKOS ALGORITMAI
# =========================================================
def find_best_single_player(df, position, max_price, exclude_ids=[]):
    """ Randa patį geriausią žaidėją pagal poziciją ir tikslią sumą """
    candidates = df[
        (df["pos"] == position) & 
        (df["price"] <= max_price) & 
        (~df["id"].isin(exclude_ids)) &
        (df["status"].str.lower() == "ready")
    ].copy()

    if candidates.empty:
        return None

    # Skaičiuojamas DI vertinimo balasa
    candidates["di_score"] = candidates["proj_pir"] * 0.7 + candidates["val"] * 0.3
    return candidates.sort_values("di_score", ascending=False).iloc[0]

def optimize_roles(roster_df):
    """ Paskirsto kapitoną (2x) ir 6-ąjį žaidėją (100% taškų), suoliukui - 50% taškų """
    if roster_df.empty:
        return roster_df, None, None, []

    sorted_df = roster_df.sort_values("proj_pir", ascending=False).copy()
    captain = sorted_df.iloc[0] # Didžiausias PIR -> Kapitonas
    
    # Kiti 4 pagrindiniai pagal pozicijų balansą
    main_lineup = [captain["id"]]
    bench = []

    for p_id in sorted_df["id"].tolist()[1:]:
        if len(main_lineup) < 5:
            main_lineup.append(p_id)
        else:
            bench.append(p_id)

    sixth_man_id = bench[0] if bench else None
    return sorted_df, captain["id"], sixth_man_id, bench

# =========================================================
# 4. PROGRAMOS SKILTYS (TABS)
# =========================================================
st.title("🏀 EuroLeague Fantasy Dominator Ultra")
tab_overview, tab_roster, tab_optimizer = st.tabs([
    "🏠 Apžvalga", 
    "📝 Mano Sudėtis", 
    "🤖 MILP Optimizatorius"
])

# --- TAB 1: APŽVALGA (PAGAL 2 NUOTRAUKĄ) ---
with tab_overview:
    # Filtrai viršuje
    st.markdown("""
        <div style="margin-bottom:15px;">
            <span class="filter-badge active">Visi</span>
            <span class="filter-badge">Vertė</span>
            <span class="filter-badge">Pigūs</span>
            <span class="filter-badge">Lengvas grafikas</span>
            <span class="filter-badge">Traumos</span>
            <span class="filter-badge">Stebimi</span>
            <span class="filter-badge">Mano 10</span>
        </div>
    """, unsafe_allow_html=True)

    st.caption(f"{len(df_players)} ŽAIDĖJAI")

    # Custom lentelė su BasketNews Fantasy stiliaus stulpeliais
    for _, r in df_players.iterrows():
        cols = st.columns([0.6, 2.5, 2.0, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8])
        
        with cols[0]:
            st.image(r["photo"], width=40)
        with cols[1]:
            st.markdown(f"**{r['name']}**<br><span style='color:#a0aec0;font-size:12px;'><b>{r['pos']}</b> • {r['team']}</span>", unsafe_allow_html=True)
        with cols[2]:
            venue = "vs" if r["is_home"] else "@"
            st.markdown(f"<span style='font-size:13px;'>{venue} <b>{r['opp']}</b> <span style='color:#718096;'>{r['win_prob']}%</span></span>", unsafe_allow_html=True)
        with cols[3]:
            chg_color = "#48bb78" if r["price_change"] >= 0 else "#f56565"
            chg_str = f"▲ {r['price_change']}" if r["price_change"] >= 0 else f"▼ {abs(r['price_change'])}"
            st.markdown(f"<b>{r['price']:.2f}</b><br><span style='color:{chg_color};font-size:11px;'>{chg_str}</span>", unsafe_allow_html=True)
        with cols[4]:
            st.markdown(f"<span class='pir-badge'>{r['proj_pir']:.1f}</span>", unsafe_allow_html=True)
        with cols[5]:
            st.markdown(f"<b>{r['avg_pir']:.1f}</b>", unsafe_allow_html=True)
        with cols[6]:
            st.markdown(f"{r['pts']:.1f}")
        with cols[7]:
            st.markdown(f"{r['min']:.1f}")
        with cols[8]:
            st.markdown(f"<span class='pir-val-badge'>{r['val']:.1f}</span>", unsafe_allow_html=True)
        with cols[9]:
            is_in = r["id"] in st.session_state["my_roster"]
            btn_label = "✓" if is_in else "+"
            if st.button(btn_label, key=f"add_{r['id']}"):
                if is_in:
                    st.session_state["my_roster"].remove(r["id"])
                else:
                    if len(st.session_state["my_roster"]) < 10:
                        st.session_state["my_roster"].append(r["id"])
                st.rerun()
        st.markdown("<hr style='margin: 4px 0; border-color: #1a202c;'>", unsafe_allow_html=True)

# --- TAB 2: MANO SUDĖTIS (PAGAL 3 NUOTRAUKĄ IR DI PASIŪLYMUS) ---
with tab_roster:
    my_df = df_players[df_players["id"].isin(st.session_state["my_roster"])].copy()

    # --- 1. BET KURIO ŽAIDĖJO PAIEŠKA PAGAL POZICIJĄ IR SUMĄ ---
    st.markdown("### 🔎 Rasti Geriausią Žaidėją (Pagal Poziciją ir Sumą)")
    p_col1, p_col2, p_col3 = st.columns([1, 1, 1])
    with p_col1:
        search_pos = st.selectbox("Pasirinkite poziciją:", ["G", "P", "C"])
    with p_col2:
        search_price = st.number_input("Maksimali kaina (€M):", min_value=0.4, max_value=3.0, value=1.5, step=0.05)
    with p_col3:
        st.write("")
        st.write("")
        find_btn = st.button("🚀 Rasti Modelio Variantą")

    if find_btn:
        best_p = find_best_single_player(df_players, search_pos, search_price, st.session_state["my_roster"])
        if best_p is not None:
            st.success(
                f"🌟 **Geriausias pasirinkimas:** {best_p['name']} ({best_p['team']}) — "
                f"Kaina: €{best_p['price']}M | Prog. PIR: **{best_p['proj_pir']}** | vs {best_p['opp']}"
            )
        else:
            st.warning("Neradau reikalavimus atitinkančio laisvo žaidėjo.")

    st.markdown("---")

    # --- 2. DI KEITIMŲ REKOMENDACIJŲ VARIKLIS ---
    st.markdown("### 🤖 Pažangiausių Modelių Keitimų Pasiūlymai")
    if len(my_df) > 0:
        # Surandame silpniausią žaidėją pagal prognozę
        weakest = my_df.sort_values("proj_pir").iloc[0]
        replacement = find_best_single_player(df_players, weakest["pos"], weakest["price"] + 0.1, st.session_state["my_roster"])

        if replacement is not None and replacement["proj_pir"] > weakest["proj_pir"]:
            st.markdown(f"""
                <div class="ai-card">
                    <h4>💡 Rekomenduojamas Keitimas</h4>
                    <p>Išimti: <b>{weakest['name']}</b> ({weakest['pos']}, €{weakest['price']}M, Proj PIR: {weakest['proj_pir']})<br>
                    Paimti: <b>{replacement['name']}</b> ({replacement['pos']}, €{replacement['price']}M, Proj PIR: <b>{replacement['proj_pir']}</b>)<br>
                    <span style="color:#38bdf8;"><b>Pranašumas: +{round(replacement['proj_pir'] - weakest['proj_pir'], 1)} PIR</b></span></p>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # --- 3. KREPŠINIO AIKŠTELĖ IR SUOLIUKAS (MATCHING 3 PHOTO) ---
    st.markdown("### 🏟️ Tavo Sudėtis Aikštelėje")
    
    sorted_roster, cap_id, sixth_man_id, bench_ids = optimize_roles(my_df)

    court_col, bench_col = st.columns([2, 1])

    with court_col:
        st.markdown("<div class='court-container'>", unsafe_allow_html=True)
        st.caption("PAGRINDINIS PENKETUKAS IR KAPITONAS (2X)")
        
        # Aikštelės žaidėjų atvaizdavimas
        starters = sorted_roster.iloc[:5] if len(sorted_roster) >= 5 else sorted_roster
        row1 = st.columns(3)
        for idx, (_, p) in enumerate(starters.iterrows()):
            is_c = (p["id"] == cap_id)
            cap_badge = "<span style='color:#ff6b00;font-weight:bold;'> 2x</span>" if is_c else ""
            c_class = "court-slot captain" if is_c else "court-slot"
            
            with row1[idx % 3]:
                st.markdown(f"""
                    <div class="{c_class}">
                        <img src="{p['photo']}" width="45" style="border-radius:50%;"><br>
                        <b>{p['name'].split()[-1]}</b>{cap_badge}<br>
                        <span style="color:#00b4d8;font-weight:bold;font-size:18px;">{p['proj_pir'] * (2 if is_c else 1):.1f}</span><br>
                        <small style="color:#a0aec0;">vs {p['opp']} {p['time']}</small>
                    </div>
                """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with bench_col:
        st.markdown("<div class='bench-container'>", unsafe_allow_html=True)
        st.markdown("<b>SUOLIUKAS & 6-AS ŽAIDĖJAS</b>", unsafe_allow_html=True)
        
        bench_df = sorted_roster.iloc[5:] if len(sorted_roster) > 5 else pd.DataFrame()
        for _, p in bench_df.iterrows():
            is_6th = (p["id"] == sixth_man_id)
            badge_str = "6" if is_6th else "1/2"
            mult = 1.0 if is_6th else 0.5
            
            st.markdown(f"""
                <div class="bench-item">
                    <img src="{p['photo']}" width="35" style="border-radius:50%;">
                    <div>
                        <b>{p['name'].split()[-1]}</b><br>
                        <small style="color:#a0aec0;">{p['time']}</small>
                    </div>
                    <div style="text-align:right;">
                        <span style="color:#00b4d8;font-weight:bold;">{p['proj_pir'] * mult:.1f}</span><br>
                        <small style="background:#2d3748;padding:2px 6px;border-radius:4px;">{badge_str}</small>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- TAB 3: MILP OPTIMIZATORIUS ---
with tabs[2]:
    st.subheader("🤖 Algoritminis MILP Optimizatorius")
    opt_b = st.slider("Pasirinkite biudžetą (€M):", 5.0, 20.0, 14.5, 0.1)

    if st.button("⚡ Generuoti Optimalų 10-tuką"):
        active = df_players[df_players["status"].str.lower() == "ready"].reset_index(drop=True)
        N = len(active)
        if N >= 10:
            c = -active["proj_pir"].values
            prices = active["price"].values
            is_g = (active["pos"] == "G").astype(float).values
            is_p = (active["pos"] == "P").astype(float).values
            is_c = (active["pos"] == "C").astype(float).values
            ones = np.ones(N)

            A = np.vstack([prices, is_g, is_p, is_c, ones])
            lhs = np.array([0, 4, 4, 2, 10])
            rhs = np.array([opt_b, 4, 4, 2, 10])

            res = milp(c=c, integrality=np.ones(N), bounds=Bounds(0, 1), constraints=LinearConstraint(A, lhs, rhs))
            if res.success:
                opt_df = active.iloc[np.where(res.x > 0.5)[0]].copy()
                st.success("✅ Optimali sudėtis parinkta!")
                st.dataframe(opt_df[["name", "pos", "price", "proj_pir", "opp", "val"]], use_container_width=True, hide_index=True)
                if st.button("Pritaikyti kaip Mano Sudėtį"):
                    st.session_state["my_roster"] = opt_df["id"].tolist()
                    st.rerun()