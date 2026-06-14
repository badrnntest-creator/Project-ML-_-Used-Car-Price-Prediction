import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Prediction Prix Voiture",
    page_icon="🚗",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        padding: 0.5rem 0;
    }
    .subtitle {
        font-size: 0.95rem;
        color: #555;
        text-align: center;
        margin-bottom: 1rem;
    }
    .feature-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.3rem;
    }
    .impact-bar {
        width: 6px;
        border-radius: 3px;
        align-self: stretch;
    }
    .impact-high { background: #2ecc71; }
    .impact-medium { background: #f39c12; }
    .impact-low { background: #95a5a6; }
    .impact-pct {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 0.1rem 0.45rem;
        border-radius: 10px;
        white-space: nowrap;
    }
    .pct-high { background: #d4edda; color: #155724; }
    .pct-medium { background: #fff3cd; color: #856404; }
    .pct-low { background: #e2e3e5; color: #495057; }
    .result-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 0.5rem;
    }
    .result-price {
        font-size: 2.6rem;
        font-weight: 800;
        color: #f0c040;
    }
    .result-range {
        font-size: 0.95rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    .impact-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.9rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-left: 3px solid #1a1a2e;
        padding: 0.5rem 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.35rem;
        font-size: 0.85rem;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 0.8rem;
        border-top: 1px solid #eee;
    }
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column"] > div[data-testid="stVerticalBlock"] {
        gap: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Donnees marque -> modeles ────────────────────────────────────
BRAND_MODELS = {
    'volkswagen'   : ['golf','polo','passat','transporter','touran','lupo','caddy','sharan','tiguan','bora'],
    'bmw'          : ['3er','5er','1er','x_reihe','7er','z_reihe','andere','m_reihe','6er','i3'],
    'mercedes_benz': ['c_klasse','e_klasse','a_klasse','andere','clk','slk','m_klasse','s_klasse','b_klasse','vito'],
    'opel'         : ['corsa','astra','vectra','zafira','omega','andere','meriva','tigra','insignia','signum'],
    'audi'         : ['a4','a3','a6','80','andere','tt','a5','a1','a8','q5'],
    'ford'         : ['focus','fiesta','mondeo','ka','andere','galaxy','escort','transit','c_max','s_max'],
    'renault'      : ['twingo','clio','megane','scenic','laguna','andere','kangoo','espace','modus','r19'],
    'peugeot'      : ['207','206','307','308','partner','andere','407','106','508','3008'],
    'fiat'         : ['punto','500','panda','bravo','andere','doblo','stilo','ducato','grande_punto','tipo'],
    'seat'         : ['ibiza','leon','andere','toledo','cordoba','alhambra','arosa','exeo','altea','mii'],
    'skoda'        : ['fabia','octavia','andere','superb','roomster','yeti','rapid','citigo','kodiaq','scala'],
    'mazda'        : ['andere','323','6','mx_reihe','3','cx_reihe','2','5','121','rx_reihe'],
    'nissan'       : ['andere','micra','qashqai','almera','primera','x_trail','note','juke','navara','350z'],
    'toyota'       : ['andere','corolla','aygo','yaris','avensis','rav','auris','verso','hilux','prius'],
    'honda'        : ['andere','civic','accord','jazz','crv','hrv','fr_v','legend','stream','concerto'],
    'hyundai'      : ['andere','i10','i20','i30','ix35','tucson','santa','getz','coupe','accent'],
    'kia'          : ['andere','rio','ceed','sportage','picanto','venga','sorento','carnival','carens','pro_ceed'],
    'volvo'        : ['andere','v70','s40','v40','xc_reihe','c30','s60','s80','v50','850'],
    'mini'         : ['mini','andere','cooper','one','clubman','countryman','cabrio','paceman','roadster','coupe'],
    'porsche'      : ['andere','911','boxster','cayenne','macan','panamera','cayman','918','356','928'],
}

BRAND_FR = {
    'volkswagen':'Volkswagen','bmw':'BMW','mercedes_benz':'Mercedes-Benz',
    'opel':'Opel','audi':'Audi','ford':'Ford','renault':'Renault',
    'peugeot':'Peugeot','fiat':'Fiat','seat':'SEAT','skoda':'Skoda',
    'mazda':'Mazda','nissan':'Nissan','toyota':'Toyota','honda':'Honda',
    'hyundai':'Hyundai','kia':'Kia','volvo':'Volvo','mini':'MINI','porsche':'Porsche'
}

def model_fr(m):
    mapping = {
        'andere':'Autre','x_reihe':'Serie X','z_reihe':'Serie Z',
        'm_reihe':'Serie M','c_klasse':'Classe C','e_klasse':'Classe E',
        'a_klasse':'Classe A','m_klasse':'Classe M','s_klasse':'Classe S',
        'b_klasse':'Classe B','grande_punto':'Grande Punto',
        'c_max':'C-Max','s_max':'S-Max','mx_reihe':'MX','cx_reihe':'CX',
        'xc_reihe':'XC','fr_v':'FR-V','pro_ceed':'Pro Ceed',
        'x_trail':'X-Trail'
    }
    return mapping.get(m, m.replace('_',' ').upper())

MODEL_VEHICLE_TYPE = {
    'volkswagen'   : {'golf':'limousine','polo':'kleinwagen','passat':'kombi','transporter':'bus',
                      'touran':'bus','tiguan':'suv','lupo':'kleinwagen','caddy':'bus',
                      'sharan':'bus','bora':'limousine'},
    'bmw'          : {'3er':'limousine','5er':'limousine','1er':'limousine','x_reihe':'suv',
                      '7er':'limousine','z_reihe':'cabrio','m_reihe':'coupe','6er':'coupe',
                      'i3':'kleinwagen','andere':'andere'},
    'mercedes_benz': {'c_klasse':'limousine','e_klasse':'limousine','a_klasse':'kleinwagen',
                      'clk':'coupe','slk':'cabrio','m_klasse':'suv','s_klasse':'limousine',
                      'b_klasse':'kleinwagen','vito':'bus','andere':'andere'},
    'opel'         : {'corsa':'kleinwagen','astra':'limousine','vectra':'limousine','zafira':'bus',
                      'omega':'limousine','meriva':'kleinwagen','tigra':'cabrio',
                      'insignia':'limousine','signum':'limousine','andere':'andere'},
    'audi'         : {'a4':'limousine','a3':'kleinwagen','a6':'limousine','80':'limousine',
                      'tt':'coupe','a5':'coupe','a1':'kleinwagen','a8':'limousine',
                      'q5':'suv','andere':'andere'},
    'ford'         : {'focus':'limousine','fiesta':'kleinwagen','mondeo':'limousine',
                      'ka':'kleinwagen','galaxy':'bus','escort':'limousine',
                      'transit':'bus','c_max':'bus','s_max':'bus','andere':'andere'},
    'renault'      : {'twingo':'kleinwagen','clio':'kleinwagen','megane':'limousine',
                      'scenic':'bus','laguna':'limousine','kangoo':'bus',
                      'espace':'bus','modus':'kleinwagen','r19':'limousine','andere':'andere'},
    'peugeot'      : {'207':'kleinwagen','206':'kleinwagen','307':'limousine',
                      '308':'limousine','partner':'bus','407':'limousine',
                      '106':'kleinwagen','508':'limousine','3008':'suv','andere':'andere'},
    'fiat'         : {'punto':'kleinwagen','500':'kleinwagen','panda':'kleinwagen',
                      'bravo':'limousine','doblo':'bus','stilo':'limousine',
                      'ducato':'bus','grande_punto':'kleinwagen','tipo':'limousine','andere':'andere'},
    'seat'         : {'ibiza':'kleinwagen','leon':'limousine','toledo':'limousine',
                      'cordoba':'limousine','alhambra':'bus','arosa':'kleinwagen',
                      'exeo':'limousine','altea':'bus','mii':'kleinwagen','andere':'andere'},
    'skoda'        : {'fabia':'kleinwagen','octavia':'limousine','superb':'limousine',
                      'roomster':'bus','yeti':'suv','rapid':'limousine',
                      'citigo':'kleinwagen','kodiaq':'suv','scala':'limousine','andere':'andere'},
    'mazda'        : {'323':'limousine','6':'limousine','mx_reihe':'cabrio','3':'limousine',
                      'cx_reihe':'suv','2':'kleinwagen','5':'bus','121':'kleinwagen',
                      'rx_reihe':'coupe','andere':'andere'},
    'nissan'       : {'micra':'kleinwagen','qashqai':'suv','almera':'limousine',
                      'primera':'limousine','x_trail':'suv','note':'kleinwagen',
                      'juke':'suv','navara':'andere','350z':'coupe','andere':'andere'},
    'toyota'       : {'corolla':'limousine','aygo':'kleinwagen','yaris':'kleinwagen',
                      'avensis':'limousine','rav':'suv','auris':'limousine',
                      'verso':'bus','hilux':'andere','prius':'limousine','andere':'andere'},
    'honda'        : {'civic':'limousine','accord':'limousine','jazz':'kleinwagen',
                      'crv':'suv','hrv':'suv','fr_v':'bus','legend':'limousine',
                      'stream':'bus','andere':'andere'},
    'hyundai'      : {'i10':'kleinwagen','i20':'kleinwagen','i30':'limousine',
                      'ix35':'suv','tucson':'suv','santa':'suv','getz':'kleinwagen',
                      'coupe':'coupe','accent':'limousine','andere':'andere'},
    'kia'          : {'rio':'kleinwagen','ceed':'limousine','sportage':'suv',
                      'picanto':'kleinwagen','venga':'kleinwagen','sorento':'suv',
                      'carnival':'bus','carens':'bus','pro_ceed':'limousine','andere':'andere'},
    'volvo'        : {'v70':'kombi','s40':'limousine','v40':'kombi','xc_reihe':'suv',
                      'c30':'limousine','s60':'limousine','s80':'limousine',
                      'v50':'kombi','850':'limousine','andere':'andere'},
    'mini'         : {'mini':'kleinwagen','cooper':'kleinwagen','one':'kleinwagen',
                      'clubman':'kombi','countryman':'suv','cabrio':'cabrio',
                      'paceman':'suv','roadster':'cabrio','coupe':'coupe','andere':'andere'},
    'porsche'      : {'911':'coupe','boxster':'cabrio','cayenne':'suv','macan':'suv',
                      'panamera':'limousine','cayman':'coupe','andere':'andere'},
}

VEHICLE_TYPE_FR = {
    'limousine' :'Berline','kleinwagen':'Citadine','kombi':'Break',
    'bus':'Monospace / Van','cabrio':'Cabriolet','coupe':'Coupe',
    'suv':'SUV / 4x4','andere':'Autre'
}

GEARBOX_OPTIONS = {'Manuelle':'manuell','Automatique':'automatik'}
FUEL_OPTIONS = {
    'Essence':'benzin','Diesel':'diesel','Hybride':'hybrid',
    'Electrique':'elektro','GPL':'lpg','GNV':'cng','Autre':'andere'
}
DAMAGE_OPTIONS = {
    'Non — Vehicule en bon etat'  : 'nein',
    'Oui — Vehicule avec dommages': 'ja'
}

# ── Chargement pipeline ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.join(os.path.dirname(__file__), '..', 'models')
    pipeline = joblib.load(os.path.join(base, 'pipeline_final.pkl'))
    encoders = joblib.load(os.path.join(base, 'encoders.pkl'))
    return pipeline, encoders

try:
    pipeline, encoders = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Erreur de chargement : {e}")

def encode_feature(col, val, encoders):
    le = encoders.get(col)
    if le is None:
        return 0
    classes = list(le.classes_)
    return int(le.transform([val])[0]) if val in classes else 0

def impact_bar(pct):
    if pct >= 15:
        bar_cls, pct_cls = "impact-high", "pct-high"
    elif pct >= 7:
        bar_cls, pct_cls = "impact-medium", "pct-medium"
    else:
        bar_cls, pct_cls = "impact-low", "pct-low"
    return bar_cls, pct_cls

# ── En-tete ──────────────────────────────────────────────────────
st.markdown('<div class="main-title">Prediction du Prix de Vente d\'une Voiture d\'Occasion</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle">EHTP - MSDE Edition 7 | Module 5 : Machine Learning</div>',
            unsafe_allow_html=True)

col_form, col_result = st.columns([1.4, 1])

with col_form:
    # ── Ligne 1 : Marque / Modele / Annee ──────────────────────
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        bar_cls, pct_cls = impact_bar(6)
        st.markdown(f'<div class="feature-row"><div class="impact-bar {bar_cls}"></div>'
                     f'<span><b>Marque</b> <span class="impact-pct {pct_cls}">6%</span></span></div>',
                     unsafe_allow_html=True)
        brand = st.selectbox(
            " ", options=sorted(BRAND_MODELS.keys()),
            format_func=lambda x: BRAND_FR.get(x, x), label_visibility="collapsed"
        )

    with c2:
        bar_cls, pct_cls = impact_bar(3)
        st.markdown(f'<div class="feature-row"><div class="impact-bar {bar_cls}"></div>'
                     f'<span><b>Modele</b> <span class="impact-pct {pct_cls}">3%</span></span></div>',
                     unsafe_allow_html=True)
        modeles_disponibles = BRAND_MODELS.get(brand, ['andere'])
        model_car = st.selectbox(
            " ", options=modeles_disponibles,
            format_func=lambda x: model_fr(x), label_visibility="collapsed"
        )

    with c3:
        bar_cls, pct_cls = impact_bar(34)
        st.markdown(f'<div class="feature-row"><div class="impact-bar {bar_cls}"></div>'
                     f'<span><b>Annee d\'immatriculation</b> <span class="impact-pct {pct_cls}">34%</span></span></div>',
                     unsafe_allow_html=True)
        year = st.slider(
            " ", min_value=1990, max_value=2012, value=2008,
            label_visibility="collapsed"
        )
        car_age = 2016 - year

    auto_vehicle_type = MODEL_VEHICLE_TYPE.get(brand, {}).get(model_car, 'andere')

    # ── Ligne 2 : Type (auto) / Etat / Puissance ───────────────
    c4, c5, c6 = st.columns([1, 1, 1])

    with c4:
        bar_cls, pct_cls = impact_bar(7)
        st.markdown(f'<div class="feature-row"><div class="impact-bar {bar_cls}"></div>'
                     f'<span><b>Type de carrosserie</b> <span class="impact-pct {pct_cls}">7%</span></span></div>',
                     unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card" style="margin-top:0.3rem;">'
                     f'Detecte : <b>{VEHICLE_TYPE_FR.get(auto_vehicle_type, auto_vehicle_type)}</b>'
                     f'</div>', unsafe_allow_html=True)

    with c5:
        bar_cls, pct_cls = impact_bar(18)
        st.markdown(f'<div class="feature-row"><div class="impact-bar {bar_cls}"></div>'
                     f'<span><b>Etat du vehicule</b> <span class="impact-pct {pct_cls}">18%</span></span></div>',
                     unsafe_allow_html=True)
        damage_fr = st.radio(
            " ", options=list(DAMAGE_OPTIONS.keys()),
            horizontal=False, label_visibility="collapsed"
        )
        not_repaired = DAMAGE_OPTIONS[damage_fr]

    with c6:
        bar_cls, pct_cls = impact_bar(13)
        st.markdown(f'<div class="feature-row"><div class="impact-bar {bar_cls}"></div>'
                     f'<span><b>Puissance (PS)</b> <span class="impact-pct {pct_cls}">13%</span></span></div>',
                     unsafe_allow_html=True)
        power_ps = st.slider(
            " ", min_value=10, max_value=500, value=120, step=5,
            label_visibility="collapsed"
        )

    # ── Ligne 3 : Carburant / Kilometrage / Boite ──────────────
    c7, c8, c9 = st.columns([1, 1, 1])

    with c7:
        bar_cls, pct_cls = impact_bar(8)
        st.markdown(f'<div class="feature-row"><div class="impact-bar {bar_cls}"></div>'
                     f'<span><b>Carburant</b> <span class="impact-pct {pct_cls}">8%</span></span></div>',
                     unsafe_allow_html=True)
        fuel_fr = st.selectbox(
            " ", options=list(FUEL_OPTIONS.keys()), label_visibility="collapsed"
        )
        fuel_type = FUEL_OPTIONS[fuel_fr]

    with c8:
        bar_cls, pct_cls = impact_bar(6)
        st.markdown(f'<div class="feature-row"><div class="impact-bar {bar_cls}"></div>'
                     f'<span><b>Kilometrage</b> <span class="impact-pct {pct_cls}">6%</span></span></div>',
                     unsafe_allow_html=True)
        kilometer = st.slider(
            " ", min_value=0, max_value=200000, value=80000, step=5000,
            label_visibility="collapsed"
        )

    with c9:
        bar_cls, pct_cls = impact_bar(4)
        st.markdown(f'<div class="feature-row"><div class="impact-bar {bar_cls}"></div>'
                     f'<span><b>Boite de vitesses</b> <span class="impact-pct {pct_cls}">4%</span></span></div>',
                     unsafe_allow_html=True)
        gearbox_fr = st.selectbox(
            " ", options=list(GEARBOX_OPTIONS.keys()), label_visibility="collapsed"
        )
        gearbox = GEARBOX_OPTIONS[gearbox_fr]

    st.markdown("")
    predict_btn = st.button(
        "Predire le Prix", type="primary", use_container_width=True
    )

with col_result:
    st.markdown('<div class="subtitle" style="font-weight:700; font-size:1.05rem; '
                 'margin-bottom:0.5rem;">Resultat de la Prediction</div>',
                 unsafe_allow_html=True)

    if predict_btn and model_loaded:
        input_data = pd.DataFrame([{
            'car_age'                   : car_age,
            'powerPS'                   : power_ps,
            'kilometer'                 : kilometer,
            'vehicleType_encoded'       : encode_feature('vehicleType',       auto_vehicle_type, encoders),
            'gearbox_encoded'           : encode_feature('gearbox',           gearbox,           encoders),
            'fuelType_encoded'          : encode_feature('fuelType',          fuel_type,         encoders),
            'notRepairedDamage_encoded' : encode_feature('notRepairedDamage', not_repaired,      encoders),
            'brand_encoded'             : encode_feature('brand',             brand,             encoders),
            'model_encoded'             : encode_feature('model',             model_car,         encoders),
        }])

        try:
            log_pred   = pipeline.predict(input_data)[0]
            price_pred = np.expm1(log_pred)
            price_min  = price_pred * 0.85
            price_max  = price_pred * 1.15

            st.markdown(f"""
            <div class="result-box">
                <div style="font-size:0.95rem; margin-bottom:0.4rem;">Prix estime</div>
                <div class="result-price">{price_pred:,.0f} EUR</div>
                <div class="result-range">
                    Fourchette : {price_min:,.0f} EUR — {price_max:,.0f} EUR
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Impact de notRepairedDamage ──
            input_alt = input_data.copy()
            alt_damage = 'ja' if not_repaired == 'nein' else 'nein'
            input_alt['notRepairedDamage_encoded'] = encode_feature(
                'notRepairedDamage', alt_damage, encoders)
            price_alt = np.expm1(pipeline.predict(input_alt)[0])

            diff = price_pred - price_alt
            diff_pct = (diff / price_alt) * 100 if price_alt != 0 else 0

            st.markdown("---")
            st.markdown("**Impact de l'etat du vehicule (18%) :**")
            if not_repaired == 'nein':
                st.markdown(f"""
                <div class="impact-card">
                    <span>Bon etat (actuel)</span>
                    <strong style="color:#2ecc71;">{price_pred:,.0f} EUR</strong>
                </div>
                <div class="impact-card">
                    <span>Avec dommages</span>
                    <strong style="color:#e74c3c;">{price_alt:,.0f} EUR</strong>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="impact-card">
                    <span>Avec dommages (actuel)</span>
                    <strong style="color:#e74c3c;">{price_pred:,.0f} EUR</strong>
                </div>
                <div class="impact-card">
                    <span>Sans dommages</span>
                    <strong style="color:#2ecc71;">{price_alt:,.0f} EUR</strong>
                </div>
                """, unsafe_allow_html=True)
            st.caption(f"Ecart : {abs(diff):,.0f} EUR ({abs(diff_pct):.0f}%)")

            st.markdown("---")
            st.markdown("**Recapitulatif :**")

            recap = {
                "Marque"      : BRAND_FR.get(brand, brand),
                "Modele"      : model_fr(model_car),
                "Type"        : VEHICLE_TYPE_FR.get(auto_vehicle_type, auto_vehicle_type),
                "Annee"       : f"{year}",
                "Etat"        : damage_fr,
                "Puissance"   : f"{power_ps} PS",
                "Carburant"   : fuel_fr,
                "Kilometrage" : f"{kilometer:,} km",
                "Boite"       : gearbox_fr,
            }

            for key, val in recap.items():
                st.markdown(
                    f'<div class="metric-card"><strong>{key}</strong> : {val}</div>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"Erreur prediction : {e}")

    elif not predict_btn:
        st.info("Renseignez les caracteristiques et cliquez sur 'Predire le Prix'.")

st.markdown("""
<div class="footer">
    Projet ML - EHTP MSDE Edition 7 | Dataset : eBay Kleinanzeigen | Modele : XGBoost
</div>
""", unsafe_allow_html=True)
