"""Infidelity Predictor - Streamlit Web Application.

パートナーの情報を入力して、浮気リスクを予測するWebアプリ。
学術研究に基づく予測モデル（RandomForest + SHAP）を使用。
"""
import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models_ml", "model.pkl")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "infidelity_predictor.duckdb")


# ────────────────────────────────────────────────────────────
# Page Config
# ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="浮気リスク予測",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ────────────────────────────────────────────────────────────
# Custom CSS (Light Theme)
# ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Light base theme adjustments */
    .stApp {
        background-color: #ffffff;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #6c757d;
        font-size: 1.1rem;
    }
    .step-header {
        color: #212529;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e9ecef;
    }
    .result-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .disclaimer-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        color: #856404;
        font-size: 0.85rem;
    }
    .source-tag {
        display: inline-block;
        background: #e9ecef;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        color: #495057;
        margin: 2px;
    }
    div[data-testid="stSlider"] > div {
        padding-top: 0;
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────
# Load Model
# ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
    # Recreate SHAP explainer from model (avoids numba pickle incompatibility)
    artifacts["explainer"] = shap.TreeExplainer(artifacts["model"])
    return artifacts


@st.cache_resource
def load_personality_reference():
    """Load personality norms from DuckDB."""
    try:
        import duckdb
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.sql("SELECT * FROM main_dimensions.dim_personality_reference").fetchdf()
        con.close()
        return df
    except Exception:
        return None


@st.cache_resource
def load_japan_context():
    """Load Japanese divorce/infidelity statistics."""
    try:
        import duckdb
        con = duckdb.connect(DB_PATH, read_only=True)
        df = con.sql("SELECT * FROM main_dimensions.dim_japan_context").fetchdf()
        con.close()
        return df
    except Exception:
        return None


# ────────────────────────────────────────────────────────────
# Helper Functions
# ────────────────────────────────────────────────────────────

FEATURE_LABELS_JP = {
    "age": "年齢",
    "education_years": "教育年数",
    "religiousness": "信仰心",
    "occupation": "職業レベル",
    "honesty_humility": "誠実-謙虚さ (H-H)",
    "emotionality": "情緒性",
    "extraversion": "外向性",
    "agreeableness": "協調性",
    "conscientiousness": "誠実性",
    "openness": "開放性",
    "years_in_relationship": "交際/結婚年数",
    "has_children": "子供の有無",
    "satisfaction_rating": "関係満足度",
    "love_rating": "愛情度",
    "desire_rating": "欲求度",
}


def create_gauge_chart(probability):
    """Create animated gauge chart for probability."""
    # Color based on risk level
    if probability < 0.15:
        color = "#2ecc71"
        label = "低リスク"
    elif probability < 0.30:
        color = "#f39c12"
        label = "やや注意"
    elif probability < 0.50:
        color = "#e67e22"
        label = "注意"
    else:
        color = "#e74c3c"
        label = "高リスク"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 48, "color": "#212529"}},
        title={"text": f"浮気リスク: {label}", "font": {"size": 20, "color": "#495057"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#dee2e6",
                     "tickfont": {"color": "#6c757d"}},
            "bar": {"color": color, "thickness": 0.75},
            "bgcolor": "#f8f9fa",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 15], "color": "rgba(46, 204, 113, 0.15)"},
                {"range": [15, 30], "color": "rgba(243, 156, 18, 0.15)"},
                {"range": [30, 50], "color": "rgba(230, 126, 34, 0.15)"},
                {"range": [50, 100], "color": "rgba(231, 76, 60, 0.15)"},
            ],
            "threshold": {
                "line": {"color": "#495057", "width": 3},
                "thickness": 0.8,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#212529"},
        height=300,
        margin=dict(t=80, b=30, l=40, r=40),
    )
    return fig


def create_shap_waterfall(shap_values, feature_names, base_value):
    """Create SHAP waterfall chart showing feature contributions."""
    # Ensure feature_names is a flat list
    if isinstance(feature_names, (list, np.ndarray)):
        feature_names = list(feature_names) if isinstance(feature_names, np.ndarray) else feature_names
        # Flatten if nested
        if feature_names and isinstance(feature_names[0], (list, tuple)):
            feature_names = [item for sublist in feature_names for item in sublist]
    
    # Ensure shap_values is 1D
    shap_values = np.array(shap_values)
    if shap_values.ndim == 2:
        # (n_features, n_classes) -> take positive class (last column)
        shap_values = shap_values[:, -1]
    elif shap_values.ndim > 2:
        shap_values = shap_values.flatten()
    shap_values = shap_values.ravel()

    # Sort by absolute SHAP value
    abs_vals = np.abs(shap_values)
    sorted_idx = np.argsort(abs_vals)[::-1][:10]  # Top 10
    
    # Extract features and values using integer indexing
    features = []
    values = []
    for i in range(len(sorted_idx)):
        idx = sorted_idx[i]
        if hasattr(idx, 'item'):
            idx = idx.item()  # Convert numpy scalar to Python int
        else:
            idx = int(idx)
        feat_name = feature_names[idx]
        features.append(FEATURE_LABELS_JP.get(feat_name, feat_name))
        val = shap_values[idx]
        if hasattr(val, 'item'):
            val = val.item()
        values.append(float(val))
    
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont={"color": "#212529", "size": 12},
    ))
    fig.update_layout(
        title={"text": "各要因の寄与度 (SHAP値)", "font": {"color": "#212529", "size": 16}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#212529"},
        xaxis={"title": "浮気リスクへの寄与 →", "gridcolor": "#e9ecef", "zerolinecolor": "#dee2e6"},
        yaxis={"autorange": "reversed", "gridcolor": "rgba(0,0,0,0)"},
        height=400,
        margin=dict(l=160, r=60, t=60, b=40),
    )
    return fig


def create_hexaco_radar(personality_scores, reference_df=None):
    """Create HEXACO radar chart."""
    traits = ["honesty_humility", "emotionality", "extraversion",
              "agreeableness", "conscientiousness", "openness"]
    trait_labels = ["誠実-謙虚さ", "情緒性", "外向性", "協調性", "誠実性", "開放性"]

    values = [personality_scores.get(t, 3.0) for t in traits]
    values_closed = values + [values[0]]
    labels_closed = trait_labels + [trait_labels[0]]

    fig = go.Figure()

    # Population average line
    avg_values = [3.4, 3.2, 3.5, 3.1, 3.6, 3.4]
    avg_closed = avg_values + [avg_values[0]]
    fig.add_trace(go.Scatterpolar(
        r=avg_closed,
        theta=labels_closed,
        name="平均",
        line={"color": "#6c757d", "dash": "dash", "width": 1},
        fill="none",
    ))

    # User input
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        name="パートナー",
        fill="toself",
        fillcolor="rgba(102, 126, 234, 0.2)",
        line={"color": "#667eea", "width": 2},
    ))

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True, "range": [1, 5],
                "gridcolor": "#e9ecef", "linecolor": "#dee2e6",
                "tickfont": {"color": "#6c757d"},
            },
            "angularaxis": {
                "gridcolor": "#e9ecef", "linecolor": "#dee2e6",
                "tickfont": {"color": "#212529", "size": 13},
            },
            "bgcolor": "rgba(0,0,0,0)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#212529"},
        showlegend=True,
        legend={"font": {"color": "#212529"}},
        height=400,
        margin=dict(t=40, b=40, l=80, r=80),
    )
    return fig


# ────────────────────────────────────────────────────────────
# Main App
# ────────────────────────────────────────────────────────────
def main():
    # Header
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image("images/image.png")
    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ <strong>注意:</strong> このツールは学術研究データに基づく統計的予測であり、結果は参考情報としてのみご利用ください。パートナーへの不当な疑いや不信感を助長する目的での使用は推奨しません。
    </div>
    """, unsafe_allow_html=True)

    # Load model
    try:
        artifacts = load_model()
        model = artifacts["model"]
        explainer = artifacts["explainer"]
        imputer = artifacts["imputer"]
        feature_names = artifacts["feature_names"]
    except FileNotFoundError:
        st.error("モデルが見つかりません。`python scripts/train_model.py` を実行してください。")
        return

    ref_df = load_personality_reference()
    japan_ctx = load_japan_context()

    # ──── Step 1: Basic Info ────
    st.markdown('<div class="step-header">Step 1: パートナーの基本情報</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("年齢", 18, 80, 30, key="age")
    with col2:
        education = st.selectbox(
            "最終学歴",
            ["高校卒", "大学卒", "大学院卒"],
            index=1,
            key="edu",
        )
        edu_map = {"高校卒": 12, "大学卒": 16, "大学院卒": 18}
        education_years = edu_map[education]
    with col3:
        occupation = st.slider(
            "職業レベル (1〜7)",
            1, 7, 4, key="occ",
            help="1=学生・無職、2=アルバイト・パート、3=一般事務・販売、"
            "4=技術職・専門職、5=管理職・経営幹部、6=士業・医師など高度専門職、7=経営者・役員",
        )

    col4, col5 = st.columns(2)
    with col4:
        religiousness = st.slider(
            "信仰心 (1〜5)",
            1, 5, 2, key="relig",
            help="1=宗教に全く関心がない、2=初詣やお墓参り程度、3=行事ごとにお参りする、"
            "4=定期的に礼拝・お祈りをする、5=日常生活の中心が信仰",
        )
    with col5:
        has_children = st.checkbox("子供がいる", key="children")

    # ──── Step 2: Personality (HEXACO) ────
    st.markdown('<div class="step-header">Step 2: パートナーの性格 (HEXACO 6因子)</div>', unsafe_allow_html=True)
    st.caption("各項目について、パートナーに最も当てはまる度合いを選んでください (1=全く当てはまらない, 5=非常に当てはまる)")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        honesty = st.slider(
            "🤝 誠実-謙虚さ (Honesty-Humility)",
            1.0, 5.0, 3.4, 0.1,
            help="高い人の例: 損をしても嘘をつかない、自慢話をしない、お世辞で人を操ろうとしない。  \n"
            "低い人の例: 自分の利益のために嘘をつく、ルールを都合よく解釈する、特別扱いされたがる。",
            key="hh",
        )
        emotionality = st.slider(
            "💭 情緒性 (Emotionality)",
            1.0, 5.0, 3.2, 0.1,
            help="高い人の例: 些細なことで不安になる、映画で泣きやすい、危険を強く恐れる。  \n"
            "低い人の例: ストレスに動じない、他人の感情に鈍感、肝が据わっている。",
            key="emo",
        )
        extraversion = st.slider(
            "🎉 外向性 (Extraversion)",
            1.0, 5.0, 3.5, 0.1,
            help="高い人の例: パーティーの中心にいたがる、初対面でもすぐ打ち解ける、注目されるのが好き。  \n"
            "低い人の例: 一人の時間を好む、大人数の場が苦手、控えめで静か。",
            key="ext",
        )

    with p_col2:
        agreeableness = st.slider(
            "🕊️ 協調性 (Agreeableness)",
            1.0, 5.0, 3.1, 0.1,
            help="高い人の例: 批判されても怒らない、他人のミスを許せる、争いを避けて譲歩する。  \n"
            "低い人の例: 気に入らないとすぐ反論する、他人のミスを厳しく指摘する、妥協を嫌う。",
            key="agr",
        )
        conscientiousness = st.slider(
            "📋 誠実性 (Conscientiousness)",
            1.0, 5.0, 3.6, 0.1,
            help="高い人の例: 計画通りに物事を進める、デスクが整理されている、細部までこだわる。  \n"
            "低い人の例: 締め切りをよく忘れる、衝動的に行動する、片付けが苦手。",
            key="con",
        )
        openness = st.slider(
            "🎨 開放性 (Openness)",
            1.0, 5.0, 3.4, 0.1,
            help="高い人の例: 未知の文化や芸術に興味がある、哲学的な議論を楽しむ、型にはまらない発想をする。  \n"
            "低い人の例: 慣れたやり方を好む、抽象的な話に興味がない、実用性を重視する。",
            key="opn",
        )

    # ──── Step 3: Relationship ────
    st.markdown('<div class="step-header">Step 3: 関係性の情報</div>', unsafe_allow_html=True)

    r_col1, r_col2 = st.columns(2)
    with r_col1:
        years_together = st.slider(
            "交際/結婚期間 (年)",
            0.0, 40.0, 3.0, 0.5,
            help="交際開始または結婚してからの年数。同棲期間も含めてください。",
            key="years",
        )
        satisfaction = st.slider(
            "関係への満足度 (1=とても不満〜5=とても満足)",
            1.0, 5.0, 4.0, 0.1,
            help="パートナーがあなたとの関係にどの程度満足していると思うか。  \n"
            "1=強い不満がある、3=普通、5=非常に満足。",
            key="sat",
        )
    with r_col2:
        love = st.slider(
            "愛情の深さ (1=低い〜7=とても高い)",
            1.0, 7.0, 5.5, 0.1,
            help="パートナーがあなたに対してどの程度愛情を感じていると思うか。  \n"
            "1=ほとんど感じていない、4=普通、7=深い愛情を感じている。",
            key="love",
        )
        desire = st.slider(
            "欲求の強さ (1=低い〜7=とても高い)",
            1.0, 7.0, 4.8, 0.1,
            help="パートナーがあなたに対してどの程度性的な欲求や魅力を感じていると思うか。  \n"
            "1=ほとんど感じていない、4=普通、7=非常に強く感じている。",
            key="desire",
        )

    # ──── Predict Button ────
    st.markdown("---")

    if st.button("🔮 浮気リスクを分析する", width="stretch", type="primary"):
        with st.spinner("分析中..."):
            # Prepare input
            input_data = {
                "age": age,
                "education_years": education_years,
                "religiousness": religiousness,
                "occupation": occupation,
                "honesty_humility": honesty,
                "emotionality": emotionality,
                "extraversion": extraversion,
                "agreeableness": agreeableness,
                "conscientiousness": conscientiousness,
                "openness": openness,
                "years_in_relationship": years_together,
                "has_children": float(has_children),
                "satisfaction_rating": satisfaction,
                "love_rating": love,
                "desire_rating": desire,
            }

            # Create feature vector in correct order
            X_input = pd.DataFrame([{f: input_data.get(f, 0) for f in feature_names}])

            # Predict
            prob = model.predict_proba(X_input)[0][1]

            # SHAP values
            shap_vals = explainer.shap_values(X_input)
            if isinstance(shap_vals, list):
                # Old SHAP: list of arrays per class, each (n_samples, n_features)
                shap_vals_positive = shap_vals[1][0]
            else:
                shap_vals_arr = np.array(shap_vals)
                if shap_vals_arr.ndim == 3:
                    # New SHAP: (n_samples, n_features, n_classes)
                    shap_vals_positive = shap_vals_arr[0, :, 1]
                else:
                    # (n_samples, n_features)
                    shap_vals_positive = shap_vals_arr[0]

            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1]

        # ──── Results ────
        st.markdown('<div class="step-header">分析結果</div>', unsafe_allow_html=True)

        # Gauge Chart
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            gauge_fig = create_gauge_chart(prob)
            st.plotly_chart(gauge_fig, use_container_width=True)

        with res_col2:
            # HEXACO Radar
            personality_scores = {
                "honesty_humility": honesty,
                "emotionality": emotionality,
                "extraversion": extraversion,
                "agreeableness": agreeableness,
                "conscientiousness": conscientiousness,
                "openness": openness,
            }
            radar_fig = create_hexaco_radar(personality_scores, ref_df)
            st.plotly_chart(radar_fig, use_container_width=True)

        # SHAP Waterfall
        st.markdown("#### 要因分析")
        waterfall_fig = create_shap_waterfall(shap_vals_positive, feature_names, base_val)
        st.plotly_chart(waterfall_fig, use_container_width=True)

        # Interpretation
        st.markdown("#### 解釈のポイント")

        top_factors = np.argsort(np.abs(shap_vals_positive))[::-1][:3]
        for i in range(len(top_factors)):
            idx = top_factors[i]
            if hasattr(idx, 'item'):
                idx = idx.item()
            else:
                idx = int(idx)
            feat = feature_names[idx]
            val = shap_vals_positive[idx]
            if hasattr(val, 'item'):
                val = val.item()
            else:
                val = float(val)
            label = FEATURE_LABELS_JP.get(feat, feat)
            direction = "リスク上昇" if val > 0 else "リスク低下"
            icon = "🔴" if val > 0 else "🟢"
            st.markdown(f"{icon} **{label}**: {direction}に寄与 (SHAP値: {val:+.3f})")

        # Japan Context
        if japan_ctx is not None:
            st.markdown("#### 日本の統計的コンテキスト")
            judicial = japan_ctx[
                (japan_ctx["stat_type"] == "judicial") &
                (japan_ctx["infidelity_filing_rate"].notna())
            ]
            if not judicial.empty:
                # Find matching age group
                age_str = f"{(age // 10) * 10}-{(age // 10) * 10 + 9}"
                if age >= 60:
                    age_str = "60+"
                matched = judicial[judicial["age_group"] == age_str]
                if not matched.empty:
                    avg_rate = matched["infidelity_filing_rate"].mean()
                    st.info(
                        f"📊 日本の司法統計によると、{age_str}歳の年齢層では離婚調停申立ての"
                        f"約 **{avg_rate:.1%}** が「不貞行為」を理由としています。"
                    )

        # Sources
        st.markdown("---")
        st.markdown("#### 学術的根拠・データソース")
        sources = [
            ("Fair (1978)", "Extramarital Affairs Dataset, 6,366件"),
            ("GSS (NORC)", "General Social Survey, EVSTRAY/XMARSEX変数"),
            ("Vowels et al. (2022)", "Is Infidelity Predictable? (OSF)"),
            ("Reinhardt & Reinhard (2023)", "HEXACO Honesty-Humility, 5,677件"),
            ("裁判所", "司法統計年報 (離婚調停申立て動機)"),
            ("厚生労働省", "人口動態調査 (離婚統計)"),
            ("Open Psychometrics", "Big Five Personality Test, 19,719件"),
        ]
        for name, desc in sources:
            st.markdown(f'<span class="source-tag">{name}</span> {desc}', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
