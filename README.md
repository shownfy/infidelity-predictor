# 浮気リスク予測 Web アプリ

学術研究に基づき、パートナーの性格特性・関係性情報から浮気リスクを統計的に予測する Streamlit Web アプリケーションです。

## 概要

- **予測モデル**: RandomForest Classifier + SHAP (解釈可能AI)
- **データパイプライン**: dbt-core + DuckDB (ディメンショナルモデリング)
- **データソース**: 7つの学術・政府データセット
- **UI**: Streamlit (ライトテーマ、ステップ形式入力、ゲージチャート・レーダーチャート・SHAP寄与度グラフ)

## データソース

| ソース | 件数 | 役割 |
|--------|------|------|
| Fair's Extramarital Affairs (1978) | 6,366 | コア訓練データ |
| General Social Survey (GSS) | 10,000 | コア訓練データ |
| Selterman/Vowels OSF | 1,295 | コア訓練データ |
| Reinhardt HEXACO (2023) | 5,677 | HEXACO Honesty-Humility |
| 日本 司法統計 | 24 | コンテキスト表示 |
| 日本 人口動態調査 | 17 | コンテキスト表示 |
| Open Psychometrics Big Five | 19,719 | 性格特性の正規化 |

## セットアップ

```bash
# 仮想環境作成
python3 -m venv .venv
source .venv/bin/activate

# 依存関係インストール
pip install -r requirements.txt

# データ取得
python scripts/fetch_fair.py
python scripts/fetch_gss.py
python scripts/fetch_selterman.py
python scripts/fetch_reinhardt.py
python scripts/fetch_japan_stats.py
python scripts/fetch_big_five.py

# dbt パイプライン実行
dbt seed --profiles-dir .
dbt run --profiles-dir .

# モデル訓練
python scripts/train_model.py

# アプリ起動
streamlit run app/main.py
```

## プロジェクト構成

```
infidelity-predictor/
├── app/
│   └── main.py              # Streamlit アプリ
├── images/
│   └── image.png            # ヘッダー画像
├── models/
│   ├── staging/              # ソースデータのクレンジング
│   ├── dimensions/           # ディメンションテーブル
│   │   ├── dim_person.sql
│   │   ├── dim_personality.sql
│   │   ├── dim_relationship.sql
│   │   ├── dim_japan_context.sql
│   │   └── dim_personality_reference.sql
│   ├── facts/                # ファクトテーブル
│   │   └── fct_infidelity.sql
│   └── marts/                # ワイドテーブル (ML用)
│       └── marts_infidelity_features.sql
├── models_ml/
│   └── model.pkl             # 訓練済みモデル
├── scripts/                  # データ取得・モデル訓練スクリプト
├── seeds/                    # CSV データファイル
├── dbt_project.yml
├── profiles.yml
├── architecture.md           # アーキテクチャ詳細
└── requirements.txt
```

## 予測モデルの特徴量

- **HEXACO 6因子**: 誠実-謙虚さ、情緒性、外向性、協調性、誠実性、開放性
- **人口統計**: 年齢、教育年数、信仰心、職業
- **関係性**: 交際年数、子供の有無、満足度、愛情度、欲求度

## UI フロー

1. **Step 1**: パートナーの基本情報 (年齢、学歴、職業レベル、信仰心、子供の有無)
2. **Step 2**: パートナーの性格 - HEXACO 6因子 (各項目に具体例付きヘルプあり)
3. **Step 3**: 関係性の情報 (交際期間、満足度、愛情、欲求)
4. **分析結果**: ゲージチャート、HEXACOレーダーチャート、SHAP要因分析、日本の統計コンテキスト

## 免責事項

このツールは学術研究データに基づく統計的予測であり、結果は参考情報としてのみご利用ください。パートナーへの不当な疑いや不信感を助長する目的での使用は推奨しません。

## 主要参考文献

- Fair, R. C. (1978). A Theory of Extramarital Affairs. *Journal of Political Economy*.
- Vowels, L. M., Vowels, M. J., & Mark, K. P. (2022). Is Infidelity Predictable? *Archives of Sexual Behavior*.
- Reinhardt, R. O. & Reinhard, M.-A. (2023). Honesty-Humility and Relationship Dishonesty. *JPSP*.
- Lee, K. & Ashton, M. C. (2004). The HEXACO Personality Factors.
