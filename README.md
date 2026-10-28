# 第5回空戦AIチャレンジ（ACAC5）オープン部門 — 階層型ルールエージェントと実験記録

4機の戦闘機を1つのエージェントで指揮し、相手の護衛対象機を先に撃墜する ASRCAISim1（配布 v1.1.0）向けのエージェント一式です。
コード、提出物、約250本の対戦実験の集計、わかったこと・効かなかったこと・次に試すべきことをまとめています。

- 知見の詳細 → [docs/FINDINGS.md](docs/FINDINGS.md)
- 次に試すべきアイデア → [docs/NEXT_IDEAS.md](docs/NEXT_IDEAS.md)
- 版ごとの差分と実戦スコア → [docs/VERSIONS.md](docs/VERSIONS.md)
- 暫定ルール（v1.1.0 不具合対応）への適合確認 → [docs/COMPLIANCE.md](docs/COMPLIANCE.md)
- 実戦ログ（リーダーボード）の解析 → [docs/FIELD_ANALYSIS.md](docs/FIELD_ANALYSIS.md)

> シミュレータ本体（simulator_dist）は配布条件により含めていません。`tools/setup_links.py` で手元の配布物にリンクしてください。

## 実戦（public）スコア

2026-10-28 21:23 時点（上位は 2127.7）。新規投稿の点数は、選択されて実戦の対戦が積まれるまで、初期の格付け対戦だけで決まります。

| 提出物 | 中身 | public | 戦績 |
|---|---|---|---|
| acac5_v9_p2.zip | 攻撃高度13.5 km、護衛対象機を Pk≥0.55・90 km 以内で射撃、レース中は仮想脅威を無視 | **1827.2** | 15-2-1 |
| acac5_v10_loft.zip | p2 ＋護衛対象機の射撃前に15.5 kmへ上昇 | 1815.2 | 17-5-1 |
| acac5_v12_tc.zip | p2 ＋「脅威時計」（被射撃見込みから一定時間で先撃ち→背面離脱） | 1783.4 | 10-2-1 |
| acac5_v11_rush.zip | p2 ＋攻撃役が想定弾で後退しない | 1754.4 | 17-5-1 |
| acac5_v10_loft_raid.zip | loft ＋自陣接近機への射撃強化 | 1744.0 | 16-6-1 |
| acac5_v12_tcx.zip | tc（攻撃役はレース中免除） | 1740.5 | 9-4-0 |
| acac5_v9c_pk.zip | v9b ＋命中確率モデルの再学習 | 1739.1 | 10-3-0 |
| acac5_v11_rush3.zip | rush ＋迎撃役1機・攻撃役3機 | 1706.4 | 17-6-0 |
| acac5_v9b.zip | 攻撃高度14 km、Pk≥0.65・85 km | 1702.1 | 7-6-0 |
| acac5_v7_rule.zip | 初版（12 km、Pk≥0.6・70 km） | 1697.5 | 15-6-2 |
| acac5_v9c_adapt.zip | 敵の高度を見て自機高度を切替 | 1574.2 | 初期格付けのみ |
| acac5_v13_lr.zip | tc ＋正対目標への遠距離先撃ち（65 km 以内） | 未採点 | 実戦ログ解析（docs/FIELD_ANALYSIS.md）から設計 |

実戦ログの解析は [docs/FIELD_ANALYSIS.md](docs/FIELD_ANALYSIS.md) を参照してください。

## 構成

```
agents/TacB/        最新の開発版エージェント（全機能。既定値では各機能オフ＝v8g 相当）
agents/Tac/         v8g 相当
agents/snap_*/      実験・提出に使った凍結版（評価の対戦相手としても使用）
agents/Lab/         交戦ラボ用エージェント（解析専用・提出しない）
tools/              対戦評価・ラボ・集計・提出物作成
rl/                 上位方策の模倣学習→PPO（リーグ自己対戦）
submit/             提出した zip（公式 validate.py で検証済み）
results/            実験集計（all_experiments.csv ほか）
docs/               技術資料・知見
```

### エージェントの構造（agents/TacB/agent.py）
1. **世界モデル**: レーダ航跡（真ID付き3D）と匿名IRST方位のデータアソシエーション（fusion.py、EKF＝tracker.py）、護衛対象機の周回軌道事前分布、撃墜推定。
2. **司令部（rule_commander）**: 自陣護衛機へ向かう敵（raider）に正対迎撃役を割当、残りは敵護衛機の将来位置へ攻撃。
3. **スキル**: ATTACK / ENGAGE / DEFEND / DRAG / SUPPORT（クランク） / EVADE / VDRAG。
4. **反射層**: MWS回避、仮想ミサイルによる見えない脅威の推定（vthreat.py）、脅威時計、場外防止、エネルギー管理。
5. **射撃管制**: 学習済み命中確率モデル（pk_model.json, LightGBM）＋射程正規化 rn。

主要な挙動は `overrides`（args.json）で切り替えます。提出物の設定は [docs/VERSIONS.md](docs/VERSIONS.md) を参照。

## 使い方（WSL2/Linux, Python 3.10）

```bash
python tools/setup_links.py --sim-dir ../simulator_dist      # 配布物へのリンク
# 対戦評価（色交互、2並列）
python tools/run_matches.py --me agents/TacB --opp agents/snap_v7 --n 16 --workers 2 \
  --me-args '{"overrides":{"attack_alt":13500,"cruise_alt":13000,"escort_max_dist":90000,"k_escort":1.2,"pk_thr_escort":0.55,"race_tte":99999,"engage_radius":12000}}'
# 固定パネル（v7 / スナイパー型 / p2 / ベンチマーク）での評価と集計
tools/panel2.sh NAME agents/TacB '{"...":...}' 16 ; python tools/panel_summary.py NAME
# 交戦ラボ（命中・回避の計測）
python tools/lab.py --episodes 60 --workers 2 --out results/col_lab_x.jsonl
LAB_COLD=1 python tools/lab.py ...           # 正対から一定時間後に背面へ向けた場合の被命中率
python tools/esc_lab.py --helper 1 ...       # 護衛対象機への射撃（僚機誘導あり）
# 提出物の作成と公式検証
python tools/make_submission.py --src agents/TacB --pk agents/TacB/pk_model.json --overrides '{...}' --out submit/x.zip
python tools/check_submission.py submit/x.zip
```
