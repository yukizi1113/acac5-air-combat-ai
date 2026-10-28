# 提出物の版と実戦スコア

全提出物は `args.json` の `overrides` だけが異なります（コードは各時点の TacB 凍結版）。public スコアは 2026-10-28 時点です。

共通の基準 **P2** = `{"attack_alt":13500,"cruise_alt":13000,"escort_max_dist":90000,"k_escort":1.2,"pk_thr_escort":0.55,"race_tte":99999,"engage_radius":12000}`

| 提出物 | コード | overrides | public | 状態 |
|---|---|---|---|---|
| acac5_v7_rule.zip | snap_v7 | `{}`（12 km、Pk≥0.6、70 km） | 1697.5 | |
| acac5_v9b.zip | snap_v9d | `{"k_escort":1.2,"race_tte":99999,"engage_radius":12000,"pk_thr_escort":0.65,"escort_max_dist":85000,"attack_alt":14000,"cruise_alt":13500}` | 1702.1 | 選択期間あり（1799.6→1815→1702） |
| acac5_v9_p2.zip | snap_v9d | P2 | **1827.2** | 最良 |
| acac5_v9c_pk.zip | snap_v9h | v9b と同じ＋Pk モデル再学習 | 1739.1 | |
| acac5_v9c_adapt.zip | snap_v9k | v9b ＋ `"adapt_alt":true` | 1574.2 | 初期格付けのみ |
| acac5_v10_loft.zip | snap_v9m | P2 ＋ `"loft_alt":15500` | 1815.2 | |
| acac5_v10_loft_raid.zip | snap_v9m | P2 ＋ `"loft_alt":15500,"raid_shot":true,"defend_ignore_vt":true` | 1744.0 | |
| acac5_v11_rush.zip | snap_v10a | P2 ＋ `"attack_ignore_vt":true` | 1754.4 | |
| acac5_v11_rush3.zip | snap_v10a | rush ＋ `"n_interceptors":1` | 1706.4 | |
| acac5_v12_tc.zip | snap_v10e | P2 ＋ `"tc_hold":6,"tc_fire_first":true` | 1783.4 | |
| acac5_v12_tcx.zip | snap_v10e | tc ＋ `"tc_exempt_attack":true` | 1740.5 | |
| acac5_v13_lr.zip | snap_v13a | tc ＋ `"hot_pk":0.0,"hot_dist":65000,"hot_deg":35,"hot_rn":0.3` | 未採点 | 実戦ログ解析に基づく |

## コード版の系譜（agents/snap_*）

| 版 | 主な追加 |
|---|---|
| v3 | 階層型ルール（司令部・スキル・反射層）、センサ融合 |
| v4 | 近接回避ルールを仮想ミサイルの脅威推定に置換 |
| v7 | 初回提出版。Pk モデル（LightGBM）による射撃管制 |
| v8f / Tac | 迎撃割当の改良、エネルギー管理 |
| v9d | 護衛対象機レースのパラメータ化（race_tte など） |
| v9h | Pk モデルの再学習（1.7 万発） |
| v9k | 高度の適応切替（adapt_alt） |
| v9m | 射撃前上昇（loft） |
| v10a | 攻撃役の仮想脅威無視（attack_ignore_vt） |
| v10e | 脅威時計（tc_*）＋先撃ち |
| v13a | IRST 方位の順序不変化、正対遠距離射撃の rn 上限（hot_rn） |
| TacB（最新） | v10e ＋ IRST 方位リストの順序不変化（暫定ルール2対応）、すり抜け対策（leak_shot、既定オフ） |
