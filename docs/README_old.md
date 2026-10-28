# ACAC5 (第5回空戦AIチャレンジ) 階層型エージェント 開発一式

## 構成

```
acac5_project/
├── agents/Tac/           提出用パッケージ本体（このフォルダの中身をzip化して投稿）
│   ├── __init__.py       提出必須の4関数 + ポリシー切替（ルール / NN）
│   ├── agent.py          M方式Agent：世界モデル・オプション・反射層・射撃管制
│   ├── fusion.py         レーダ＋IRST(匿名ID)の多目標データアソシエーション
│   ├── tracker.py        EKF（等速度モデル、レーダ3D / IRST方位 / 護衛機周回軌道の事前分布）
│   ├── pk_model.py       命中確率モデルの推論（numpyのみ。LightGBMも可）
│   ├── pk_model.json     学習済み命中確率モデル
│   ├── rl_obs.py         上位方策用エンティティ・トークン観測
│   ├── nn_model.py       Transformer型 Actor-Critic（オプション選択＋射撃モード）
│   └── config.json       戦術パラメータ（既定値は agent.py 内）
├── agents/Lab/           交戦ラボ用エージェント（解析専用・提出しない）
├── rl/rollout.py         学習用エピソード実行（報酬計算・遷移記録）
├── rl/train.py           模倣学習(BC) → PPO（リーグ自己対戦＋PFSP）
├── tools/run_matches.py  並列対戦評価（詳細統計・神視点ミサイルログ）
├── tools/lab.py          交戦ラボ（1対1射撃シナリオを大量生成）
├── tools/train_pk.py     命中確率モデルの学習・比較（GroupKFold）
├── tools/analyze_lab.py  ラボ結果の集計
├── tools/make_submission.py / check_submission.py  提出zip作成・公式validate.pyでの検証
└── docs/                 技術資料
```

## 現状（2026-10-22）
- 最新の提出候補：`submit/acac5_v9b.zip`（公式 validate.py 3勝0敗・タイムアウトなし）
  - 中身は `agents/snap_v9d` ＋ overrides（攻撃高度14 km、護衛対象機射撃 Pk≥0.65・85 km以内、レース時は仮想脅威を無視）
  - 自前パネル（v7・スナイパー型・ベンチマーク）平均 0.77（v7は0.58）
- 中間版：`submit/acac5_v9_p2.zip`（攻撃高度13.5 km版）
- 開発中のエージェント本体は `agents/TacB`（v9系。`agents/Tac` は v8g 相当）
- 旧提出：`submit/acac5_v7_rule.zip`（実戦 1552.6）
- `agents/snap_v3` `snap_v4` `snap_v7` `snap_v8f` `snap_v9d` は評価・リーグ学習の固定対戦相手（削除しないこと）

### v9 で追加した評価・解析ツール
- `tools/panel.sh` / `tools/panel2.sh NAME AGENT 'overrides' [N]`：固定パネル（v7・スナイパー型・突撃型 or p2・ベンチマーク）で評価、`tools/panel_summary.py` で集計
- `tools/esc_lab.py`：護衛対象機への射撃ラボ（高度・距離・アスペクト・射撃後機動・僚機誘導 `--helper 1`）、`tools/analyze_esclab.py` `tools/fit_esclab.py`
- `tools/flight_lab.py`：上昇スケジュール別の前進距離
- `TRACE_FILE=...` を設定すると TacB が2秒ごとの各機の行動・位置を JSONL に記録

## 環境構築（Windows PC：WSL2 推奨）

配布シミュレータの whl は Linux 用のみです。Windows では WSL2 (Ubuntu 22.04) か Docker を使います。

```bash
# PowerShell(管理者) で一度だけ
wsl --install -d Ubuntu-22.04

# 以降 Ubuntu 内
sudo apt update && sudo apt install -y python3.10-venv unzip
cd ~ && mkdir acac && cd acac
cp /mnt/c/Users/honob/Box/Investment/competition/空戦AI/simulator_dist.zip . && unzip -q simulator_dist.zip
cp -r /mnt/c/Users/honob/Box/Investment/competition/空戦AI/acac5_project .   # 本フォルダ
python3.10 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
pip install pandas==2.1.4 scikit-learn==1.6.1 lightgbm==4.5.0 timeout-decorator==0.5.0 opencv-python-headless==4.11.0.86
pip install -r simulator_dist/dist/linux/requirements.txt
cd acac5_project && python tools/setup_links.py --sim-dir ../simulator_dist
python -c "import ASRCAISim1; print('OK')"
```

> Box 同期フォルダ上で直接学習すると同期負荷が大きいので、WSL 側（~/acac）で作業し、成果物だけ Box に戻すことを推奨します。

## よく使うコマンド

```bash
# ベンチマークと評価（色を交互に、2並列）
python tools/run_matches.py --me agents/Tac --opp Agents/BenchMark --n 20 --workers 4 --out results/eval.jsonl

# 命中確率モデルのデータ収集（交戦ラボ）と学習
python tools/lab.py --episodes 300 --workers 4 --out results/col_lab2.jsonl
GOD_LOG=1 SHOT_LOG=$PWD/data/shots_col_m1.jsonl python tools/run_matches.py --me agents/Tac --n 40 --workers 4 --out results/col_m1.jsonl
python tools/train_pk.py --shots 'data/shots_*.jsonl' --results 'results/col_*.jsonl' --out agents/Tac/pk_model.json

# 強化学習（上位方策）
python rl/train.py bc  --workers 6 --episodes 120 --out runs/bc
python rl/train.py ppo --workers 6 --init runs/bc/policy.pt --iters 300 --eps-per-iter 12 --out runs/ppo
# NN方策で評価
python tools/run_matches.py --me agents/Tac --me-args '{"policy":"nn","weights":"../../runs/ppo/policy.pt"}' --n 20

# 提出物
python tools/make_submission.py --policy rule --out submit/acac5_rule.zip
python tools/make_submission.py --policy nn --weights runs/ppo/policy.pt --out submit/acac5_nn.zip
python tools/check_submission.py submit/acac5_nn.zip
```

## 実験の作法
- 実験中に `agents/Tac` を編集すると、対戦ごとに再importされるため結果が混ざります。
  `tools/snap.sh TAG` で `agents/snap_TAG` に凍結してから `tools/exp.sh` に渡してください。
