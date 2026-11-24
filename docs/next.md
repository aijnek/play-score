# 続きの作業 (2026-11-24 時点)

目標は、人が確かめなくても楽譜 (PDF) を正しく読み取れること。試験台 `bench.py` で「正しい小節」と、間違っているのに印のない小節 (silent) を測り、
策を1つずつ試して、効いたものだけを採っている。試した策と数字はすべて [experiments.md](experiments.md) にある (0〜28)。

## 今の数字

| | 正しい小節 | silent | 印 (うち合っている小節) |
| --- | --- | --- | --- |
| dev (13曲 348小節、MuseScore 条件)、2026-11-24 の基準 | 324 (93.1%) | 21 | 32 (29) |
| dev、23〜26 を入れたあと | 331 (95.1%) | 9 | 18 (10) |
| test (7曲 370小節、両条件)、6162e2b | 352 (95.1%) | 14 | 47 (43) |
| test、23〜28 を入れたあと | 356 (96.2%) | 6 | 39 (31) |

dev の「23〜26 を入れたあと」は段ごとのやり直しをつないだ数字で、27・28 を入れた通しの測定は dev では行っていない。

## 次にやること

1. dev を今のコードで通して測る (27 の「PDF と合っていた小節を戻す」と 28 の臨時記号を入れた状態の確認)。
2. 残る間違いは音の長さとタイの終わり側。符頭の検査では見えない。PDF には旗 (U+E240〜)・連桁 (太い線)・付点・休符のグリフも入っているので、
   声部ごとの長さを PDF から組み立てて比べられるかを調べる。まず正解との検算 (`glyphs.py` の検算と同じやり方) で読み取りの確かさを見る。
3. 誤った印を減らす。MuseScore 条件の test では、pick の印 9 と見比べ (verify) の印 9 がすべて合っている小節に付いていた。
   - pick の時点でデータの小節の数が PDF と合わないページでは、PDF と比べられず pick の印が残る。小節の数が合わないこと自体を手がかりにする
     (PDF のほうが正しい)。
   - PDF と合っている小節に付いた verify の印を外してよいかを、dev と test の結果で数える。
4. clara_polonaise1 22・23・34・35 は、正解のデータが印刷と合っていない (28 に書いた)。採点から外すか、正解を印刷に合わせる。
5. PDF の検査は MuseScore の PDF でしか確かめていない。ほかのソフト (Dorico、Finale、Sibelius、LilyPond) の PDF で、グリフが読めるか、
   タイとスラーの区別 (`glyphs.py` の 1.6) が合うかを見る。読めない PDF では検査を行わないだけで、害はないはず。
6. アプリで PDF を取り込む操作 (ブラウザから元の PDF を送る) は、サーバー側の受け口だけを確かめた。実際の PDF をブラウザで取り込んで通す確認が残っている。
7. test_clara_trio3 の拍子記号の誤り (24小節)。

試して効かなかったもの (繰り返さない): 見比べ画像の分割 (9)、タイを数えさせる (10)、替えた小節の保護 (13)、高い音への印 (14)、
pick の印をやめる (18。PDF の検査があるページでは 26 で採用)、Claude に小節を書き出させて照合 (21)、homr を拡大画像で読み直す (22)。

## 回し方

```
uv run python bench.py raw                       # Claude なし。数秒 (OMR の結果は bench/work にキャッシュされる)
BENCH_JOBS=3 uv run python bench.py full --yes-full --cond=musescore --label=名前   # 本番どおり。dev で約40分
BENCH_JOBS=3 uv run python bench.py full --yes-full --cond=musescore --from=名前:glyph --label=別の名前   # その run の PDF の段だけをやり直す
BENCH_JOBS=3 uv run python bench.py full --yes-full --piece=test_aloha_oe --piece=test_chorale_bwv153_1 \
  --piece=test_chorale_bwv347 --piece=test_clara_polonaise2 --piece=test_clara_polonaise4 \
  --piece=test_clara_trio3 --piece=test_handel_lascia --label=名前   # test、両条件
```

- 結果は `bench/work/results_<段階>_<名前>.json` と `bench/work/history.jsonl`、途中のデータは `bench/work/runs/<名前>/` に残る。
  同じ名前の run を2つ同時に回さない。今日の run: base (基準)、pk_glyph (24 の pick まで)、v23 (その続き)、g3 (25)、test25 (test)。
- `--from=名前:段` の段は pick・proof・verify・audit・glyph。Claude の揺れを分けて比べるときは、同じ run から同じ段を2回やり直す。
- 結果の表の下に、段ごとに直した数と壊した数、突き合わせの結果ごとの間違い、印を付けた段ごとの当たり外れが出る。
  小節ごとの記録は結果の JSON の `measures` にある。

## 注意

- `bench/` は git に入っていない (正解・画像・OMR のキャッシュ・結果)。別の環境では `uv run --with music21 python make_truth.py`
  で正解を作り直すと、12 の変更で beach_prayer の弧が1つスラーからタイに変わり (鳴る音は同じ)、譜面の画像も変わる。
  過去の数字とは厳密には比べられなくなる。
- 正解ファイルの更新時刻が画像より新しいと、試験台が画像を描き直す。正解を書き換えずに作り直しを確かめたいときは、
  元のファイルを `cp -p` で退避して戻す。
- MuseScore 4 の PDF 書き出しはときどき落ちる。試験台は5回までやり直すが、それでも落ちたら回し直す。
- PDF の符頭のデータ (`bench/work/pages/<曲>_musescore/p{n}_glyphs.json`) は、`glyphs.py` の読み取りを変えたら消して作り直す
  (消すと、次に試験台を回したときに MuseScore でもう一度 PDF を書き出して読む)。画像は元のものを使う。
  schoenberg_op19_6 だけは、いま書き出す PDF と保存してある画像の割り付けが違う (音は同じ)。
- Claude の呼び出しには `--strict-mcp-config --safe-mode --disable-slash-commands` を付けている。外すと、利用者の MCP・CLAUDE.md・スキルの
  読み込みで呼び出しごとのトークンが増える。
- 利用上限に達すると、`server.claude` は解除まで10分ごとに再試行して待つ。週間上限のときは長く待つことになるので、run を止めておく。
