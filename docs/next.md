# 続きの作業 (2026-11-18 時点)

目標は、人が確かめなくても楽譜 (PDF) を正しく読み取れること。試験台 `bench.py` で「正しい小節」と、間違っているのに印のない小節 (silent) を測り、
策を1つずつ試して、効いたものだけを採っている。試した策と数字はすべて [experiments.md](experiments.md) にある (0〜21)。
週間の利用上限に達したので、ここで止めた。

## 今の数字

dev (13曲 348小節、MuseScore 条件、本番どおり、コード d80f572):

| | 正しい小節 | silent | 印 (うち合っている小節) |
| --- | --- | --- | --- |
| 2026-11-17 の作業前 | 310 (89.1%) | 22 | 49 (33) |
| 今 | 325 (93.4%) | 14 | 34 (25) |

test (7曲 370小節、両条件) は 6162e2b で 352 (95.1%)、silent 14、印 47 (43)。17・19・20 を入れた版ではまだ測っていない。

## 次にやること

1. test を今のコードで測る。止めた run の続きなので、そのまま回し直せばよい (下の「回し方」)。
   結果を experiments.md に「試験用の曲で 17〜20 を確かめる」として書く。
2. dev の silent 14 を減らす。内訳 (20 までの結果):
   - clara_polonaise1 22・23・34・35: 段の終わりで切れた弧がタイ (シ♮が続く) かスラー (次はシ♭) か、譜面だけでは決まらない。
     正解のほうがあいまいなので、採点から外すか、印を付ける規則 (段末で切れる弧の先に臨時記号のない音) を考える。
   - dichterliebe2 5・9・11・15・17、schoenberg_op19_6 6・7、clara_polonaise1 27: 2つの読みがどちらも間違っているのに、
     pick で Claude が片方 (たいてい A) を選んでいる。「どちらでもない」と答えさせる工夫が要る。
     例: 違う音の一覧 (19) を1音ずつ「写真にあるか」と問う形にする、選んだ読みの違う音だけを拡大して見せる、など。
   - clara_polonaise3 22、dichterliebe2 18: 2つの読みが一致したまま間違っている。
3. 誤った印 25 を減らす。pick の「決まらなかった」印をやめると silent が増える (18 で確かめた) ので、別の手がかりが要る。
4. test の silent の多い曲 (Verovio 条件の test_clara_trio3、MuseScore 条件の test_clara_polonaise4) を調べる。
   ただし test の曲に合わせて調整はしない (見るのは傾向だけ)。

試して効かなかったもの (繰り返さない): 見比べ画像の分割 (9)、タイを数えさせる (10)、替えた小節の保護 (13)、高い音への印 (14)、
pick の印をやめる (18)、Claude に小節を書き出させて照合 (21、正解と一致するのは96小節中16小節だけ)。

## 回し方

```
uv run python bench.py raw                       # Claude なし。数分。OMR の結果は bench/work にキャッシュされる
BENCH_JOBS=3 uv run python bench.py pick --yes-full --cond=musescore --label=名前   # pick の段まで
BENCH_JOBS=3 uv run python bench.py full --yes-full --cond=musescore --label=名前   # 本番どおり。dev で約40分
BENCH_JOBS=3 uv run python bench.py full --yes-full --piece=test_aloha_oe --piece=test_chorale_bwv153_1 \
  --piece=test_chorale_bwv347 --piece=test_clara_polonaise2 --piece=test_clara_polonaise4 \
  --piece=test_clara_trio3 --piece=test_handel_lascia --label=名前   # test、両条件
```

結果は `bench/work/results_<段>_<名前>.json` と `bench/work/history.jsonl` に残る。同じ段・同じ曲の run を2つ同時に回すと、
`bench/work/lib/<曲>_<条件>_<段>` を取り合って壊れるので、1つずつ回す。

## 注意

- `bench/` は git に入っていない (正解・画像・OMR のキャッシュ・結果)。別の環境では `uv run --with music21 python make_truth.py`
  で正解を作り直すと、12 の変更で beach_prayer の弧が1つスラーからタイに変わり (鳴る音は同じ)、譜面の画像も変わる。
  過去の数字とは厳密には比べられなくなる。
- 正解ファイルの更新時刻が画像より新しいと、試験台が画像を描き直す。正解を書き換えずに作り直しを確かめたいときは、
  元のファイルを `cp -p` で退避して戻す。
- MuseScore 4 の PDF 書き出しはときどき落ちる。試験台は5回までやり直すが、それでも落ちたら回し直す。
- Claude の呼び出しには `--strict-mcp-config` を付けている。外すと MCP の道具の説明で1回あたり約6万トークン増える。
- 利用上限に達すると、`server.claude` は解除まで10分ごとに再試行して待つ。週間上限のときは長く待つことになるので、
  run を止めておく。
- 最後の見比べだけ、pick だけをやり直して比べた試行では、使い捨てのスクリプトを使った (リポジトリには入れていない)。
  やり方は experiments.md の各節に書いてある (本番どおりの結果の lib を使い、meta.json の印の文言で出どころを分けて採点し直す)。
