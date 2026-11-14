"""正解つきの楽譜で、読み取りの出来を数える開発用ツール (render.py と同じく、アプリ本体からは使わない)。

bench/truth/*.musicxml (正解) を Verovio で写真風の画像にし、アプリの読み取りに通して、鳴る音の食い違いを数える。
  uv run python bench.py raw fixed        校正なし (fixed はオクターブの自動修正あり)
  uv run python bench.py full --yes-full  Claude を使う本番どおりの処理 (サブスクリプションを使う)
  uv run python bench.py --selftest       採点そのものの検算
"""
import io
import json
import shutil
import subprocess
import sys
import time
import types
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import resvg_py
import verovio
from PIL import Image, ImageFilter

import measure
import server

BENCH = Path(__file__).parent / "bench"
TRUTH, WORK = BENCH / "truth", BENCH / "work"
FONT, SCALE, WIDTH = "Leland", 100, 1700  # アプリの確認用の描画 (既定の字体) と違う字体で描く
STAGES = ("raw", "fixed", "full")
CONDS = ("clean", "photo")
NAMES = "C C# D D# E F F# G G# A A# B".split()
STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# defer: リピート・括弧・D.C.・テンポ・強弱・スラー・連桁・音部記号の種類は採点しない (音の高さに反映済みか、鳴る音に関係しない)。
# 校正や見比べがそれらを直す変更を入れたら、鳴る順に展開した比較と記号ごとの採点を足す


# ---------- 画像 ----------

def engrave(name):
    """正解を Verovio でページ画像 (JPEG) にし、(clean, photo) の2条件で bench/work/pages/<name>_<cond>/p{n}.jpg に書く。"""
    tk = verovio.toolkit(False)
    tk.setResourcePath(str(Path(verovio.__file__).parent / "data"))
    tk.setOptions({"font": FONT, "pageWidth": 2100, "pageHeight": 2970, "scale": SCALE, "header": "none", "footer": "none",
                   "breaks": "auto", "adjustPageHeight": False})
    tk.loadFile(str(TRUTH / f"{name}.musicxml"))
    pages = tk.getPageCount()
    if pages > 3:
        raise SystemExit(f"{name}: {pages}ページになった (3ページまで)。bench/truth の小節を減らすか SCALE を上げる")
    for cond in CONDS:
        shutil.rmtree(WORK / "pages" / f"{name}_{cond}", ignore_errors=True)
        (WORK / "pages" / f"{name}_{cond}").mkdir(parents=True)
    for n in range(1, pages + 1):
        png = bytes(resvg_py.svg_to_bytes(svg_string=tk.renderToSVG(n), background="white", width=WIDTH))
        im = Image.open(io.BytesIO(png)).convert("RGB")
        im.save(WORK / "pages" / f"{name}_clean" / f"p{n}.jpg", quality=92)  # アプリが写真を保存するのと同じ品質
        degrade(im, zlib.crc32(f"{name}{n}".encode())).save(WORK / "pages" / f"{name}_photo" / f"p{n}.jpg", quality=70)
    return pages


def degrade(im, seed):
    """写真らしい劣化 (同じ seed なら同じ結果): 約1度の傾き、軽いぼかし、明るさのむら、紙の色、ノイズ。JPEG 圧縮は保存側。"""
    rng = np.random.default_rng(seed)
    im = im.rotate(1.0 if seed % 2 else -1.0, resample=Image.BILINEAR, expand=True, fillcolor=(225, 223, 218))
    im = im.filter(ImageFilter.GaussianBlur(0.8))
    a = np.asarray(im, dtype=np.float32)
    h, w = a.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 0.82 + 0.18 * (0.6 * x / w + 0.4 * y / h)  # 片側が暗い
    light *= 1 - 0.12 * (((x / w - 0.5) ** 2 + (y / h - 0.5) ** 2) * 2)  # 四隅が少し暗い
    a = a * light[..., None] * np.array([1.0, 0.985, 0.95], dtype=np.float32) + rng.normal(0, 3, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


# ---------- 音の一覧と採点 ----------

def q(x):
    """music21 が書き出す連符は 827/5040 のように割り切れない長さになる。1/48 拍に丸めて、連符を正しい長さで比べる。
    # defer: 1/48 拍より短い音 (32分の3連より細かい) は区別できない。そういう曲を入れる時に分解能を上げる"""
    return Fraction(round(x * 48), 48)


def events(root):
    """最初のパートを小節ごとの {"ev": [(段, 拍位置, 高さ, 長さ)], "key", "time"} にする。位置と長さは4分音符を1とする分数。
    タイでつながった音は小節をまたいでも1つにまとめ、装飾音符と休符は数えない。声部は区別しない。"""
    out, div, key, time, open_ties = [], 1, 0, ("4", "4"), {}
    for m in root.find("part").findall("measure"):
        ev, cursor, last = [], 0, 0
        for el in m:
            if el.tag == "attributes":
                div = int(el.findtext("divisions", div))
                key = int(el.findtext("key/fifths", key))
                t = el.find("time")
                time = (t.findtext("beats"), t.findtext("beat-type")) if t is not None else time
            elif el.tag in ("backup", "forward"):
                cursor += int(el.findtext("duration")) * (-1 if el.tag == "backup" else 1)
            elif el.tag == "note" and el.find("grace") is None:
                dur = int(el.findtext("duration", "0"))
                onset = last if el.find("chord") is not None else cursor
                if el.find("chord") is None:
                    last, cursor = cursor, cursor + dur
                p = el.find("pitch")
                if p is None:
                    continue
                midi = 12 * (int(p.findtext("octave")) + 1) + STEP[p.findtext("step")] + int(float(p.findtext("alter", "0")))
                staff = int(el.findtext("staff", "1"))
                tied = {t.get("type") for t in el.findall("tie")} | {t.get("type") for t in el.findall("notations/tied")}
                if "stop" in tied and (staff, midi) in open_ties:
                    open_ties[(staff, midi)][3] += Fraction(dur, div)
                    if "start" not in tied:
                        del open_ties[(staff, midi)]
                    continue
                e = [staff, Fraction(onset, div), midi, Fraction(dur, div)]
                ev.append(e)
                if "start" in tied:
                    open_ties[(staff, midi)] = e
        out.append({"ev": ev, "key": key, "time": time})
    for m in out:
        m["ev"] = [(e[0], q(e[1]), e[2], q(e[3])) for e in m["ev"]]
    return out


def fmt(e):
    return {"staff": e[0], "onset": str(e[1]), "pitch": f"{NAMES[e[2] % 12]}{e[2] // 12 - 1}", "dur": str(e[3])}


def pair_up(T, R, key, dist):
    """key が同じ音どうしを、dist が近い順に1対1で組にする。(組, 余ったT, 余ったR) を返す。"""
    groups = defaultdict(lambda: ([], []))
    for i, t in enumerate(T):
        groups[key(t)][0].append(i)
    for j, r in enumerate(R):
        groups[key(r)][1].append(j)
    pairs = []
    for ts, rs in groups.values():
        cand = sorted((dist(T[i], R[j]), i, j) for i in ts for j in rs)
        used_t, used_r = set(), set()
        for _, i, j in cand:
            if i not in used_t and j not in used_r:
                used_t.add(i), used_r.add(j)
                pairs.append((T[i], R[j]))
    gone_t, gone_r = {id(t) for t, _ in pairs}, {id(r) for _, r in pairs}
    return pairs, [t for t in T if id(t) not in gone_t], [r for r in R if id(r) not in gone_r]


def diff_measure(T, R):
    """1小節の音を、正しい / 高さ違い (オクターブ・半音・その他) / 長さ違い / 位置違い / 欠け / 余りに分ける。
    (数, 食い違いの一覧) を返す。優先順は高さ→長さ→位置。"""
    n, why = Counter(), []
    same = Counter(T) & Counter(R)
    n["correct"] = sum(same.values())
    T, R = list((Counter(T) - same).elements()), list((Counter(R) - same).elements())
    for kind, key, dist in (("pitch", lambda e: (e[0], e[1], e[3]), lambda t, r: abs(t[2] - r[2])),
                            ("duration", lambda e: (e[0], e[1], e[2]), lambda t, r: abs(t[3] - r[3])),
                            ("onset", lambda e: (e[0], e[2], e[3]), lambda t, r: abs(t[1] - r[1]))):
        pairs, T, R = pair_up(T, R, key, dist)
        for t, r in pairs:
            d = abs(t[2] - r[2])
            k = ("octave" if d % 12 == 0 else "semitone" if d == 1 else "other_pitch") if kind == "pitch" else kind
            n[k] += 1
            why.append({"kind": k, "truth": fmt(t), "result": fmt(r)})
    n["missing"], n["extra"] = len(T), len(R)
    why += [{"kind": "missing", "truth": fmt(t)} for t in T] + [{"kind": "extra", "result": fmt(r)} for r in R]
    return n, why


def align(T, R):
    """小節の並びどうしを Needleman-Wunsch で合わせる (アプリは小節を落とす・割る・足すことがある)。
    費用は音の食い違いの数。(truthの添字, resultの添字) の組を返す。"""
    cost = lambda a, b: sum(((Counter(a["ev"]) - Counter(b["ev"])) + (Counter(b["ev"]) - Counter(a["ev"]))).values())
    n, m = len(T), len(R)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.cumsum([0] + [len(t["ev"]) + .5 for t in T])
    D[0, :] = np.cumsum([0] + [len(r["ev"]) + .5 for r in R])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + .9 * cost(T[i - 1], R[j - 1]),
                          D[i - 1, j] + len(T[i - 1]["ev"]) + .5, D[i, j - 1] + len(R[j - 1]["ev"]) + .5)
    i, j, pairs = n, m, []
    while i and j:
        if D[i, j] == D[i - 1, j - 1] + .9 * cost(T[i - 1], R[j - 1]):
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + len(T[i - 1]["ev"]) + .5:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def score(truth, result):
    """正解と読み取り結果 (どちらも root) を比べる。{"n": 数, "right": [小節ごとの○×], "wrong": 食い違いのある小節の一覧}。"""
    T, R = events(truth), events(result)
    pairs = align(T, R)
    hit_t, hit_r = {i for i, _ in pairs}, {j for _, j in pairs}
    n = Counter(measures=len(T), result_measures=len(R), missing_measures=len(T) - len(hit_t), extra_measures=len(R) - len(hit_r))
    right, wrong = [False] * len(T), []
    for i, t in enumerate(T):
        if i not in hit_t:
            n["missing"] += len(t["ev"])
            wrong.append({"measure": i + 1, "what": "measure missing", "notes": len(t["ev"])})
    for j, r in enumerate(R):
        if j not in hit_r:
            n["extra"] += len(r["ev"])
            wrong.append({"result_measure": j + 1, "what": "extra measure", "notes": len(r["ev"])})
    for i, j in pairs:
        c, why = diff_measure(T[i]["ev"], R[j]["ev"])
        n.update(c)
        right[i] = not why
        for k in ("key", "time"):
            if T[i][k] != R[j][k]:
                n[f"{k}_wrong"] += 1
                why.append({"kind": f"{k}_wrong", "truth": T[i][k], "result": R[j][k]})
        if why:
            wrong.append({"measure": i + 1, "result_measure": j + 1, "what": why})
    n["right_measures"] = sum(right)
    n["truth_notes"] = sum(len(t["ev"]) for t in T)
    return {"n": n, "right": right, "wrong": sorted(wrong, key=lambda w: w.get("measure", 10 ** 6))}


# ---------- 読み取り ----------

def cached_run(real, cache):
    """server.subprocess.run の代わり。homr を呼ぶ処理だけ、画像ごとに結果を bench/work/omr/ に取っておき、2回目からは OMR を飛ばす。"""
    def run(cmd, cwd=None, **kw):
        if "--write-staff-positions" not in cmd:
            return real(cmd, cwd=cwd, **kw)
        d, name = Path(cwd), Path(cmd[-1]).stem
        key = WORK / "omr" / f"{zlib.crc32((d / cmd[-1]).read_bytes()):08x}"
        if not (key / f"{name}.txt").exists():  # 五線の位置 (.txt) は校正の段の切り出しに要る
            r = real(cmd, cwd=cwd, **kw)
            if not (d / f"{name}.musicxml").exists():
                return r
            key.mkdir(parents=True, exist_ok=True)
            for f in (f"{name}.musicxml", f"{name}_geo.json", f"{name}.txt"):
                if (d / f).exists():
                    shutil.copy(d / f, key / f)
            return r
        for f in key.iterdir():
            shutil.copy(f, d / f.name)
        return types.SimpleNamespace(stderr="", stdout="", returncode=0)
    return run


def read_song(name, cond, stage):
    """画像の並びを、隔離したライブラリの1曲として server.process_page に1ページずつ通し、score.musicxml の root を返す。
    raw / fixed は Claude を呼ぶ proof_page を空にして動かす。full は本物 (呼び出し側が許可を確認済み)。"""
    song = f"{name}_{cond}_{stage}"
    d = WORK / "lib" / song
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    pages = sorted((WORK / "pages" / f"{name}_{cond}").glob("p*.jpg"), key=lambda p: int(p.stem[1:]))
    for p in pages:
        shutil.copy(p, d / p.name)
    (d / "meta.json").write_text(json.dumps({"id": song, "title": song, "pages": len(pages), "measures": 0, "pending": 0,
                                             "step": None, "log": []}))
    saved = server.LIB, server.proof_page, server.subprocess, measure.fix_octaves
    server.LIB = WORK / "lib"  # library/ には書かない
    server.subprocess = types.SimpleNamespace(run=cached_run(subprocess.run, WORK), DEVNULL=subprocess.DEVNULL)
    if stage != "full":
        server.proof_page = lambda song, n: None
    if stage == "raw":
        measure.fix_octaves = lambda root, geo: []
    try:
        for n in range(1, len(pages) + 1):
            server.process_page(song, n)
    finally:
        server.LIB, server.proof_page, server.subprocess, measure.fix_octaves = saved
    return ET.parse(d / "score.musicxml").getroot()


# ---------- 集計 ----------

KINDS = ("correct", "octave", "semitone", "other_pitch", "duration", "onset", "missing", "extra")


def total(ns):
    t = Counter()
    for n in ns:
        t.update(n)
    return t


def line(label, n):
    notes = n["truth_notes"] or 1
    return (f"{label:<22}{n['right_measures']:>4}/{n['measures']:<4}{100 * n['right_measures'] / max(n['measures'], 1):>5.0f}%  "
            f"{100 * n['correct'] / notes:>4.0f}%  " + " ".join(f"{n[k]:>4}" for k in KINDS[1:])
            + f"   {n['missing_measures']:>3} {n['extra_measures']:>3}   {n['key_wrong']:>3} {n['time_wrong']:>3}")


def report(runs):
    """runs: {(stage, cond): {piece: score}}。条件ごとに全体と曲ごとの表を出す。"""
    head = f"{'':<22}{'measures right':>14}  notes  " + " ".join(f"{k[:4]:>4}" for k in KINDS[1:]) + "   mis ext   key time"
    for (stage, cond), res in runs.items():
        print(f"\n[{stage} / {cond}]\n{head}")
        print(line("ALL", total(r["n"] for r in res.values())))
        for piece, r in res.items():
            print(line(piece, r["n"]))
    for cond in CONDS:  # オクターブ修正の効果: 小節単位で、直って正解になった数と、壊れた数
        if ("raw", cond) in runs and ("fixed", cond) in runs:
            a, b = runs["raw", cond], runs["fixed", cond]
            better = sum(not x and y for p in a for x, y in zip(a[p]["right"], b[p]["right"]))
            worse = sum(x and not y for p in a for x, y in zip(a[p]["right"], b[p]["right"]))
            na, nb = total(r["n"] for r in a.values()), total(r["n"] for r in b.values())
            print(f"\noctave fix ({cond}): measures newly right {better}, newly wrong {worse}; "
                  f"notes correct {na['correct']} -> {nb['correct']}, octave errors {na['octave']} -> {nb['octave']}")


# ---------- 検算 ----------

def selftest():
    """正解どうしは全部合うこと、1種類ずつ入れた間違いが、その種類だけに数えられることを確かめる。"""
    import copy
    base = ET.parse(TRUTH / "mozart_k545.musicxml").getroot()
    for f in sorted(TRUTH.glob("*.musicxml")):
        r = ET.parse(f).getroot()
        s = score(r, r)
        assert s["n"]["right_measures"] == s["n"]["measures"] and not s["wrong"] and s["n"]["correct"] == s["n"]["truth_notes"], f.name

    def notes(root, m):
        return root.find("part").findall("measure")[m].findall("note")
    def pitched(root, m):  # 和音でない、休符でない音
        return [n for n in notes(root, m) if n.find("pitch") is not None and n.find("chord") is None]
    def pitch_up(r):
        n = pitched(r, 2)[1]
        a = n.find("pitch/alter")
        if a is None:
            a = ET.SubElement(n.find("pitch"), "alter")
            a.text = "0"
            n.find("pitch").remove(a)
            n.find("pitch").insert(1, a)
        a.text = str(int(float(a.text)) + 1)
    def octave(r):
        pitched(r, 2)[1].find("pitch/octave").text = str(int(pitched(r, 2)[1].findtext("pitch/octave")) + 1)
    def other(r):  # 全音2つ上: 音名を2つ進める (B は次の C へ行くので使わない)
        n = [n for n in pitched(r, 2) if n.findtext("pitch/step") in "CDEF"][0]
        n.find("pitch/step").text = "CDEFGAB"["CDEFGAB".index(n.findtext("pitch/step")) + 1]
    def dur(r):  # 小節の最後の音は、伸ばしても後ろの音がずれない
        n = notes(r, 3)[-1]
        assert n.find("chord") is None
        n.find("duration").text = str(int(n.findtext("duration")) * 2)
    def drop(r):  # 和音の2音目を消す (後ろの音がずれない)
        n = [n for n in notes(r, 5) if n.find("chord") is not None][0]
        r.find("part").findall("measure")[5].remove(n)
    def add(r):
        n = copy.deepcopy(pitched(r, 4)[0])
        n.find("pitch/octave").text = str(int(n.findtext("pitch/octave")) + 2)
        n.insert(0, ET.Element("chord"))
        r.find("part").findall("measure")[4].insert(list(r.find("part").findall("measure")[4]).index(pitched(r, 4)[0]) + 1, n)
    def del_measure(r):
        part = r.find("part")
        part.remove(part.findall("measure")[7])
    cases = {"semitone": pitch_up, "octave": octave, "other_pitch": other, "duration": dur, "missing": drop,
             "extra": add, "missing_measures": del_measure}
    base_n = score(base, base)["n"]
    for kind, inject in cases.items():
        r = copy.deepcopy(base)
        inject(r)
        s = score(base, r)
        got = {k: s["n"][k] for k in (*KINDS[1:], "missing_measures", "extra_measures", "key_wrong", "time_wrong") if s["n"][k]}
        if kind == "missing_measures":
            want = {"missing": len(events(base)[7]["ev"]), "missing_measures": 1}
        else:
            want = {kind: 1}
        assert got == want, f"{kind}: {got} != {want}"
        assert s["n"]["right_measures"] == base_n["measures"] - 1, f"{kind}: right {s['n']['right_measures']}"
    print("selftest OK:", ", ".join(cases))


# ---------- 実行 ----------

def main(argv):
    if "--selftest" in argv:
        return selftest()
    stages = [a for a in argv if a in STAGES]
    only = [a[8:] for a in argv if a.startswith("--piece=")]
    pieces = [p.stem for p in sorted(TRUTH.glob("*.musicxml")) if not only or p.stem in only]
    if not stages or not pieces:
        raise SystemExit(__doc__)
    pages = {p: engrave(p) for p in pieces}
    if "full" in stages:
        n = sum(pages.values()) * len(CONDS)
        print(f"full は Claude を使う本番どおりの処理です。{n}ページ処理します (1ページ数分、Claude を10回前後呼ぶ)")
        if "--yes-full" not in argv:
            raise SystemExit("実行するなら --yes-full を付ける")
    runs = {}
    for stage in stages:
        for cond in CONDS:
            runs[stage, cond] = {}
            for p in pieces:
                t0 = time.time()
                result = read_song(p, cond, stage)
                runs[stage, cond][p] = score(ET.parse(TRUTH / f"{p}.musicxml").getroot(), result)
                print(f"{stage} {cond} {p}: {pages[p]}ページ {time.time() - t0:.0f}秒", flush=True)
        (WORK / f"results_{stage}.json").write_text(json.dumps(
            {cond: {p: {"counts": r["n"], "wrong_measures": r["wrong"]} for p, r in runs[stage, cond].items()} for cond in CONDS},
            ensure_ascii=False, indent=1))
    report(runs)


if __name__ == "__main__":
    main(sys.argv[1:])
