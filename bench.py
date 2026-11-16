"""正解つきの楽譜で、読み取りの出来を数える開発用ツール (render.py と同じく、アプリ本体からは使わない)。

bench/truth/*.musicxml (正解。make_truth.py で作る) を PDF と同じきれいな画像にし、アプリの読み取りに通して、鳴る音の食い違いを数える。
  uv run python bench.py raw              校正なし
  uv run python bench.py full --yes-full  Claude を使う本番どおりの処理 (サブスクリプションを使う)
  uv run python bench.py --selftest       採点そのものの検算
  --piece=名前 (繰り返し可) で曲を、--cond=verovio|musescore で描き方を絞る。--test で test_ の曲も測る。--label=名前 で結果の名前を付ける
"""
import importlib.metadata
import io
import json
import os
import shutil
import subprocess
import sys
import time
import types
import xml.etree.ElementTree as ET
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pypdfium2
import resvg_py
import verovio
from PIL import Image

import server
from consensus import align, events

BENCH = Path(__file__).parent / "bench"
TRUTH, WORK = BENCH / "truth", BENCH / "work"
WIDTH = 2000  # アプリ (index.html) が PDF のページを画像にするときの横幅。JPEG の品質 0.92 も合わせる
FONTS = ("Leland", "Bravura", "Gootville", "Leipzig")  # Verovio で描くときの字体。曲ごとに替え、PDF を作るソフトの違いに近づける
MSCORE = "/Applications/MuseScore 4.app/Contents/MacOS/mscore"  # brew install --cask musescore
STAGES = ("raw", "full")
CONDS = ("verovio", "musescore")
NAMES = "C C# D D# E F F# G G# A A# B".split()

# defer: リピート・括弧・D.C.・テンポ・強弱・スラー・連桁・音部記号の種類は採点しない (音の高さに反映済みか、鳴る音に関係しない)。
# 校正や見比べがそれらを直す変更を入れたら、鳴る順に展開した比較と記号ごとの採点を足す


# ---------- 画像 ----------

def engrave(name):
    """正解を PDF のページと同じきれいな画像 (横2000px の JPEG) にし、bench/work/pages/<name>_<cond>/p{n}.jpg に書く。
    verovio: Verovio で字体を曲ごとに替えて描く。musescore: MuseScore で PDF にし、それを画像にする。ページ数を返す。"""
    pages, src = {}, (TRUTH / f"{name}.musicxml").stat().st_mtime
    for cond in CONDS:
        out = WORK / "pages" / f"{name}_{cond}"
        done = sorted(out.glob("p*.jpg"))
        if done and min(f.stat().st_mtime for f in done) > src:  # 正解が変わっていなければ描き直さない
            pages[cond] = len(done)
            continue
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        ims = verovio_pages(name) if cond == "verovio" else musescore_pages(name)
        if len(ims) > 5:
            raise SystemExit(f"{name} ({cond}): {len(ims)}ページになった (5ページまで)。make_truth.py で小節を減らす")
        for n, im in enumerate(ims, 1):
            im.save(out / f"p{n}.jpg", quality=92)
        pages[cond] = len(ims)
    return pages


def verovio_pages(name):
    tk = verovio.toolkit(False)
    tk.setResourcePath(str(Path(verovio.__file__).parent / "data"))
    tk.setOptions({"font": FONTS[zlib.crc32(name.encode()) % len(FONTS)], "pageWidth": 2100, "pageHeight": 2970,
                   "scale": 100, "header": "none", "footer": "none", "breaks": "auto", "adjustPageHeight": False})
    tk.loadFile(str(TRUTH / f"{name}.musicxml"))
    return [Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=tk.renderToSVG(n), background="white", width=WIDTH))))
            .convert("RGB") for n in range(1, tk.getPageCount() + 1)]


def musescore_pages(name):
    pdf = WORK / "pages" / f"{name}_musescore" / "score.pdf"
    for _ in range(5):  # MuseScore 4.7 の書き出しは同じ入力でもときどき落ちる
        if not subprocess.run([MSCORE, "-o", str(pdf), str(TRUTH / f"{name}.musicxml")], capture_output=True).returncode:
            break
    else:
        raise SystemExit(f"{name}: MuseScore で PDF にできなかった")
    doc = pypdfium2.PdfDocument(str(pdf))
    ims = [p.render(scale=WIDTH / p.get_size()[0]).to_pil().convert("RGB") for p in doc]
    ims = [im for im in ims if np.asarray(im.convert("L")).min() < 128]  # 白紙のページ (MuseScore がときどき出す) は除く
    doc.close()
    pdf.unlink()
    return ims


# ---------- 音の一覧と採点 ----------

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


def score(truth, result, flags=()):
    """正解と読み取り結果 (どちらも root) を比べる。{"n": 数, "right": [小節ごとの○×], "wrong": 食い違いのある小節の一覧}。
    flags はアプリが「確かめきれなかった」と印を付けた小節 (読み取り結果の小節番号)。間違っているのに印のない小節を
    silent、合っているのに印のある小節を false_flags と数える。人が確かめなくてよいと言えるのは silent が 0 のとき。"""
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
    flagged = {i for i, j in pairs if str(j + 1) in set(map(str, flags))}
    n["flagged"] = len(set(map(str, flags)))
    n["silent"] = sum(not ok and i not in flagged for i, ok in enumerate(right))
    n["false_flags"] = sum(ok and i in flagged for i, ok in enumerate(right))
    n["truth_notes"] = sum(len(t["ev"]) for t in T)
    return {"n": n, "right": right, "wrong": sorted(wrong, key=lambda w: w.get("measure", 10 ** 6))}


# ---------- 読み取り ----------

HOMR_VERSION = importlib.metadata.version("homr")  # 版が変われば OMR をやり直す
AUDIVERIS_VERSION = "5.11.0"


def cached_run(real, cache):
    """server.subprocess.run の代わり。homr を呼ぶ処理だけ、画像ごとに結果を bench/work/omr/ に取っておき、2回目からは OMR を飛ばす。"""
    def run(cmd, cwd=None, **kw):
        if cmd[0] == server.AUDIVERIS:  # Audiveris も画像ごとに結果を取っておく
            out, img = Path(cmd[4]), Path(cwd) / cmd[-1]
            key = WORK / "audiveris" / f"{zlib.crc32(img.read_bytes()):08x}_{AUDIVERIS_VERSION}"
            if not (key / "p.mxl").exists():
                r = real(cmd, cwd=cwd, **kw)
                if (out / f"{img.stem}.mxl").exists():
                    key.mkdir(parents=True, exist_ok=True)
                    shutil.copy(out / f"{img.stem}.mxl", key / "p.mxl")
                return r
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(key / "p.mxl", out / f"{img.stem}.mxl")
            return types.SimpleNamespace(stderr="", stdout="", returncode=0)
        if "--write-staff-positions" not in cmd:
            return real(cmd, cwd=cwd, **kw)
        d, name = Path(cwd), Path(cmd[-1]).stem
        key = WORK / "omr" / f"{zlib.crc32((d / cmd[-1]).read_bytes()):08x}_{HOMR_VERSION}"
        if not (key / f"{name}.txt").exists():  # 五線の位置 (.txt) は校正の段の切り出しに要る
            r = real(cmd, cwd=cwd, **kw)
            if not (d / f"{name}.musicxml").exists():
                return r
            key.mkdir(parents=True, exist_ok=True)
            for f in (f"{name}.musicxml", f"{name}.txt"):
                if (d / f).exists():
                    shutil.copy(d / f, key / f)
            return r
        for f in key.iterdir():
            shutil.copy(f, d / f.name)
        return types.SimpleNamespace(stderr="", stdout="", returncode=0)
    return run


def read_song(name, cond, stage):
    """画像の並びを、隔離したライブラリの1曲として server.process_page に1ページずつ通し、(score.musicxml の root, 印の付いた小節) を返す。
    server の差し替えは patched() の中で行う。"""
    song = f"{name}_{cond}_{stage}"
    d = WORK / "lib" / song
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    pages = sorted((WORK / "pages" / f"{name}_{cond}").glob("p*.jpg"), key=lambda p: int(p.stem[1:]))
    for p in pages:
        shutil.copy(p, d / p.name)
    (d / "meta.json").write_text(json.dumps({"id": song, "title": song, "pages": len(pages), "measures": 0, "pending": 0,
                                             "step": None, "log": []}))
    for n in range(1, len(pages) + 1):
        try:
            server.process_page(song, n)
        except RuntimeError as e:  # 読み取れなかったページは、そのページの小節がすべて欠けたものとして数える
            print(f"{song} p{n}: {str(e)[:80]}", flush=True)
    if not (d / "score.musicxml").exists():
        return ET.fromstring("<score-partwise><part id='P1'/></score-partwise>"), set()
    return ET.parse(d / "score.musicxml").getroot(), set(server.read_meta(song).get("flags", {}))


class patched:
    """読み取りを試験用にする: library/ には書かない、OMR の結果を使い回す。
    raw は Claude を呼ぶ proof_page を空にする。full は本物 (呼び出し側が許可を確認済み)。"""
    def __init__(self, stage):
        self.stage = stage

    def __enter__(self):
        self.saved = server.LIB, server.proof_page, server.subprocess
        server.LIB = WORK / "lib"
        server.subprocess = types.SimpleNamespace(run=cached_run(subprocess.run, WORK), DEVNULL=subprocess.DEVNULL)
        if self.stage != "full":
            server.proof_page = lambda song, n: None

    def __exit__(self, *a):
        server.LIB, server.proof_page, server.subprocess = self.saved


# ---------- 集計 ----------

KINDS = ("correct", "octave", "semitone", "other_pitch", "duration", "onset", "missing", "extra")


def total(ns):
    t = Counter()
    for n in ns:
        t.update(n)
    return t


def line(label, n):
    notes = n["truth_notes"] or 1
    return (f"{label:<24}{n['right_measures']:>4}/{n['measures']:<4}{100 * n['right_measures'] / max(n['measures'], 1):>5.1f}%  "
            f"{100 * n['correct'] / notes:>5.1f}%  " + " ".join(f"{n[k]:>4}" for k in KINDS[1:])
            + f"   {n['missing_measures']:>3} {n['extra_measures']:>3}   {n['key_wrong']:>3} {n['time_wrong']:>4}"
            + f"   {n['flagged']:>4} {n['silent']:>4} {n['false_flags']:>4}")


def report(runs):
    """runs: {(stage, cond): {piece: score}}。条件ごとに全体と曲ごとの表を出す。
    flag: 確かめきれなかった印の数、silent: 間違っているのに印のない小節、fals: 合っているのに印のある小節。"""
    head = (f"{'':<24}{'measures right':>14}   notes  " + " ".join(f"{k[:4]:>4}" for k in KINDS[1:])
            + "   mis ext   key time   flag sile fals")
    for (stage, cond), res in runs.items():
        print(f"\n[{stage} / {cond}]\n{head}")
        print(line("ALL", total(r["n"] for r in res.values())))
        for piece, r in res.items():
            print(line(piece, r["n"]))
    for stage in dict.fromkeys(s for s, _ in runs):  # 段階ごとに、条件をまとめた dev (改善の判断に使う曲) と test の合計
        print(f"\n[{stage} / all conditions]")
        for split in ("dev", "test"):
            ns = [r["n"] for (s, _), res in runs.items() if s == stage for p, r in res.items()
                  if p.startswith("test_") == (split == "test")]
            if ns:
                print(line(split.upper(), total(ns)))


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
    conds = [a[7:] for a in argv if a.startswith("--cond=")] or list(CONDS)
    label = next((a[8:] for a in argv if a.startswith("--label=")), "")
    pieces = [p.stem for p in sorted(TRUTH.glob("*.musicxml"))
              if (p.stem in only if only else "--test" in argv or not p.stem.startswith("test_"))]
    if not stages or not pieces:
        raise SystemExit(__doc__)
    pages = {p: engrave(p) for p in pieces}
    if "full" in stages:
        n = sum(pages[p][c] for p in pieces for c in conds)
        print(f"full は Claude を使う本番どおりの処理です。{n}ページ処理します (1ページ数分、Claude を10回前後呼ぶ)")
        if "--yes-full" not in argv:
            raise SystemExit("実行するなら --yes-full を付ける")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "*.py"], capture_output=True, text=True).stdout.strip())
    runs = {}
    for stage in stages:
        def one(job):
            p, cond = job
            t0 = time.time()
            root, flags = read_song(p, cond, stage)
            print(f"{stage} {cond} {p}: {pages[p][cond]}ページ {time.time() - t0:.0f}秒", flush=True)
            return score(ET.parse(TRUTH / f"{p}.musicxml").getroot(), root, flags)
        jobs = [(p, c) for c in conds for p in pieces]
        # 曲ごとに別のフォルダで動くので並べて走らせてよい。Claude の呼び出しは1曲の中でも6本まで並ぶ
        with patched(stage), ThreadPoolExecutor(int(os.environ.get("BENCH_JOBS", 3))) as pool:
            for (p, c), r in zip(jobs, pool.map(one, jobs)):
                runs.setdefault((stage, c), {})[p] = r
        name = f"results_{stage}{'_' + label if label else ''}.json"
        (WORK / name).write_text(json.dumps(
            {"commit": commit, "dirty": dirty, "label": label,
             **{c: {p: {"counts": r["n"], "right": r["right"], "wrong_measures": r["wrong"]} for p, r in runs[stage, c].items()}
                for c in conds}}, ensure_ascii=False, indent=1))
        t = total(r["n"] for c in conds for r in runs[stage, c].values())
        with open(WORK / "history.jsonl", "a") as f:  # 試した策ごとの数字を1行ずつ残す
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "stage": stage, "label": label, "commit": commit,
                                "dirty": dirty, "conds": conds, "pieces": pieces,
                                **{k: t[k] for k in ("measures", "right_measures", "truth_notes", "correct", "flagged",
                                                     "silent", "false_flags")}}, ensure_ascii=False) + "\n")
    report(runs)


if __name__ == "__main__":
    main(sys.argv[1:])
