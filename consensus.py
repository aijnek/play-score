"""2つの OMR (homr と Audiveris) の読み取りを小節ごとに突き合わせる。

2つは別々の方法で読むので、鳴る音が一致した小節はほぼ正しい (試験台の dev で一致した小節の96%が正解)。
食い違った小節だけを、拍子に合うかどうかや Claude で決める。試験台 (bench.py) の採点も同じ「鳴る音」を使う。
"""
import math
from collections import Counter
from fractions import Fraction

import numpy as np

STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def q(x):
    """music21 が書き出す連符は 827/5040 のように割り切れない長さになる。1/48 拍に丸めて、連符を正しい長さで比べる。
    # defer: 1/48 拍より短い音 (32分の3連より細かい) は区別できない。そういう曲を入れる時に分解能を上げる"""
    return Fraction(round(x * 48), 48)


def events(root):
    """最初のパートを小節ごとの {"ev": [(段, 拍位置, 高さ, 長さ)], "key", "time"} にする。位置と長さは4分音符を1とする分数。
    タイでつながった音は小節をまたいでも1つにまとめ、装飾音符と休符は数えない。声部は区別しない。
    同じ手・同じ時刻・同じ高さの音は1つにまとめる。"""
    out, div, key, time, open_ties = [], 1, 0, ("4", "4"), {}
    part = root.find("part")
    for m in part.findall("measure") if part is not None else []:
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
    for m in out:  # 同じ手で同時に同じ鍵を押す音 (2つの声部が符頭を共有するなど) は1つしか鳴らないので、長いほうだけ残す
        longest = {}
        for e in m["ev"]:
            k = (e[0], q(e[1]), e[2])
            longest[k] = max(longest.get(k, 0), q(e[3]))
        m["ev"] = [(*k, d) for k, d in longest.items()]
    return out


def align(T, R):
    """小節の並びどうしを Needleman-Wunsch で合わせる (OMR は小節を落とす・割る・足すことがある)。
    費用は音の食い違いの数。(Tの添字, Rの添字) の組を返す。"""
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


def set_divisions(part, k):
    """パートの分解能を k 倍にする (長さはすべて k 倍)。"""
    for e in part.iter():
        if e.tag in ("divisions", "duration"):
            e.text = str(int(e.text) * k)


def common_divisions(a, b):
    """2つのパートの分解能を、両方で割り切れる1つの値にそろえる。途中で分解能が変わるパートは扱わない (False)。"""
    da = {int(d.text) for d in a.iter("divisions")}
    db = {int(d.text) for d in b.iter("divisions")}
    if len(da) != 1 or len(db) != 1:
        return False
    da, db = da.pop(), db.pop()
    L = math.lcm(da, db)
    set_divisions(a, L // da)
    set_divisions(b, L // db)
    return True


def fills(measure, div, time):
    """小節の各声部 (段・声部ごと) の長さが拍子どおりか。"""
    want = Fraction(4 * div * int(time[0]), int(time[1]))
    length = Counter()
    for n in measure.findall("note"):
        if n.find("chord") is None and n.find("grace") is None:
            length[(n.findtext("staff") or "1", n.findtext("voice") or "1")] += int(n.findtext("duration") or 0)
    return bool(length) and all(v == want for v in length.values())


def transplant(dst, src):
    """dst の小節の音を src の小節の音に入れ替える。小節頭の拍子・調・音部記号や小節線は dst のものを残す。
    分解能はそろえてあること。"""
    kids, skids = list(dst), list(src)
    first = next((i for i, e in enumerate(kids) if e.tag == "note"), len(kids))
    sfirst = next((i for i, e in enumerate(skids) if e.tag == "note"), len(skids))
    lead = [e for e in kids[:first] if e.tag not in ("backup", "forward", "barline")]  # 小節頭の拍子・調・音部記号など
    body = [e for e in skids[sfirst:]  # 音と、小節途中の音部記号は src の並びのまま
            if e.tag in ("note", "backup", "forward") or (e.tag == "attributes" and e.find("clef") is not None)]
    dst[:] = lead + body + [e for e in kids if e.tag == "barline"]


def merge(homr, aud):
    """homr のページ (root) を土台に、Audiveris (root) と小節ごとに突き合わせる。homr を書き換え、
    {"agree": [小節の並び順], "replaced": [...], "disagree": [...], "alt": {並び順: Audiveris の小節}} を返す。
    一致しない小節は、homr が拍子に合わず Audiveris が合うときだけ Audiveris の音に入れ替える。
    それ以外の食い違いは alt に Audiveris の小節 (分解能はそろえ済み) を残し、どちらを採るかは後で決める。"""
    hp, ap = homr.find("part"), aud.find("part")
    res = {"agree": [], "replaced": [], "disagree": [], "alt": {}}
    if hp is None or ap is None or not common_divisions(hp, ap):
        return res
    H, A = events(homr), events(aud)
    hm, am = hp.findall("measure"), ap.findall("measure")
    div = int(next(hp.iter("divisions")).text)
    for i, j in align(H, A):
        if Counter(H[i]["ev"]) == Counter(A[j]["ev"]):
            res["agree"].append(i)
        elif not fills(hm[i], div, H[i]["time"]) and fills(am[j], div, H[i]["time"]):
            transplant(hm[i], am[j])
            res["replaced"].append(i)
        else:
            res["disagree"].append(i)
            res["alt"][i] = am[j]
    return res
