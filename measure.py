"""写真の音符が五線のどの高さにあるかを測り、データの音の高さと突き合わせる。

homr は音符の位置を図形から測っているのに、音の高さには使わない (高さは別のモデルが決める)。
この測定は独立した証拠になるので、オクターブ違いの自動修正に使う。
"""
from collections import defaultdict

import numpy as np

CLEF_REF = {"G": 4 * 7 + 4, "F": 3 * 7 + 3, "C": 4 * 7}  # 音部記号の基準音 (ソ4, ファ3, ド4) の 通し番号=オクターブ*7+音名


def read_notes(root):
    """段ごと・手ごと (1=右手, 2=左手) の音の一覧。pos は一番下の線を1として、線と間を1ずつ数えた位置。"""
    systems, bottom = [], {1: 30, 2: 18}  # 一番下の線の音 (ト音記号ならミ4、へ音記号ならソ2)
    for m in root.find("part").findall("measure"):
        brk = m.find("print")
        if not systems or (brk is not None and brk.get("new-system") == "yes"):
            systems.append({1: [], 2: []})
        clefs, found, cursor, last = {1: [], 2: []}, [], 0, 0
        for el in m:
            if el.tag == "attributes":
                for c in el.findall("clef"):
                    ref = CLEF_REF[c.findtext("sign")] + 7 * int(c.findtext("clef-octave-change", "0"))
                    clefs[int(c.get("number", "1"))].append((cursor, ref - 2 * (int(c.findtext("line", "2")) - 1)))
            elif el.tag in ("backup", "forward"):
                cursor += int(el.findtext("duration")) * (-1 if el.tag == "backup" else 1)
            elif el.tag == "note" and el.find("grace") is None:
                chord = el.find("chord") is not None
                onset = last if chord else cursor
                if not chord:
                    last, cursor = cursor, cursor + int(el.findtext("duration", "0"))
                p = el.find("pitch")
                if p is not None:
                    found.append((el, int(el.findtext("staff", "1")), onset,
                                  int(p.findtext("octave")) * 7 + "CDEFGAB".index(p.findtext("step"))))
        order = {1: 0, 2: 0}
        for el, st, onset, dia in found:
            b = bottom[st]
            for t, v in sorted(clefs[st]):
                if t <= onset:
                    b = v
            order[st] += 1
            systems[-1][st].append({"el": el, "m": m, "pos": dia - b + 1, "bot": b, "onset": onset, "order": order[st]})
        for st in (1, 2):
            if clefs[st]:
                bottom[st] = max(clefs[st])[1]
    return systems


def data_columns(notes):
    """同じ時刻に鳴る音 (声部をまとめて) を1列とする。"""
    d = defaultdict(list)
    for n in notes:
        d[(n["m"], n["onset"])].append(n)
    return [{"set": frozenset(n["pos"] for n in ns), "notes": ns} for ns in d.values()]


def detected(geo):
    """homr が見つけた符頭を、段ごと・手ごとの位置 [(x, 位置, 五線の線間)] にする。
    homr 自身が出す位置は五線の線の読み取りが数px揺れて半音ずれることがあるので、五線全体から線間を決め直す。"""
    out = []
    for system in geo:
        best = {}
        for si, staff in enumerate(system):
            g = np.array(staff["grid"])
            xs, Y = g[:, 0], g[:, 1:]
            for hi, H in enumerate([Y[:, :5], Y[:, 5:]] if Y.shape[1] == 10 else [Y]):  # 10本は大譜表が1つにまとまったもの
                sp = np.median(np.diff(H, axis=1), axis=1)
                s = np.array([np.median(sp[abs(xs - x) < 250]) for x in xs])
                top = np.median(H - np.arange(5) * s[:, None], axis=1)
                top = np.array([np.median(top[abs(xs - x) < 60]) for x in xs])
                for x, y in staff["notes"]:
                    i = int(np.argmin(abs(xs - x)))
                    pos = round(2 * (top[i] + 4 * s[i] - y) / s[i]) + 1
                    dist = abs(y - top[i] - 2 * s[i]) / s[i]  # 五線の中心からの距離。2つの五線に入った音は近いほうにする
                    if (round(x), round(y)) not in best or dist < best[(round(x), round(y))][0]:
                        best[(round(x), round(y))] = (dist, si, hi, x, pos, s[i])
        slots = sorted({v[1:3] for v in best.values()},
                       key=lambda k: -sum(v[1:3] == k for v in best.values()))[:2]
        slots.sort()
        out.append({st: sorted((v[3], v[4], v[5]) for v in best.values() if v[1:3] == k)
                    for st, k in enumerate(slots, 1)})
    return out


def det_columns(notes):
    """x が近い符頭を1列とする (2度の音は左右にずれるので、符頭の幅より少し広く取る)。"""
    cols = []
    for x, pos, s in notes:
        if cols and x - cols[-1][-1][0] <= 1.3 * s:
            cols[-1].append((x, pos, s))
        else:
            cols.append([(x, pos, s)])
    return [{"set": frozenset(p for _, p, _ in c)} for c in cols]


def align(A, B):
    """列の並びどうしを Needleman-Wunsch で合わせる。費用は音の集合の食い違い。ずれた分は列ごと飛ばす。"""
    n, m = len(A), len(B)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + len(A[i - 1]["set"])
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + len(B[j - 1]["set"])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j - 1] + 0.9 * len(A[i - 1]["set"] ^ B[j - 1]["set"]),
                          D[i - 1][j] + len(A[i - 1]["set"]), D[i][j - 1] + len(B[j - 1]["set"]))
    i, j, pairs = n, m, []
    while i and j:
        if D[i][j] == D[i - 1][j - 1] + 0.9 * len(A[i - 1]["set"] ^ B[j - 1]["set"]):
            pairs.append((A[i - 1], B[j - 1]))
            i, j = i - 1, j - 1
        elif D[i][j] == D[i - 1][j] + len(A[i - 1]["set"]):
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def matched(root, geo):
    """(データの列, 測定の列, 手) の組。段の数が写真と合わないページは何も返さない。
    段の中の音の数が2割以上違う手も外す (列がずれて当てにならない)。"""
    data, det = read_notes(root), detected(geo)
    if len(data) != len(det):
        return []
    return [(a, b, st) for s, d in zip(data, det) for st in (1, 2)
            if st in d and abs(len(s[st]) - len(d[st])) < 0.2 * max(len(s[st]), len(d[st]))
            for a, b in align(data_columns(s[st]), det_columns(d[st]))]


def usable(a, b, far):
    """全音符、2度を含む列、加線が3本以上の音 (far の位置) は、測定が当てにならないので使わない。"""
    return (not any(n["el"].findtext("type") == "whole" for n in a["notes"])
            and not any(q - p == 1 for ps in (a["set"], b["set"]) for p in ps for q in ps)
            and not any(p <= -5 or p >= 15 for p in far))


def fix_octaves(root, geo):
    """列の音の数が測定と同じで、全部がちょうど1オクターブ (7) ずれているときだけ、測定の高さに直す。
    データ側が加線3本以上でも直す (オクターブの読み違いはその外側へ出やすい)。測定側は当てにならない範囲を除く。
    直した (小節番号, 手) を返す。"""
    # defer: 8va の記号は見ていない。homr が 8va を読んで実音にした音は、書かれた高さへ戻してしまう (試した14ページで1音)。
    # 校正が使えれば直るが、8va の多い曲で校正なしの結果が悪くなったら、記号を読むまでその段では直さない
    fixed = set()
    for a, b, st in matched(root, geo):
        if len(a["set"]) == len(a["notes"]) == len(b["set"]) and usable(a, b, b["set"]):
            diff = {q - p for p, q in zip(sorted(a["set"]), sorted(b["set"]))}
            if len(diff) == 1 and abs(d := diff.pop()) == 7:
                for n in a["notes"]:
                    pitch = n["el"].find("pitch/octave")
                    pitch.text = str(int(pitch.text) + d // 7)
                fixed.add((a["notes"][0]["m"].get("number"), st))
    return sorted(fixed, key=lambda f: (int(f[0]), f[1]))

