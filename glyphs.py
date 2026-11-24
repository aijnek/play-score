"""楽譜ソフトが書き出した PDF から、符頭のグリフ (SMuFL) と五線・小節線・弧の座標を読み、データの音符と突き合わせる。

画像からの読み取り (OMR も Claude も) は、加線の多い音や2つの声部が重なる音を同じように読み違える。PDF に文字として
入っている符頭は座標が正確なので、別の根拠として使える。読めるのは SMuFL の字体 (MuseScore など) か、Finale の Maestro と同じ並びの字体 (Kousaku など) で書き出された PDF で、
スキャンした PDF や他の字体の PDF では小節が取れず、検査を行わない。確かめるのは符頭の種類と高さ、タイの始まる音、臨時記号。音の長さ (旗や連桁の数) は見ない。
# defer: 8va の記号は見ない (データが実音で、符頭が1オクターブ下に描かれていると食い違いになる)。8va のある楽譜で誤った印が増えたら記号を読む
"""
import ctypes
from collections import Counter, defaultdict
from fractions import Fraction

import pypdfium2
import pypdfium2.raw as raw

HEADS = {0xE0A2: "w", 0xE0A3: "h", 0xE0A4: "q"}  # 全音符・2分音符・黒い符頭
ACCS = {0xE260: -1, 0xE261: 0, 0xE262: 1, 0xE263: 2, 0xE264: -2}  # フラット・ナチュラル・シャープ・ダブルシャープ・ダブルフラット
SHARPS, FLATS = "FCGDAEB", "BEADGCF"
# Finale の Maestro と同じ並びの字体 (Kousaku など) の文字 → SMuFL。pdfium が返す Unicode で引く。
# 符頭の文字 (œ) が何度も出る字体にだけ当てる (歌詞やコード名の b や n を臨時記号と取り違えないため)
MAESTRO = {0x153: 0xE0A4, 0x2D9: 0xE0A3, ord("w"): 0xE0A2, ord("#"): 0xE262, ord("b"): 0xE260, ord("n"): 0xE261,
           0xBA: 0xE264, 0x2039: 0xE263, 0xDC: 0xE263,
           0x152: 0xE4E5, 0x2030: 0xE4E6, 0x2248: 0xE4E7, 0xAE: 0xE4E8, 0xD3: 0xE4E4, 0x2211: 0xE4E3}  # 休符
STEPS = "CDEFGAB"
SIGN = {"G": 4 * 7 + 4, "F": 3 * 7 + 3, "C": 4 * 7}  # 音部記号が示す音 (ト4・ヘ3・ハ4) の全音階上の番号


def pdf_measures(page):
    """PDF の1ページを、大譜表の小節ごとの {"top", "low": 上下の五線の一番下の線の y, "half": 線と間の間隔, "heads": [(種類, y)]}
    の並び (読む順) にする。グリフが読めない PDF (スキャンなど) や大譜表でないページは []。"""
    rows, vert, arcs, short = defaultdict(float), [], [], []
    for o in page.get_objects(filter=[raw.FPDF_PAGEOBJ_PATH], max_depth=3):
        l, b, r, t = o.get_bounds()
        seg = [raw.FPDFPath_GetPathSegment(o.raw, i) for i in range(raw.FPDFPath_CountSegments(o.raw))]
        kinds = [raw.FPDFPathSegment_GetType(g) for g in seg]
        if kinds[1:4] == [raw.FPDF_SEGMENT_BEZIERTO] * 3 and kinds.count(raw.FPDF_SEGMENT_BEZIERTO) == 6 and r - l > t - b:
            # タイやスラーの弧: 始点から終点へ行って戻る2本の曲線 (あいだを直線でつなぐソフトもある)。0番と3番の点が両端
            ends, m = [], o.get_matrix().get()
            for g in (seg[0], seg[3]):
                x, y = ctypes.c_float(), ctypes.c_float()
                raw.FPDFPathSegment_GetPoint(g, x, y)
                ends.append((m[0] * x.value + m[2] * y.value + m[4], m[1] * x.value + m[3] * y.value + m[5]))
            arcs.append(sorted(ends))
            continue
        if t - b < 2 and r - l > 3:  # 五線と加線 (連桁はもっと太い)
            short.append(((b + t) / 2, l, r))
            if r - l > 30:  # 加線は短い
                rows[round((b + t) / 2, 1)] += r - l
        elif r - l < 3 and t - b > 20:
            vert.append(((l + r) / 2, b, t))
    if not rows:
        return []
    merged = {}  # 途中で切れた段 (コーダの前など) は、同じ線が少しずれた高さで2本に分かれている
    for y in sorted(rows):
        near = next((k for k in merged if abs(k - y) < 1), y)
        merged[near] = merged.get(near, 0) + rows[y]
    ys = sorted(y for y, w in merged.items() if w > max(merged.values()) / 10)  # 五線 (加線は短いので入っていない。最後の段は短いことがある)
    staves, i = [], 0
    while i + 4 < len(ys):  # 等間隔に並ぶ5本を五線とする
        gaps = [b - a for a, b in zip(ys[i:i + 5], ys[i + 1:i + 5])]
        if max(gaps) < 1.3 * min(gaps):
            staves.append(ys[i:i + 5])
            i += 5
        else:
            i += 1
    if not staves or len(staves) % 2:
        return []
    doc, tp, known, texts = page.pdf, page.get_textpage(), defaultdict(dict), []

    def ink(o):  # そのグリフだけを描いたときの濃さ (黒い符頭と白い符頭を見分ける)
        bm = raw.FPDFTextObj_GetRenderedBitmap(doc.raw, page.raw, o.raw, 8)
        w, h, stride = raw.FPDFBitmap_GetWidth(bm), raw.FPDFBitmap_GetHeight(bm), raw.FPDFBitmap_GetStride(bm)
        buf = ctypes.string_at(raw.FPDFBitmap_GetBuffer(bm), stride * h)
        raw.FPDFBitmap_Destroy(bm)
        return sum(buf[3::4]) / max(w * h, 1)  # アルファ値の平均
    found, fonts = [], Counter()
    for o in page.get_objects(filter=[raw.FPDF_PAGEOBJ_TEXT], max_depth=3):
        buf, name, scale = ctypes.create_string_buffer(16), ctypes.create_string_buffer(128), ctypes.c_float()
        k = raw.FPDFTextObj_GetText(o.raw, tp.raw, ctypes.cast(buf, ctypes.POINTER(ctypes.c_ushort)), 8)
        code = buf.raw[:max(k - 2, 0)].decode("utf-16-le", errors="replace")[:1]  # k は終端を含むバイト数
        raw.FPDFFont_GetBaseFontName(raw.FPDFTextObj_GetFont(o.raw), name, 128)
        raw.FPDFTextObj_GetFontSize(o.raw, scale)
        found.append((code and ord(code), name.value, scale.value, o))
        fonts[name.value] += code == "œ"
    for code, font, scale, o in found:
        if fonts[font] >= 5:
            code = code and MAESTRO.get(code, code)
        l, b, r, t = o.get_bounds()
        m = o.get_matrix().get()
        size = (font, round(r - l, 1), round(t - b, 1))
        if code and code not in known[size]:
            known[size][code] = ink(o)
        texts.append((code, size, round(m[4], 1), round(m[5], 1), m[0] * scale, o))
    heads, accs, rests = set(), set(), []
    for c, size, x, y, a, o in texts:
        if not c and any(k in HEADS or k in ACCS for k in known[size]):
            # pdfium は、同じ文字がすぐ近くに重なると (和音の隣の符頭など) 文字を返さない。同じ大きさで濃さの近い既知のグリフとみなす
            dark = ink(o)
            c = min(known[size], key=lambda k: abs(known[size][k] - dark))
        if c in HEADS:
            heads.add((HEADS[c], x, y, a))
        if c in ACCS:
            accs.add((x, y, ACCS[c]))
        if c and 0xE4E0 <= c <= 0xE4FF:  # 休符
            rests.append((x, y))
    full = max((a for *_, a in heads), default=0)
    heads = {h[:3] for h in heads if h[3] > .8 * full}  # 小さい符頭 (装飾音符) は除く。同じ場所に2度描かれた符頭 (声部の共有) は1つ
    systems, out = [], []
    for low, top in reversed(list(zip(staves[::2], staves[1::2]))):  # y は下から上へ増える。段は上から
        half = (top[4] - top[0]) / 8
        bars = sorted(x for x, b, t in vert if b < low[4] + half and t > top[4] - half)  # 小節線は上の五線から下の五線の上端まで通る
        bars = [x for i, x in enumerate(bars) if i == 0 or x - bars[i - 1] > 8 * half]  # 複縦線は1本と数える
        systems.append((low, top, half, bars, []))
    for k, x, y in heads:
        # 符頭は、いちばん近い段のもの。ただし加線の多い音は隣の段に近づくので、五線の側へ続く加線が上下どちらにあるかを先に見る
        half = systems[0][2]
        over = [s for s in systems if s[0][0] > y]   # 上にある段 (いちばん近いのは最後)
        under = [s for s in systems if s[1][4] < y]
        side = []
        if over and under and len(over) + len(under) == len(systems):  # 段と段のあいだの音
            def reaches(edge, step):  # 五線の端の線から符頭まで、加線が途切れずに続いているか
                want = [edge + step * j for j in range(1, 20) if (y - edge - step * j) * step > .5 * half * abs(step)]
                return all(any(abs(ly - w) < .3 * half and l < x + 1.2 * half < r for ly, l, r in short) for w in want)
            a, b = reaches(over[-1][0][0], -2 * half), reaches(under[0][1][4], 2 * half)
            if a != b:
                side = over[-1:] if a else under[:1]
        min(side or systems, key=lambda s: max(s[0][0] - y, y - s[1][4], 0))[4].append((k, x, y))
    for low, top, half, bars, mine in systems:
        cand = []  # (弧の端と符頭の高さの差, 弧の番号, 符頭の x, y)
        for i, ((x0, y0), (x1, y1)) in enumerate(arcs):
            # 弧の端は符頭のすぐ横か上下にある。両端が同じ高さの符頭に付いていればタイ、段の終わりへ抜ける平らな弧もタイ
            first = [(abs(y - y0), x, y) for k, x, y in mine if x - half < x0 < x + 6 * half and abs(y - y0) < 2 * half]
            last = [y for k, x, y in mine if x - 9 * half < x1 < x + 3 * half and abs(y - y1) < 2 * half]
            out_of_system = x1 > bars[-1] - half and abs(y0 - y1) < half
            for dy, hx, hy in first:
                if not (out_of_system or any(abs(hy - y2) < .3 * half for y2 in last)):
                    continue
                # 同じ高さの音で始まって終わるスラー (あいだに別の音がある) はタイではない。2つの声部で片方が伸ばすタイも
                # あいだに音があるが、タイの端は符頭のすぐ脇 (高さの差が線1本半以内) に付く
                # defer: 1.6 は MuseScore の PDF で測った値 (タイ 1.4〜1.5、スラー 1.8)。他のソフトの PDF でタイの見落としが出たら測り直す
                # (2度の和音で隣にずらして描かれた符頭は、あいだの音に数えない)
                between = any(hx + 3 * half < x < x1 - 3 * half and abs(y - hy) < 8 * half for k, x, y in mine)
                if not (between and dy > 1.6 * half):
                    cand.append((dy, i, hx, hy))
        ties, used = [], set()  # タイの始まりの符頭 (x, y)。和音では弧が並ぶので、近い組から1つの符頭に1本ずつ当てる
        for dy, i, hx, hy in sorted(cand):
            if i not in used and (hx, hy) not in ties:
                used.add(i)
                ties.append((hx, hy))
        for a, b in zip(bars, bars[1:]):
            inside = [(k, y) for k, x, y in mine if a - half < x < b - half]
            # 音も休符もない区切り (段の頭から反復記号までなど) は小節ではない
            if inside or any(a < x < b and low[0] - 4 * half < y < top[4] + 4 * half for x, y in rests):
                # 符頭ごとに、すぐ左の同じ高さに印刷された臨時記号 (なければ None)。調号の記号を拾っても、同じ高さなら効き目は同じ
                marks = [next((v for ax, ay, v in sorted(accs, reverse=True)
                               if x - 10 * half < ax < x and abs(ay - y) < .3 * half
                               and not any(ax < hx < x and abs(hy - y) < .3 * half for _, hx, hy in mine)), None)
                         for k, x, y in mine if a - half < x < b - half]
                # 途中で切れた段では、小節のある側の線の高さを使う
                line = lambda y0: next((y for y, l, r in short if abs(y - y0) < 1 and r - l > 30 and l - 1 < (a + b) / 2 < r + 1), y0)
                out.append({"top": line(top[0]), "low": line(low[0]), "half": half, "x": [a, b], "heads": inside,
                            "ties": [y for x, y in ties if a - half < x < b - half],
                            "xs": [x for k, x, y in mine if a - half < x < b - half], "marks": marks})
    return out


def data_heads(part):
    """データの小節ごとに {"heads": [(種類, 五線 "1"|"2", 五線の一番下の線から数えた位置)], "ties": [タイの始まる音の (五線, 位置)]}。
    装飾音符と休符は除き、同じ五線・同じ時刻・同じ位置・同じ種類の符頭 (2つの声部が共有する) は1つにする。"""
    out, clef, fifths = [], {"1": (SIGN["G"], 2), "2": (SIGN["F"], 4)}, 0
    for m in part.findall("measure"):
        notes, changes, pos, last = [], [], 0, 0
        for e in m:
            if e.tag == "attributes":
                fifths = int(e.findtext("key/fifths") or fifths)
                for c in e.findall("clef"):
                    changes.append((pos, c.get("number") or "1",
                                    (SIGN[c.findtext("sign")] + 7 * int(c.findtext("clef-octave-change") or 0),
                                     int(c.findtext("line") or {"G": 2, "F": 4, "C": 3}[c.findtext("sign")]))))
            d = int(e.findtext("duration") or 0)
            if e.tag == "backup":
                pos -= d
            elif e.tag == "forward":
                pos += d
            elif e.tag == "note" and e.find("grace") is None:
                if e.find("chord") is None:
                    last, pos = pos, pos + d
                if e.find("pitch") is not None:
                    notes.append((last, e))
        for c in [c for c in changes if c[0] <= 0]:  # 小節の頭の音部記号
            clef[c[1]] = c[2]
            changes.remove(c)
        seen, start, sounds, names = {}, dict(clef), set(), {}
        for t, e in sorted(notes, key=lambda x: x[0]):  # 小節の途中で音部記号が替わるので、時刻の順に当てる
            for c in [c for c in changes if c[0] <= t]:
                clef[c[1]] = c[2]
                changes.remove(c)
            p, staff = e.find("pitch"), e.findtext("staff") or "1"
            sign, line = clef.get(staff, clef["1"])
            at = int(p.findtext("octave")) * 7 + STEPS.index(p.findtext("step")) - sign + 2 * (line - 1)
            kind = {"whole": "w", "half": "h"}.get(e.findtext("type"), "q")
            tie = e.find("tie[@type='start']") is not None or e.find("notations/tied[@type='start']") is not None
            key = (kind, staff, at, t, len(e.findall("dot")))  # 付点の有無が違えば符頭は並べて描かれる
            seen[key] = seen.get(key, False) or tie
            held = e.find("tie[@type='stop']") is not None or e.find("notations/tied[@type='stop']") is not None
            sounds.add((staff, at, t, int(float(p.findtext("alter") or 0)), held))
            names[staff, at] = p.findtext("step")
        for c in changes:
            clef[c[1]] = c[2]
        out.append({"heads": [k[:3] for k in seen], "ties": [k[1:3] for k, tie in seen.items() if tie], "clef": start,
                    "key": fifths, "names": names, "sounds": [(s, a, alter, held) for s, a, _, alter, held in sounds]})
    return out


def where(pm, y):
    """PDF の高さ y を (五線 "1"|"2", 五線の一番下の線から数えた位置) にする。上下の五線のあいだで分ける。"""
    middle = (pm["top"] + pm["low"] + 8 * pm["half"]) / 2
    return ("1", round((y - pm["top"]) / pm["half"])) if y > middle else ("2", round((y - pm["low"]) / pm["half"]))


def letter(dm, staff, at):
    """五線上の位置を、小節の頭の音部記号で (音名, オクターブ) にする。"""
    sign, line = dm["clef"].get(staff, dm["clef"]["1"])
    n = at + sign - 2 * (line - 1)
    return STEPS[n % 7], n // 7


def alters(pm, dm):
    """臨時記号の食い違い: (印刷にあってデータにない, データにあって印刷にない) を Counter((五線, 位置, 半音の変化)) で返す。
    印刷の側は、調号と、小節の中でそれまでに同じ高さに付いた臨時記号から決める。前の小節からタイで続く音は記号が
    印刷されないので、データでタイの終わりになっている音のある位置は、印刷の側を問わない。数ではなく有無で比べる
    (同じ高さの符頭が並べて描かれることがある)。"""
    key = {s: 1 for s in SHARPS[:max(dm["key"], 0)]} | {s: -1 for s in FLATS[:max(-dm["key"], 0)]}
    y = lambda s, a: (pm["top"] if s == "1" else pm["low"]) + a * pm["half"]
    spots = {(s, a) for s, a, *_ in dm["sounds"]}
    state, have = {}, set()
    for x, (k, hy), mark in sorted(zip(pm["xs"], pm["heads"], pm["marks"])):
        # 符頭の位置は、同じ高さにあるデータの音の五線と位置で呼ぶ (上の五線の音が下の五線の近くに描かれることがある)
        staff, at = next((p for p in spots if abs(y(*p) - hy) < .3 * pm["half"]), where(pm, hy))
        if mark is not None:
            state[staff, at] = mark
        have.add((staff, at, state.get((staff, at), key.get(dm["names"].get((staff, at)) or letter(dm, staff, at)[0], 0))))
    held = {(s, a) for s, a, alter, h in dm["sounds"] if h}
    want = {(s, a, alter) for s, a, alter, h in dm["sounds"]}
    return (Counter(k for k in have - want if k[:2] not in held),
            Counter(k for k in want - have if k[:2] not in held))


def mismatch(pm, dm, ties=True):
    """PDF の小節 pm とデータの小節 dm の、符頭 (種類と高さ)・タイの始まる音・臨時記号のどれかが食い違うか。"""
    y = lambda s, a: (pm["top"] if s == "1" else pm["low"]) + a * pm["half"]
    pairs = [(sorted((k, y(s, a)) for k, s, a in dm["heads"]), sorted(map(tuple, pm["heads"])))]
    if ties:
        pairs.append((sorted(("", y(s, a)) for s, a in dm["ties"]), sorted(("", v) for v in pm["ties"])))
    return (any(len(want) != len(have) or any(a != b or abs(u - v) > .3 * pm["half"] for (a, u), (b, v) in zip(want, have))
                for want, have in pairs) or (ties and any(alters(pm, dm))))


def describe(pm, dm):
    """PDF の小節 pm とデータの小節 dm の符頭とタイの違いを、英語の文にする (Claude に直させるときの手がかり)。
    音名は小節の頭の音部記号で数え、臨時記号は付けない。符頭とタイが合っているときだけ、臨時記号の違いを書く。"""
    name = lambda staff, at: "%s%d" % letter(dm, staff, at)
    head = {"w": "whole-note head", "h": "half-note (hollow) head", "q": "filled head"}
    out = []
    for label, want, have in (("notehead", Counter(dm["heads"]), Counter((k, *where(pm, y)) for k, y in pm["heads"])),
                              ("tie starting on", Counter(("", *t) for t in dm["ties"]), Counter(("", *where(pm, y)) for y in pm["ties"]))):
        for staff, hand in (("1", "upper staff"), ("2", "lower staff")):
            for text, items in (("printed but not in the data", have - want), ("in the data but not printed", want - have)):
                names = [f"{name(s, a)}{' (' + head[k] + ')' if k else ''}" for k, s, a in sorted(items.elements()) if s == staff]
                if names:
                    out.append(f"{hand}, {label} {text}: {', '.join(names)}")
    if not out:  # 符頭とタイは合っている: 臨時記号の違い
        sign = {-2: "double flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double sharp"}
        for text, items in zip(("printed as", "in the data as"), alters(pm, dm)):
            names = [f"{name(s, a)} {sign[v]}" for s, a, v in sorted(items.elements())]
            if names:
                out.append(f"accidentals (key signature and earlier accidentals in the measure applied), {text}: {', '.join(names)}")
    return "; ".join(out)


def read_pdf(path, number):
    """PDF の number ページ目 (1から) の小節の並び。読めなければ []。"""
    try:
        doc = pypdfium2.PdfDocument(str(path))
        return pdf_measures(doc[number - 1])
    except Exception:
        return []


def check(measures, part):
    """PDF の小節の並び (pdf_measures の結果) とデータのページ (part) を比べ、食い違う小節を {小節番号: 違いの説明} で返す。
    小節の数が合わなければ対応が取れないので None。"""
    data = part.findall("measure")
    if not measures or len(measures) != len(data):
        return None
    return {m.get("number"): describe(pm, dm) for m, pm, dm in zip(data, measures, data_heads(part)) if mismatch(pm, dm)}
