"""読み取った楽譜データを五線譜に描き直し、写真の段と上下に並べた「見比べ用」の画像を作る。

データが元の楽譜と合っているかは、データを鳴らしても分からない (データどおりに鳴るだけ)。
描き直した譜面を写真と突き合わせれば、食い違いが目に見える。
"""
import io
import xml.etree.ElementTree as ET
from pathlib import Path

import resvg_py
import verovio
from PIL import Image, ImageDraw, ImageOps

SHARPS, FLATS = "FCGDAEB", "BEADGCF"
NAMES = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}
WIDTH = 1800


def show_accidentals(root):
    """鳴る高さ (alter) から、譜面に書かれるはずの臨時記号を付ける。Verovio は書かれた記号しか描かないため。"""
    key = {}
    for m in root.iter("measure"):
        fifths = m.findtext("attributes/key/fifths")
        if fifths is not None:
            f = int(fifths)
            key = {s: 1 for s in SHARPS[:max(f, 0)]} | {s: -1 for s in FLATS[:max(-f, 0)]}
        state = {}  # 小節内で、その音にいま効いている変化
        for n in m.findall("note"):
            p = n.find("pitch")
            if p is None:
                continue
            for a in n.findall("accidental"):
                n.remove(a)
            k = (n.findtext("staff"), p.findtext("step"), p.findtext("octave"))
            alter = int(float(p.findtext("alter") or 0))
            if alter != state.get(k, key.get(k[1], 0)) and n.find("tie[@type='stop']") is None:
                after = max(i for i, c in enumerate(n) if c.tag in ("duration", "voice", "type", "dot"))
                acc = ET.Element("accidental")
                acc.text = NAMES.get(alter, "natural")
                n.insert(after + 1, acc)
            state[k] = alter


def tidy(measure):
    """小節を「段ごと・声部ごとに、頭から順に」書き直す。音の出る位置は変えない。
    homr は両手や声部を細かく行き来して書き出すので、そのままだと Verovio の描画が乱れ、
    実際には合っているのに違って見える (和音の中で動く声部、小節途中の音部記号で顕著)。"""
    lead, trail, clefs, voices = [], [], [], {}
    pos = prev = 0
    for e in list(measure):
        dur = int(e.findtext("duration") or 0)
        if e.tag == "backup":
            pos -= dur
        elif e.tag == "forward":
            pos += dur
        elif e.tag == "note":
            chord, grace = e.find("chord") is not None, e.find("grace") is not None
            voices.setdefault((e.findtext("staff") or "1", e.findtext("voice") or "1"), []).append((prev if chord else pos, e))
            if not chord and not grace:
                prev, pos = pos, pos + dur
        elif not voices:
            lead.append(e)
        elif e.tag == "attributes" and e.find("clef") is not None:
            clefs.append((pos, e))  # 小節の途中の音部記号は、時間上の位置を覚えておく
        else:
            (trail if e.tag == "barline" else lead).append(e)
    out, done = list(lead), set()
    length = max((on + int(e.findtext("duration") or 0) for notes in voices.values() for on, e in notes
                  if e.find("grace") is None), default=0)
    for (staff, voice), notes in sorted(voices.items()):
        at = 0
        for onset, e in sorted(notes, key=lambda x: x[0]):
            if e.find("chord") is None:
                for t, c in clefs:
                    if id(c) not in done and t <= onset and (c.find("clef").get("number") or "1") == staff:
                        out.append(c)
                        done.add(id(c))
                if onset != at:
                    gap = ET.Element("forward" if onset > at else "backup")
                    ET.SubElement(gap, "duration").text = str(abs(onset - at))
                    out.append(gap)
                    at = onset
            out.append(e)
            if e.find("chord") is None and e.find("grace") is None:
                at = onset + int(e.findtext("duration") or 0)
        for t, c in clefs:  # 小節の終わりで替わる音部記号
            if id(c) not in done and (c.find("clef").get("number") or "1") == staff:
                out.append(c)
                done.add(id(c))
        if at < length:  # 短い声部は小節の終わりまで空白で埋める。埋めないと、次の音部記号が手前に描かれる
            pad = ET.Element("forward")
            ET.SubElement(pad, "duration").text = str(length - at)
            out.append(pad)
        if length:
            back = ET.Element("backup")
            ET.SubElement(back, "duration").text = str(length)
            out.append(back)
    for e in lead:  # 小節頭の音部記号は小節線の後ろに描かせる。前だと Verovio は1つ手前の音の前に描いてしまう
        for c in e.findall("clef"):
            c.set("after-barline", "yes")
    measure[:] = out + [c for _, c in clefs if id(c) not in done] + trail


LEVELS = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4}


def set_beams(notes):
    """連桁でつながる音の並びに <beam> を付ける。2本目以降は音価から決める。"""
    # defer: 16分音符の2本目の途切れ (8つを4+4で区切るなど) は写真から読まず、つながる限り続ける。読み違いが目立ったら Claude に区切りも聞く
    levels = [LEVELS[e.findtext("type")] for e in notes]
    for i, e in enumerate(notes):
        for b in range(1, levels[i] + 1):
            before = i > 0 and levels[i - 1] >= b
            after = i < len(notes) - 1 and levels[i + 1] >= b
            kind = ("continue" if after else "end") if before else "begin" if after else "backward hook" if i else "forward hook"
            beam = ET.Element("beam", number=str(b))
            beam.text = kind
            kids = list(e)
            at = next((kids.index(c) for c in kids if c.tag in ("notations", "lyric", "play", "listen")), len(kids))
            e.insert(at, beam)


def beat_beams(root):
    """見比べ用に、連桁のない小節の8分音符以下を拍 (4分音符) ごとにつなぐ。保存するデータには付けない。
    旗のままだと、写真の連桁の本数 (32分か64分か) と見比べにくい。つなぐ範囲は写真と違ってよく、本数だけ合えばよい。"""
    div = 1
    for m in root.iter("measure"):
        div = int(m.findtext("attributes/divisions") or div)
        if m.find("note/beam") is not None:
            continue
        pos, runs = 0, {}
        for e in m:
            dur = int(e.findtext("duration") or 0)
            if e.tag == "backup":
                pos -= dur
            elif e.tag == "forward":
                pos += dur
            elif e.tag == "note" and e.find("chord") is None and e.find("grace") is None:
                run = runs.setdefault((e.findtext("staff"), e.findtext("voice")), [[]])
                if e.find("rest") is not None or e.findtext("type") not in LEVELS:
                    run.append([])
                else:
                    if run[-1] and run[-1][-1][0] != pos // div:
                        run.append([])
                    run[-1].append((pos // div, e))
                pos += dur
        for run in runs.values():
            for g in run:
                if len(g) > 1:
                    set_beams([e for _, e in g])


def tuplet_marks(root):
    """見比べ用に、連符の頭と終わりに <tuplet> を付ける (Verovio は <tuplet> がないと数字を描かない)。保存するデータには付けない。"""
    # defer: 同じ声部で連符の割り方が同じ音を、actual-notes 個ずつ1組にする。4分+8分の3連符のように音価の混ざる連符は組が
    # ずれて数字の位置が変わる。見比べがそれを食い違いと言い出したら、長さの合計で区切る
    for m in root.iter("measure"):
        if m.find("note/notations/tuplet") is not None:
            continue
        groups = {}
        for e in m.findall("note"):
            if e.find("chord") is not None or e.find("grace") is not None:
                continue
            tm = e.find("time-modification")
            g = groups.setdefault((e.findtext("staff"), e.findtext("voice")), [[]])
            ratio = tm is not None and (tm.findtext("actual-notes"), tm.findtext("normal-notes"))
            if g[-1] and (g[-1][-1][0] != ratio or len(g[-1]) == int(ratio[0])):
                g.append([])
            if ratio:
                g[-1].append((ratio, e))
        for g in groups.values():
            for run in g:
                if len(run) > 1:
                    for kind, (_, e) in (("start", run[0]), ("stop", run[-1])):
                        notations = e.find("notations")
                        if notations is None:
                            notations = ET.SubElement(e, "notations")
                        ET.SubElement(notations, "tuplet", type=kind, number="1")


def compare_images(d, n):
    """ページ n の段ごとに p{n}_c{k}.png (上: 写真 / 下: 描き直し) を作り、[(k, {その段の小節番号})] を返す。
    段の数が写真と合わないときは対応が取れないので [] を返す。"""
    root = ET.parse(d / f"p{n}.musicxml").getroot()
    show_accidentals(root)
    beat_beams(root)
    tuplet_marks(root)
    for m in root.iter("measure"):
        tidy(m)
    xml = ET.tostring(root, encoding="unicode")
    measures = root.find("part").findall("measure")
    starts = [i for i, m in enumerate(measures) if i == 0 or m.find("print[@new-system='yes']") is not None]
    if len(starts) != len(list(d.glob(f"p{n}_s*.jpg"))):
        return []
    tk = verovio.toolkit(False)
    tk.setResourcePath(str(Path(verovio.__file__).parent / "data"))  # 既定の場所は別スレッドからは引き継がれない
    tk.setOptions({"breaks": "none", "pageWidth": 20000, "pageHeight": 3000, "noJustification": True, "scale": 100,
                   "mnumInterval": 1, "header": "none", "footer": "none"})
    systems = []
    for k, first in enumerate(starts, 1):
        last = starts[k] if k < len(starts) else len(measures)
        tk.select({"measureRange": f"{first + 1}-{last}"})  # 小節は番号ではなく並び順で指定する
        tk.loadData(xml)
        png = bytes(resvg_py.svg_to_bytes(svg_string=tk.renderToSVG(1), background="white"))
        drawn = Image.open(io.BytesIO(png)).convert("RGB")
        drawn = ImageOps.expand(drawn.crop(ImageOps.invert(drawn).getbbox()), 30, "white")
        photo = Image.open(d / f"p{n}_s{k}.jpg")
        photo = photo.resize((WIDTH, photo.height * WIDTH // photo.width))
        drawn = drawn.resize((WIDTH, drawn.height * WIDTH // drawn.width))
        sheet = Image.new("RGB", (WIDTH, photo.height + drawn.height + 80), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), "PHOTO (the printed score - ground truth)", fill="red", font_size=22)
        sheet.paste(photo, (0, 40))
        draw.text((10, photo.height + 48), "RENDERED (from the data - measure numbers shown)", fill="blue", font_size=22)
        sheet.paste(drawn, (0, photo.height + 80))
        sheet.save(d / f"p{n}_c{k}.png")
        systems.append((k, {m.get("number") for m in measures[first:last]}))
    return systems
