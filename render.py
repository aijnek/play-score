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


def compare_images(d, n):
    """ページ n の段ごとに p{n}_c{k}.png (上: 写真 / 下: 描き直し) を作り、[(k, {その段の小節番号})] を返す。
    段の数が写真と合わないときは対応が取れないので [] を返す。"""
    root = ET.parse(d / f"p{n}.musicxml").getroot()
    show_accidentals(root)
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
