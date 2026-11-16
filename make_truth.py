"""bench/truth/ の正解 (ピアノ2段の MusicXML) を music21 のコーパスから作る開発用スクリプト。
  uv run --with music21 python make_truth.py   既にあるファイルは上書きしない

music21 はアプリの依存に入れない (この作業でしか使わない) ので --with で一時的に入れる。
コーパスの権利は別なので、出来たファイル (bench/) は git に入れない。
"""
import copy
import sys
from pathlib import Path

import xml.etree.ElementTree as ET

from music21 import corpus, layout, stream, clef, harmony

import server

TRUTH = Path(__file__).parent / "bench" / "truth"

# 名前: (コーパスのパス, 先頭から何小節使うか)。 test_ で始まるものは改善の判断に使わない (最後に確かめるためだけに測る)
PIANO = {
    "clara_polonaise1": ("schumann_clara/polonaise_op1n1", 40),
    "clara_polonaise3": ("schumann_clara/polonaise_op1n3", 32),
    "dichterliebe2": ("schumann_robert/dichterliebe_no2", 18),
    "beach_prayer": ("beach/prayer_of_a_tired_child", 31),
    "test_clara_polonaise2": ("schumann_clara/polonaise_op1n2", 32),
    "test_clara_polonaise4": ("schumann_clara/polonaise_op1n4", 40),
    "test_aloha_oe": ("liliuokalani/aloha_oe", 23),
    "test_handel_lascia": ("handel/rinaldo/Lascia_chio_pianga", 30),
    "test_clara_trio3": ("schumann_clara/opus17/movement3", 32),
}
# 4声のコラールを、ソプラノとアルトを上の段、テノールとバスを下の段の2声ずつにしたもの (賛美歌集のピアノ譜と同じ形)
CHORALES = {
    "chorale_bwv66_6": "bach/bwv66.6",
    "chorale_bwv269": "bach/bwv269",
    "test_chorale_bwv347": "bach/bwv347",
    "test_chorale_bwv153_1": "bach/bwv153.1",
}


def piano(path, bars):
    s = corpus.parse(path)
    staves = [p for p in s.parts if isinstance(p, stream.PartStaff)][-2:]
    out = stream.Score()
    for p in staves:
        p = copy.deepcopy(p)
        for m in list(p.getElementsByClass("Measure"))[bars:]:
            p.remove(m)
        out.insert(0, p)
    out.insert(0, layout.StaffGroup(list(out.parts), symbol="brace"))
    return out


def chorale(path):
    s = corpus.parse(path)
    sop, alto, ten, bass = s.parts[:4]
    out = stream.Score()
    for hi, lo, c in ((sop, alto, clef.TrebleClef()), (ten, bass, clef.BassClef())):
        ps = stream.PartStaff()
        for mh, ml in zip(hi.getElementsByClass("Measure"), lo.getElementsByClass("Measure")):
            m = stream.Measure(number=mh.number)
            if mh.timeSignature:
                m.timeSignature = copy.deepcopy(mh.timeSignature)
            if mh.keySignature:
                m.keySignature = copy.deepcopy(mh.keySignature)
            if mh.paddingLeft:
                m.paddingLeft = mh.paddingLeft
            for i, src in enumerate((mh, ml), 1):
                v = stream.Voice(id=str(i))
                for n in src.flatten().notesAndRests:
                    n = copy.deepcopy(n)
                    n.lyrics = []
                    n.stemDirection = "up" if i == 1 else "down"
                    v.insert(n.offset, n)
                m.insert(0, v)
            ps.append(m)
        ps.getElementsByClass("Measure")[0].insert(0, c)
        out.insert(0, ps)
    out.insert(0, layout.StaffGroup(list(out.parts), symbol="brace"))
    return out


def clean(score):
    for e in list(score.recurse().getElementsByClass(harmony.ChordSymbol)):
        e.activeSite.remove(e)
    for n in score.recurse().notes:
        n.lyrics = []
    return score


def arcs_to_ties(f):
    """ある音が終わった瞬間に始まる同じ高さの音へのスラーはタイにする。譜面ではタイと見分けがつかず、弾く人もタイとして弾く。
    アプリが homr の出力に使うのと同じ処理 (server.slurs_to_ties)。描いた画像は変わらない。"""
    tree = ET.parse(f)
    before = ET.tostring(tree.getroot())
    for part in tree.getroot().findall("part"):
        server.slurs_to_ties(part)
    if ET.tostring(tree.getroot()) != before:
        tree.write(f, encoding="UTF-8", xml_declaration=True)
        print("arcs to ties:", f.name)


def main():
    TRUTH.mkdir(parents=True, exist_ok=True)
    jobs = {**{k: lambda v=v: piano(*v) for k, v in PIANO.items()}, **{k: lambda v=v: chorale(v) for k, v in CHORALES.items()}}
    for name, make in jobs.items():
        f = TRUTH / f"{name}.musicxml"
        if f.exists():
            continue
        clean(make()).write("musicxml", fp=f)
        print("wrote", f.name)
    for f in sorted(TRUTH.glob("*.musicxml")):
        arcs_to_ties(f)


if __name__ == "__main__":
    main(*sys.argv[1:])
