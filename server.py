"""play_score: 楽譜の写真 → homr (ローカルOMR) → Claude で校正 → MusicXML を配信する。

起動: uv run server.py   (ブラウザで http://localhost:8765 。このMac専用)
"""
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps

from render import compare_images

ROOT = Path(__file__).parent
LIB = ROOT / "library"
# homr を子プロセスで動かすスクリプト。onnxruntime は既定で利用状況を Microsoft に送信し、その送信待ちで
# 終了時に固まることがあるので、送信を切ったうえで、結果を書き終えたら後始末を待たずに終了する。
HOMR = """
import os, sys, traceback, onnxruntime
onnxruntime.disable_telemetry_events()
from homr.main import main
try:
    main()
except SystemExit:
    pass
except Exception:
    traceback.print_exc()
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
"""
PORT = 8765
HOSTS = {f"localhost:{PORT}", f"127.0.0.1:{PORT}"}
MAX_UPLOAD = 40 * 1024 * 1024

jobs: queue.Queue = queue.Queue()  # OMRは重いので1本ずつ直列に処理する
lock = threading.Lock()


# ---------- ライブラリ (library/<id>/meta.json) ----------

def read_meta(song):
    return json.loads((LIB / song / "meta.json").read_text())


def update_meta(song, log=None, **kw):
    with lock:
        meta = read_meta(song)
        meta.update(kw)
        if log:
            meta["log"] = (meta["log"] + [log])[-50:]
        (LIB / song / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
        return meta


def enqueue(song, *job):
    with lock:
        meta = read_meta(song)
        meta["pending"] += 1
        (LIB / song / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    jobs.put((song, *job))


# ---------- 認識パイプライン ----------

def claude(cwd, prompt, edit=True):
    """Claude Code をヘッドレス実行する。APIキーを外し、ログイン済みサブスクリプションで動かす。"""
    tools = ["--tools", "Read,Edit", "--allowedTools", "Read(./**)", "Edit(./**)"] if edit else ["--tools", "Read", "--allowedTools", "Read(./**)"]
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    r = subprocess.run(
        # 読み書きは曲のフォルダ内だけに限定する。指示文や写真に悪意ある文が紛れても外のファイルに触れない。
        ["claude", "-p", prompt, *tools],
        cwd=cwd, env=env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=1800,
    )
    if "Not logged in" in r.stdout + r.stderr:
        raise RuntimeError("Claude Code にログインしていません。ターミナルで claude を起動して /login を実行してください")
    if r.returncode:
        raise RuntimeError((r.stdout + r.stderr).strip()[-300:] or "claude が失敗しました")
    return r.stdout.strip()


def claude_edit(d, target, prompt):
    """target(MusicXML) を Claude に直接編集させる。壊れたXMLになったら元に戻す。"""
    backup = target.read_bytes()
    try:
        summary = claude(d, prompt)
        if not ET.parse(target).getroot().findall("part/measure"):
            raise RuntimeError("編集後のMusicXMLに小節がありません")
        return summary
    except Exception:
        target.write_bytes(backup)
        raise


RULES = """Fix only what affects how the music sounds: key signature (<fifths>), time signature, clefs,
wrong pitches/octaves, missing or spurious accidentals (an accidental carries through its measure, and
<alter> must be present on every affected <note>), wrong durations (every voice of a measure must add up
to the time signature, except pickup measures), missing or extra notes, ties.
Repeats and jumps decide the playback order, so encode every one that is printed, exactly:
- repeat barlines: <barline location="left"><repeat direction="forward"/></barline> and
  <barline location="right"><repeat direction="backward"/></barline>
- volta brackets (1., 2.): <ending number="1" type="start"/> inside the left <barline> of the first measure under
  the bracket and <ending number="1" type="stop"/> inside the right <barline> of the last measure under it
  (type="discontinue" for a final bracket that is open on the right). Every start needs its stop/discontinue.
- D.C., D.S., Fine, To Coda, segno and coda signs: a <direction> in the measure where it is printed containing
  <sound dacapo="yes"/>, <sound dalsegno="segno"/>, <sound fine="yes"/>, <sound tocoda="coda"/>,
  <sound segno="segno"/> or <sound coda="coda"/> respectively.
Ignore dynamics, fingering, slurs, pedal and layout.
Be conservative: change only what you can clearly see in the images. Use Edit for targeted changes; do not
rewrite or reformat the file, and keep it valid MusicXML."""

REPLY = """When done, reply with at most 5 short lines saying what you changed. The reader is a pianist who does not know
MusicXML: use musical terms only, never tag, attribute or file names. Refer to measures only by their `number`
attribute, which is what the app shows; never use the measure numbers printed on the page, which may differ.
If a value such as the tempo is your own guess, say so. 返事は必ず日本語で書くこと。"""


def retime(measure):
    """小節を「右手を順に → 巻き戻し → 左手を順に」の形に組み直す。直したら True。
    homr は両手の音を細かく行き来しながら書き出し、付点などで長さがずれると後ろの音を遅らせて小節を
    はみ出させることがある。行き来が多いと描き直しも乱れる。各段を順に並べて両手の長さが一致する場合に限る。"""
    kids = list(measure)
    idx = [i for i, e in enumerate(kids) if e.tag in ("note", "backup", "forward")]
    if not idx:
        return False
    body = kids[idx[0]:idx[-1] + 1]
    if any(e.tag not in ("note", "backup") for e in body):
        return False  # 小節の途中に音部記号などがある場合は触らない
    staves = {}
    for e in body:
        if e.tag == "note":
            staves.setdefault(e.findtext("staff"), []).append(e)
    if len(staves) != 2 or sum(e.tag == "backup" for e in body) < 2:
        return False
    sums = [sum(int(e.findtext("duration") or 0) for e in notes if e.find("chord") is None) for notes in staves.values()]
    voices = [{e.findtext("voice") for e in notes} for notes in staves.values()]
    if sums[0] != sums[1] or any(len(v) != 1 for v in voices):
        return False
    backup = ET.Element("backup")
    ET.SubElement(backup, "duration").text = str(sums[0])
    upper, lower = staves.values()
    measure[:] = kids[:idx[0]] + upper + [backup] + lower + kids[idx[-1] + 1:]
    return True


CHECK = """You are checking machine-read piano sheet music. The image {image} shows one system twice: the top half is
a PHOTO of the printed score (ground truth), the bottom half is the same measures RENDERED from the machine-read
data, with measure numbers printed above each measure (the photo may print different numbers; ignore those).
The layout differs (spacing, beams drawn as separate flags), so match the measures by order, left to right.
Read the image, then go through it one measure at a time and one staff at a time. For every note, compare its
vertical position on the staff (which line or space, how many ledger lines) between the two halves, then its
accidental, then its note value and dots. Compare rests, ties, clef changes, key and time signatures, repeat
signs and volta brackets too.
Ignore differences that do not change what is played: beaming, stem direction, spacing, slurs, dynamics,
pedal marks, fingering, tempo text, measure numbers, courtesy accidentals, how or where rests are drawn as long as
their total length is the same, a clef change drawn just before versus just after a barline.
Do not guess: report only differences you can actually see.
Finish with a line containing only RESULT: followed by a JSON array, one object per difference:
{{"measure": <number shown in the RENDERED half>, "hand": "right"|"left", "issue": "<what the photo shows vs what is rendered>"}}
Use [] if the two halves agree everywhere."""


def result(reply):
    """返事の最後にある「RESULT: <JSON>」を取り出す。"""
    tail = reply[reply.rfind("RESULT:") + 7:]
    return json.loads(tail[min(i for i in (tail.find("["), tail.find("{")) if i >= 0):max(tail.rfind("]"), tail.rfind("}")) + 1])


def verify_page(song, n):
    """データを描き直して写真と見比べ、食い違いを直す。直した段はもう一度見比べる (直すのは3回まで)。
    直しても見比べで合わなかった小節は、直す前に戻す (直しのほうが間違っていることがあるため)。
    確かめきれなかった小節を {小節番号: 理由} で返す。"""
    d = LIB / song
    page = d / f"p{n}.musicxml"
    todo, flags, before = None, {}, None
    for attempt in (1, 2, 3, 4):
        update_meta(song, step={"phase": "verify", "page": n, "round": attempt})
        systems = compare_images(d, n)
        if not systems:
            raise RuntimeError("段の数が写真と合わないため、描き直しての見比べはできませんでした")
        if todo is not None:
            systems = [s for s in systems if s[1] & todo]

        def check(system):  # 1段につき1つの Claude。まとめて見せるより見落としが少ない
            found = result(claude(d, CHECK.format(image=f"p{n}_c{system[0]}.png"), edit=False))
            return [{**f, "system": system[0]} for f in found if str(f.get("measure")) in system[1]]
        with ThreadPoolExecutor(6) as pool:
            found = [f for fs in pool.map(check, systems) for f in fs]
        still = {str(f["measure"]) for f in found} & (todo or set())
        if still:  # 直したはずの小節がまだ違うと言われた: その直しは信用せず、直す前の状態に戻す
            old = {m.get("number"): m for m in ET.fromstring(before).iter("measure")}
            tree = ET.parse(page)
            for part in tree.getroot().findall("part"):
                for i, m in enumerate(list(part)):
                    if m.get("number") in still and m.get("number") in old:
                        part[i] = old[m.get("number")]
            tree.write(page, encoding="UTF-8", xml_declaration=True)
            flags |= {m: "直しを試みましたが、見比べ直しても写真と合わなかったので、直す前に戻しました" for m in still}
            update_meta(song, log={"kind": "proof", "page": n, "text": "見比べ直して合わなかった直しを取り消しました: 第"
                                   + "・".join(sorted(still, key=int)) + "小節"})
            found = [f for f in found if str(f["measure"]) not in still]
        if not found:
            return flags
        if attempt == 4:  # 直す回数を使い切った
            return flags | {str(f["measure"]): "見比べで違いが残りましたが、直しきれませんでした" for f in found}

        before = page.read_bytes()
        reply = result(claude_edit(d, d / f"p{n}.musicxml", f"""A checker compared an engraving of p{n}.musicxml with the
photographed score, one system at a time, and reported these possible differences. It only saw pictures and is
sometimes wrong, especially about exact pitches:
{json.dumps(found, ensure_ascii=False)}
Files: p{n}.musicxml (the data; measure numbers are the `number` attributes), p{n}_sK.jpg (zoomed photo of system K),
p{n}_cK.png (photo above, engraving of the current data below).
For each reported difference: read that measure in p{n}.musicxml to see what the data really contains, look at
the photo, and decide. Edit only where the photo clearly shows something different from the data. If the data
already matches the photo, do nothing. If you cannot tell from the photo, do not edit and list it as unsure.
List as unsure only differences from the report above, never suspicions of your own. The measure numbers printed
in the photo often differ from the data's (print does not count a pickup measure); that is not a difference.
{RULES}
Finish with a line containing only RESULT: followed by a JSON object like
{{"fixed": [{{"measure": 12, "what": "..."}}], "unsure": [{{"measure": 30, "why": "..."}}]}}
Write "what" and "why" in Japanese for a pianist who does not know MusicXML: musical terms only."""))
        fixed, unsure = reply.get("fixed", []), reply.get("unsure", [])
        with open(d / f"p{n}_verify.jsonl", "a") as f:  # 何が指摘され、どう判断したかの記録
            f.write(json.dumps({"found": found, "fixed": fixed, "unsure": unsure}, ensure_ascii=False) + "\n")
        if fixed:
            update_meta(song, log={"kind": "proof", "page": n, "text": "描き直して写真と見比べ、直しました。\n"
                                   + "\n".join(f"- 第{f['measure']}小節: {f['what']}" for f in fixed)})
        todo = {str(f["measure"]) for f in fixed}
        flags |= {str(u["measure"]): u["why"] for u in unsure}
        if not todo:
            return flags


def process_page(song, n):
    d = LIB / song
    update_meta(song, step={"phase": "omr", "page": n})
    r = subprocess.run([sys.executable, "-c", HOMR, "--write-staff-positions", f"p{n}.jpg"], cwd=d, capture_output=True, text=True)
    page = d / f"p{n}.musicxml"
    if not page.exists():
        raise RuntimeError(f"{n}ページ目の楽譜を認識できませんでした。明るい場所で、ページ全体を真上から撮り直してください。"
                           + r.stderr.strip()[-200:])

    # 読み取り直後の後始末: はみ出した小節を組み直し、小節番号を曲全体の通し番号 (画面に出る番号) にする
    score = d / "score.musicxml"
    offset = len(ET.parse(score).getroot().findall("part[1]/measure")) if score.exists() else 0
    tree = ET.parse(page)
    for part in tree.getroot().findall("part"):
        for i, m in enumerate(part.findall("measure"), 1):
            retime(m)
            m.set("number", str(offset + i))
    tree.write(page, encoding="UTF-8", xml_declaration=True)

    # 段(大譜表)ごとの拡大画像を切り出す。ページ全体だと Claude には細部が小さすぎる。
    im = Image.open(d / f"p{n}.jpg")
    W, H = im.size
    strips = []
    for k, line in enumerate((d / f"p{n}.txt").read_text().splitlines(), 1):
        _, _, cy, _, h = map(float, line.split())
        name = f"p{n}_s{k}.jpg"
        im.crop((0, int(max(0, cy - h) * H), W, int(min(1, cy + h) * H))).save(d / name)
        strips.append(name)

    update_meta(song, step={"phase": "proof", "page": n})
    try:
        summary = claude_edit(d, page, f"""You are proofreading optical music recognition (OMR) output for a piano score.
- p{n}.jpg is the photographed page. {", ".join(strips)} are zoomed crops of each system, top to bottom.
- p{n}.musicxml is the OMR result for this page. A <print new-system="yes"/> marks the first measure of each new system.
Read the images first, then the MusicXML, compare them measure by measure, and correct recognition errors
directly in p{n}.musicxml.
If a metronome mark or tempo word is printed, make sure the first measure has <direction><sound tempo="BPM"/></direction>.
{RULES}
{REPLY}""")
        update_meta(song, log={"kind": "proof", "page": n, "text": summary})
        flags = verify_page(song, n)
        update_meta(song, flags={**read_meta(song).get("flags", {}), **flags})
    except Exception as e:  # Claude が使えなくても、そこまでの結果で先へ進む
        update_meta(song, log={"kind": "skip", "page": n, "text": str(e)})

    # 曲全体の score.musicxml の末尾にこのページの小節を足し、小節番号を 1..N に振り直す
    if score.exists():
        tree, new = ET.parse(score), ET.parse(page)
        for part, new_part in zip(tree.getroot().findall("part"), new.getroot().findall("part")):
            part.extend(new_part.findall("measure"))
    else:
        tree = ET.parse(page)
    for part in tree.getroot().findall("part"):
        for i, m in enumerate(part.findall("measure"), 1):
            m.set("number", str(i))
    tree.write(score, encoding="UTF-8", xml_declaration=True)
    update_meta(song, measures=len(tree.getroot().find("part").findall("measure")))


def process_fix(song, instruction):
    d = LIB / song
    update_meta(song, step={"phase": "fix"})
    images = ", ".join(sorted(p.name for p in d.glob("p*.jpg")))
    summary = claude_edit(d, d / "score.musicxml", f"""score.musicxml is a piano score that was produced by OMR from
photographed pages. Images in this directory: {images} (pN.jpg = page N, pN_sK.jpg = zoomed crop of system K on page N).
Measures are numbered consecutively through the whole piece in the `number` attribute.
The user, who is looking at the printed score, asks for this correction:

{instruction}

Check the relevant images, then apply the correction directly in score.musicxml.
{RULES}
{REPLY}""")
    update_meta(song, log={"kind": "fix", "ask": instruction, "text": summary})


def worker():
    while True:
        song, kind, arg = jobs.get()
        try:
            (process_page if kind == "page" else process_fix)(song, arg)
        except Exception as e:
            if not (LIB / song / "meta.json").exists():
                continue  # 処理中に曲が削除された
            update_meta(song, log={"kind": "error", "text": str(e)})
        with lock:
            if not (LIB / song / "meta.json").exists():
                continue
            meta = read_meta(song)
            meta["pending"] -= 1
            meta["step"] = None
            (LIB / song / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))


def open_image(data):
    try:
        return Image.open(io.BytesIO(data))
    except Exception:
        # iPhone の HEIC は Pillow で開けないので、macOS 標準の sips で JPEG にする
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "in").write_bytes(data)
            subprocess.run(["sips", "-s", "format", "jpeg", "in", "--out", "out.jpg"], cwd=t, capture_output=True)
            im = Image.open(Path(t) / "out.jpg")
            im.load()
            return im


# ---------- HTTP ----------

class Handler(BaseHTTPRequestHandler):
    def reply(self, code, body=None, ctype="application/json", cache="no-store"):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError("ファイルが大きすぎます")
        return self.rfile.read(n)

    def handle_request(self):
        path = self.path.split("?")[0]
        method = self.command
        if method == "GET" and path == "/":
            return self.reply(200, (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8")
        # id は自前で発行した16進のみ受け付ける (パストラバーサル防止)
        m = re.fullmatch(r"/api/songs(?:/([0-9a-f]{12})(?:/(pages|score|fix)(?:/(\d+))?)?)?", path)
        if not m:
            return self.reply(404, {"error": "not found"})
        song, sub, page = m.groups()
        if song and not (LIB / song / "meta.json").exists():
            return self.reply(404, {"error": "曲が見つかりません"})
        d = LIB / song if song else None

        if not song and method == "GET":
            metas = [json.loads(p.read_text()) for p in LIB.glob("*/meta.json")]
            return self.reply(200, sorted(metas, key=lambda x: -x["created"]))
        if not song and method == "POST":
            title = str(json.loads(self.body()).get("title") or "").strip()[:100] or time.strftime("無題 %Y-%m-%d %H:%M")
            song = uuid.uuid4().hex[:12]
            (LIB / song).mkdir(parents=True)
            meta = {"id": song, "title": title, "created": time.time(), "pages": 0, "measures": 0, "pending": 0, "step": None, "log": []}
            (LIB / song / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
            return self.reply(200, meta)
        if not sub and method == "GET":
            return self.reply(200, read_meta(song))
        if not sub and method == "DELETE":
            shutil.rmtree(d)
            return self.reply(200, {})
        if sub == "score" and method == "GET":
            f = d / "score.musicxml"
            return self.reply(200, f.read_bytes(), "application/xml") if f.exists() else self.reply(404, {"error": "まだ楽譜がありません"})
        if sub == "pages" and page and method == "GET":
            f = d / f"p{int(page)}.jpg"
            if not f.exists():
                return self.reply(404, {"error": "ページがありません"})
            return self.reply(200, f.read_bytes(), "image/jpeg", "max-age=31536000, immutable")  # 写真は差し替わらない
        if sub == "pages" and method == "POST":
            try:
                im = ImageOps.exif_transpose(open_image(self.body())).convert("RGB")
            except Exception:
                return self.reply(400, {"error": "画像として読み込めませんでした (JPEG・PNG・HEIC に対応しています)"})
            with lock:
                meta = read_meta(song)
                n = meta["pages"] = meta["pages"] + 1
                (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
            im.save(d / f"p{n}.jpg", quality=92)
            enqueue(song, "page", n)
            return self.reply(200, read_meta(song))
        if sub == "fix" and method == "POST":
            instruction = str(json.loads(self.body()).get("instruction") or "").strip()[:1000]
            if not instruction or not (d / "score.musicxml").exists():
                return self.reply(400, {"error": "修正内容が空か、まだ楽譜がありません"})
            enqueue(song, "fix", instruction)
            return self.reply(200, read_meta(song))
        return self.reply(405, {"error": "method not allowed"})

    def safe(self):
        # ブラウザで開いた別サイトからの裏リクエストを拒否する (Host: DNSリバインディング, Origin: CSRF)
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in HOSTS or (origin and origin not in {"http://" + h for h in HOSTS}):
            return self.reply(403, {"error": "forbidden"})
        try:
            self.handle_request()
        except ValueError as e:
            self.reply(400, {"error": str(e)})

    do_GET = do_POST = do_DELETE = safe

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    LIB.mkdir(exist_ok=True)
    for p in LIB.glob("*/meta.json"):  # 前回の起動で中断された処理の表示を解除
        update_meta(p.parent.name, pending=0, step=None)
    threading.Thread(target=worker, daemon=True).start()
    print(f"play_score 起動中: http://localhost:{PORT}  (このMac専用)")
    # 認証が無いので、このMacの外からは繋がせない
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
