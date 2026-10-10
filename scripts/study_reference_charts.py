"""
Structure of the preset reference charts (ADR 0001): snaps, streams, colour,
finishers, density shape, kiai, repetition.  Every file is checked ranked by
md5 against osu!.db before it is read; nothing else in Songs is touched.

    python scripts/study_reference_charts.py [--json out.json]
"""
import sys, json, hashlib, statistics as st
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taiko.data import osu_db
from taiko.data.osu_parser import OsuTaikoParser
from taiko.data.grid import Grid
from taiko.eval.metrics import snap_switch_rate

SONGS = Path("D:/osu!/Songs")
REFS = {
    "deathstream": [
        ("2283239 Dragon Guardian - Tenkai e no Kippu", "Shining Blade"),
        ("1977770 Dragon Guardian - Sei Juuji Gakuen", "Arrow of Destiny"),
        ("2009040 Dragon Guardian - Ankoku Butoukai", "Dancing in the Night under the Black Moon"),
        ("2111683 SKYWINGS - TONIGHT", "I Will Meet You Under The Sky"),
        ("2111842 Dragon Guardian - Tabidachi no Asa", "Rise From This Nightmare and Save Our Lands!"),
        ("2139170 Dragon Guardian - Mahou no Sho", "Magical Obsession"),
        ("2159934 ANCIENT MYTH - Chaos to Infinity", "Frontier of Eternity"),
        ("2180835 CADAVER  REVADAC", "YaniFR & Eltigant's Miracle"),
        ("2191269 ANCIENT MYTH - Zenith", "Sunlight Symphony"),
        ("2217406 Dragon Guardian - Eien no Tsurugi", "Ancient Miracle"),
        ("2287038 SKYWINGS", "Swords Lead to Forever Free"),
        ("2390049 IRON ROSE", "Scent of Despair"),
        ("2422005 Dragon Guardian - Kikaijikake no Kami", "Dragon God"),
        ("2415871 ryu5150 - My spirit lives on", "Tatsujin"),
        ("2447178 ryu5150 - Like a Universe", "Galaxy"),
        ("2559071 ryu5150 - Mighty Red", "Scarlet Fury"),
    ],
    "speed": [
        ("2514119 jun with Alison - SUNKiSS _3 DROP ~jun side~", "Love Oni"),
        ("2514119 jun with Alison - SUNKiSS _3 DROP ~jun side~", "Sunlight Love"),
        ("1938286 Camellia - POWA OF DA WILDANES", "ONI"),
    ],
    "tech": [
        ("2132229 Terminal 11 - Adcar", "Inner Oni"),
        ("2132229 Terminal 11 - Adcar", "Hell Oni"),
        ("2132229 Terminal 11 - Adcar", "Illegal"),
        ("2132229 Terminal 11 - Adcar", "ulko's Nervousness"),
        ("2170880 Noisia - Stigma (Billain Remix)", "Contaminated Escutcheon"),
    ],
}
RANKED = {4, 5}                         # osu!.db status: ranked, approved
SNAPS = [(1, "1/1"), (1/2, "1/2"), (1/3, "1/3"), (1/4, "1/4"), (1/6, "1/6"),
         (1/8, "1/8"), (1/12, "1/12"), (1/16, "1/16")]
FAST = 1/4 + 0.01                       # a gap at or under 1/4 beat continues a stream
SPARK = " .:-=+*#%@"


def find_osu(folder, version):
    for p in (SONGS / folder).glob("*.osu"):
        if p.stem.endswith(f"[{version}]"):
            return p
    return None


def red_at(reds, t):
    cur = reds[0]
    for r in reds:
        if r.time <= t:
            cur = r
        else:
            break
    return cur


def snap_name(gap, mpb):
    for frac, name in SNAPS:
        if abs(gap - frac * mpb) <= max(2.0, 0.03 * frac * mpb):
            return name
    return "rest" if gap > mpb else "other"


def study(path, db_entry):
    bm = OsuTaikoParser().parse_file(path)
    reds = [p for p in bm.timing_points if p.uninherited]
    greens = [p for p in bm.timing_points if not p.uninherited]
    grid = Grid(bm.timing_points)
    hits = sorted((n for n in bm.notes if not n.is_long), key=lambda n: n.time)
    t = [n.time for n in hits]
    main = max(reds, key=lambda r: (next((x.time for x in reds if x.time > r.time), t[-1]) - r.time))
    mpb_main = main.beat_length

    # gaps -> snaps
    snaps = Counter(snap_name(b - a, red_at(reds, b).beat_length) for a, b in zip(t, t[1:]))
    tot = sum(snaps.values())

    # streams: maximal runs of gaps <= 1/4 beat
    runs, cur = [], [0]
    for i in range(1, len(t)):
        if t[i] - t[i - 1] <= FAST * red_at(reds, t[i]).beat_length:
            cur.append(i)
        else:
            runs.append(cur); cur = [i]
    runs.append(cur)
    lens = [len(r) for r in runs]
    in_run = lambda k: sum(l for l in lens if l >= k) / len(t)
    long_runs = [r for r in runs if len(r) >= 5]

    # colour inside streams
    col = ["k" if ("kat" in hits[i].note_type) else "d" for i in range(len(hits))]
    alt = mono = 0
    grams = Counter()
    for r in long_runs:
        s = "".join(col[i] for i in r)
        alt += sum(a != b for a, b in zip(s, s[1:])); mono += len(s) - 1
        grams.update(s[j:j + 4] for j in range(len(s) - 3))
    seg = []
    for r in long_runs:
        s = "".join(col[i] for i in r); k = 1
        for a, b in zip(s, s[1:]):
            if a == b: k += 1
            else: seg.append(k); k = 1
        seg.append(k)

    # finishers
    fin = [i for i, n in enumerate(hits) if "big" in n.note_type]
    def beat_pos(ms):
        r = red_at(reds, ms); return ((ms - r.exact_time) / r.beat_length) % r.meter
    on_beat = sum(min(beat_pos(t[i]) % 1, 1 - beat_pos(t[i]) % 1) * red_at(reds, t[i]).beat_length <= 3 for i in fin)
    on_down = sum(min(beat_pos(t[i]), red_at(reds, t[i]).meter - beat_pos(t[i])) * red_at(reds, t[i]).beat_length <= 3 for i in fin)
    in_stream = sum(0 < i < len(t) - 1 and t[i] - t[i - 1] <= FAST * mpb_main and t[i + 1] - t[i] <= FAST * mpb_main for i in fin)

    # bars: rhythm patterns at 1/48 ticks, and density per 4 bars
    bars = []
    for k, r in enumerate(reds):
        end = reds[k + 1].time if k + 1 < len(reds) else t[-1] + 1
        bar_ms = r.beat_length * r.meter
        x = r.exact_time
        while x < end and bar_ms > 0:
            bars.append((x, min(x + bar_ms, end), r.beat_length))
            x += bar_ms
    import bisect
    pats, counts = [], []
    for a, b, mpb in bars:
        lo, hi = bisect.bisect_left(t, a - 2), bisect.bisect_left(t, b - 2)
        counts.append(hi - lo)
        pats.append(tuple(round((t[j] - a) / mpb * 48) for j in range(lo, hi)))
    nonempty = [i for i, p in enumerate(pats) if len(p) >= 2]
    seen = Counter(pats[i] for i in nonempty)
    rec_all = sum(seen[pats[i]] > 1 for i in nonempty) / max(len(nonempty), 1)
    rec_15 = sum(any(pats[j] == pats[i] and j != i and abs(bars[j][0] - bars[i][0]) <= 15000 for j in nonempty)
                 for i in nonempty) / max(len(nonempty), 1)
    blocks = [sum(counts[i:i + 4]) for i in range(0, len(counts), 4)]
    mx = max(blocks) or 1
    spark = "".join(SPARK[min(len(SPARK) - 1, round(b / mx * (len(SPARK) - 1)))] for b in blocks)
    kiai, k0 = [], None
    for p in bm.timing_points:
        if p.kiai and k0 is None: k0 = p.time
        if not p.kiai and k0 is not None: kiai.append((k0 / 1000, p.time / 1000)); k0 = None
    if k0 is not None: kiai.append((k0 / 1000, t[-1] / 1000))
    kiai_mark = "".join("^" if any(s <= bars[min(i, len(bars) - 1)][0] / 1000 < e for s, e in kiai) else " "
                        for i in range(0, len(counts), 4))
    nps5 = max(sum(1 for x in t if s <= x < s + 5000) for s in t) / 5

    return dict(
        file=path.name, sr=round(db_entry.taiko_stars.get(0, float("nan")), 2),
        bpm=round(60000 / mpb_main, 1), red_lines=len(reds), sv_lines=sum(abs(p.beat_length + 100) > 1e-6 for p in greens),
        length_s=round((t[-1] - t[0]) / 1000, 1), hits=len(t), avg_nps=round(len(t) / ((t[-1] - t[0]) / 1000), 2), peak5_nps=nps5,
        snaps={k: round(v / tot, 3) for k, v in snaps.most_common()},
        switch_per_100=round(snap_switch_rate(bm.notes, grid), 2),
        stream_share={f">={k}": round(in_run(k), 3) for k in (5, 9, 17, 33, 65)},
        max_run=max(lens), median_run=st.median([l for l in lens if l >= 5] or [0]),
        kat_share=round(sum(c == "k" for c in col) / len(col), 3),
        stream_alternation=round(alt / max(mono, 1), 3), mono_segment_mean=round(st.mean(seg), 2) if seg else 0,
        top_4grams=[g for g, _ in grams.most_common(6)],
        finishers=len(fin), fin_on_beat=round(on_beat / max(len(fin), 1), 2), fin_on_downbeat=round(on_down / max(len(fin), 1), 2),
        fin_in_stream=in_stream,
        bar_recurrence=round(rec_all, 3), bar_recurrence_15s=round(rec_15, 3),
        kiai=[(round(a), round(b)) for a, b in kiai], density=spark, kiai_bars=kiai_mark,
    )


def main():
    db = osu_db.read("D:/osu!/osu!.db")
    out = {}
    for preset, refs in REFS.items():
        out[preset] = []
        for folder, version in refs:
            p = find_osu(folder, version)
            if p is None:
                print(f"MISSING {folder} [{version}]"); continue
            md5 = hashlib.md5(p.read_bytes()).hexdigest()
            e = db.get(md5)
            if e is None or e.status not in RANKED:
                print(f"SKIP not ranked ({None if e is None else e.status}): {p.name}"); continue
            out[preset].append(study(p, e))
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    for preset, rows in out.items():
        print(f"\n=== {preset} ({len(rows)} maps) ===")
        for r in rows:
            print(f"\n{r['file']}\n  SR {r['sr']}  BPM {r['bpm']}  {r['length_s']}s  {r['hits']} hits  avg {r['avg_nps']} / peak5s {r['peak5_nps']} nps"
                  f"  red {r['red_lines']}  sv {r['sv_lines']}")
            print(f"  snaps {r['snaps']}  switch/100 {r['switch_per_100']}")
            print(f"  streams {r['stream_share']}  max {r['max_run']}  median(>=5) {r['median_run']}")
            print(f"  kat {r['kat_share']}  alt-in-stream {r['stream_alternation']}  mono-seg {r['mono_segment_mean']}  4grams {r['top_4grams']}")
            print(f"  finishers {r['finishers']} on-beat {r['fin_on_beat']} downbeat {r['fin_on_downbeat']} in-stream {r['fin_in_stream']}")
            print(f"  recurrence {r['bar_recurrence']} (15s {r['bar_recurrence_15s']})  kiai {r['kiai']}")
            print(f"  |{r['density']}|\n  |{r['kiai_bars']}|")


if __name__ == "__main__":
    main()
