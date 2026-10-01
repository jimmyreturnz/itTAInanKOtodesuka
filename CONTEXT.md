# tAIkoMapper

Generates osu!taiko maps from audio. This glossary fixes the words used for
what the model reads, what it writes, and how the result is judged.

## Maps and charts

**Song**:
One audio file. The unit the train/validation split is made on.
_Avoid_: track, audio (for the unit)

**Beatmapset**:
osu!'s bundle of maps that share one song.
_Avoid_: set, mapset

**Map**:
One `.osu` difficulty of a beatmapset, as a mapper wrote it.
_Avoid_: beatmap, difficulty (for the file), diff

**Ranked map**:
A map whose file is exactly the one osu! ranked (the same md5). A rate edit,
cut version or local tweak that carries a ranked map's ID is not one.
_Avoid_: ranked difficulty, approved map

**Chart**:
The notes of a map as the model sees or produces them.
_Avoid_: map (for the model's output), notes

**Held-out song**:
A song in the validation split, never seen in training.

## Difficulty

**SR**:
A map's star rating: one number for how hard it is overall.
_Avoid_: difficulty, stars, level

**Difficulty profile**:
What kind of map it is, measured over the whole map: its motif taken map-wide,
its average and peak NPS, and its SR. Unlike a window's motif, it describes the
map without giving away any one stretch of its notes.
_Avoid_: style, skill breakdown

**SR band**:
A named range of SR used to group maps: kantan <2, futsuu 2–3,
muzukashii 3–4, oni 4–5.5, inner 5.5–7, extreme 7+.

**Reference map**:
A ranked map whose difficulty profile a generated chart should resemble.

**Preset**:
A named difficulty profile averaged from reference maps a player chose as
typical of that name: deathstream, speed, tech, and so on.
_Avoid_: style, style class

## Rhythm

**Motif**:
A 16-number summary of a chart's rhythm and patterns (snap mix, colour
changes, density, bursts).
_Avoid_: style, preset (for the vector)

**Snap**:
The beat subdivision a note sits on: 1/1, 1/2, 1/3, 1/4, 1/6 or 1/8, and
1/12 in tech charts. A note snapped to its line keeps that snap even if its
millisecond is truncated by 1 ms.
_Avoid_: divisor (outside code), beat snap

**Binary snap**:
1/1, 1/2, 1/4 or 1/8.

**Ternary snap**:
1/3 or 1/6. Normal in ranked maps when used where the music has them.
_Avoid_: irregular snap

**Unusual snap**:
Any subdivision that is neither binary nor ternary, such as 1/5, 1/7, 1/12 or
1/16.
_Avoid_: irregular snap, weird snap

**Tech**:
How much of a chart's rhythm is off the straight grid: its share of ternary and
unusual snaps. A tech chart is one where that share is high; 1/12 appears only
in tech charts.
_Avoid_: weird, irregular

**Speed**:
How much of a chart runs faster than 1/4: its share of 1/8, together with its
tempo. A speed chart is a standard chart at a higher pace, not a different
rhythm.

**Mid-stream**:
A position whose neighbours on both sides are within 1/4 beat. Ranked maps
never put a finisher there (at most 0.10% of such positions in any SR band).

**Silence**:
Audio near the song's own quietest level. Ranked maps put about 1% of their
notes there.

**Unsnapped note**:
A note on no line within 2 ms. It has no snap to be right or wrong about.
