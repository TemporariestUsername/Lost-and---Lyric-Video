"""Storyboard spec: one entry per section in analysis/out/grid.json.

world    colour world the section lives in (drives palette in the renderer)
         institute | memory | her | run | void
visual   what is drawn
type     how the lyric is set
sync     what locks to which part of the grid
"""

WORLDS = {
    "void": ("Void", "White field, one violet point"),
    "her": ("Her room", "Pale paper, violet stipple, amber voice"),
    "institute": ("Institute", "Graph paper, ink, red pen"),
    "memory": ("Memory", "Night navy, amber contour, violet light"),
    "run": ("The run", "Sodium night, neon accents"),
}

SECTIONS = {
    "Intro": dict(
        world="void",
        visual="White field. A single violet point breathes on every kick. "
               "A thin ECG line draws left to right across the five bars, "
               "one tick per downbeat. The title types in: lost and, then a "
               "blinking cursor where “found” should be.",
        type="Title only, Garamond italic. The missing word is the hook of "
             "the whole video.",
        sync="Point pulses on kicks; ECG ticks on downbeats; title types on "
             "the last bar."),
    "Verse A: Through eyes": dict(
        world="her",
        visual="The white room as an isometric line drawing: bed, a door with "
               "no handle. She is a stipple figure, seated. On eyes/years, a "
               "close-up of her stipple face with calendar pages flicking in "
               "the reflection. On “a voice came to her” an amber "
               "contour enters from frame edge and traces a hand. On “she "
               "got lost inside him” her particles stream into his "
               "outline and fill it.",
        type="Her voice: words bloom out of blur on their sung onset and "
             "dissolve upward after the line. Backing echoes (sweet, so "
             "sweet) are small ghost copies.",
        sync="Slow camera push on each downbeat; particles brighten on "
             "kicks."),
    "There comes a once": dict(
        world="memory",
        visual="Split diptych. Left panel is the “no” timeline, "
               "right is “yes”. Each “once” is a clock-tick "
               "flash and a hard cut. Both panels end the same way: his "
               "contour unravels into a straight line. “Lost to her / to "
               "the world” pulls back to a globe of dots.",
        type="Her voice, one line per panel; the parenthesised (once...) "
             "echo sits in the opposite panel.",
        sync="Cuts on downbeats; tick flashes on the word onsets of "
             "“once”."),
    "Refrain I: Stay lost now girl": dict(
        world="her",
        visual="The type is the image. “Stay lost now girl” fills "
               "the frame, then scatters to particles on the pulse. “Throw "
               "away everything he found”: map pins (the things he "
               "found) are flung out of frame one per pulse.",
        type="Her voice at display size (~220px). Echoes (forever, forever / "
             "again, again) trail as ghost copies.",
        sync="Scatter on kicks; “Never again” holds for its full "
             "length, then the frame fades to white."),
    "Break I": dict(
        world="institute",
        visual="Transition into the institute. Graph-paper grid draws in, "
               "one gridline family per pulse; the institute mark stamps on "
               "the downbeat; the case file slides in.",
        type="No lyric.",
        sync="Grid on kicks; stamp on bar 33."),
    "White Coats I": dict(
        world="institute",
        visual="Style frame A. Clipboard chart with her stipple ID photo and "
               "a vitals trace. Three faceless white coats watch. Scribbles "
               "grow over the chart on each “scribble”; each "
               "“stab” punches a hole with a red ink ring and a red "
               "frame flash. “A nice white room” becomes a "
               "blueprint floor plan. “With time” is a clock face; "
               "“find anything” is a radar sweep across the chart.",
        type="White-coat voice: monospace caps typed letter by letter into "
             "a NOTES field, block cursor blinking on the eighths. "
             "“Stab” in red bold; HOW set larger in red.",
        sync="Holes on the exact stab onsets; clock ticks on eighths; "
             "coats nod on kicks."),
    "They brought a man": dict(
        world="memory",
        visual="The Finder is drawn as one continuous amber contour that "
               "writes on over two bars, standing in the doorway. His "
               "finding is a radar sweep over a topographic map of her "
               "mind; the contours tighten to a glowing core (“something "
               "VERY hard to find”). Then the mirror: her particles find "
               "the same core in him.",
        type="Her voice; VERY set at 1.6× on its onset; (in her) / "
             "(in him) as ghost echoes.",
        sync="Radar sweep period = one bar; core flares on the downbeat "
             "after “found”."),
    "Spoken I: I can help you find": dict(
        world="memory",
        visual="Style frame B. Dark and still. Two profiles facing each "
               "other: his is a warm contour, hers is light made of points. "
               "Three threads are drawn between them as he names them: "
               "protecting, trusting, believing. These threads come back in "
               "The Cutting.",
        type="His voice: Inter Light caps, wide tracking, no animation, "
             "words appear on time like a close-mic subtitle.",
        sync="Each thread draws on from its word onset; nothing moves on "
             "the beat here."),
    "Break II": dict(
        world="memory",
        visual="The threads braid and glow; slow push-in; the ECG line from "
               "the intro returns, now amber.",
        type="No lyric.",
        sync="Thread glow on kicks."),
    "White Coats II": dict(
        world="institute",
        visual="Back to the chart, colder. The threads show faintly on her "
               "ID photo. “Too lost” crosses her chart out in red. "
               "“Not worth the time” brings down a rubber stamp. "
               "The coats turn to face camera. The clock runs faster.",
        type="White-coat voice as before.",
        sync="Stamp on a downbeat; clock ticks double speed."),
    "A gun and a bullet": dict(
        world="institute",
        visual="Ink still life on a steel tray: a revolver outline, then a "
               "single bullet, placed on consecutive downbeats. “Go "
               "and find that girl”: radar over a city grid. “He "
               "found her” twice: the crosshair locks onto her point, "
               "and the second time it dissolves. He doesn’t shoot.",
        type="Her voice for narration; the quoted order is set in "
             "white-coat mono.",
        sync="Object placements on downbeats; crosshair locks on the "
             "“found” onsets."),
    "Spoken II: I won't lose you": dict(
        world="memory",
        visual="Dark, close. His amber hand opens and sets the bullet down. "
               "The three threads re-tie.",
        type="His voice, as in Spoken I.",
        sync="Threads re-tie on the word onsets."),
    "The Run": dict(
        world="run",
        visual="Style frame C. Sodium night. Tunnel rings rush past and "
               "flash on the kick. The motel sign flickers on with "
               "“motel”; a diner sign; an alley in line art. A map "
               "route line gains a pin on every downbeat. “Forever and "
               "ever and ever”: headlight trails loop into an infinity "
               "sign.",
        type="Her voice, warm-tinted, with glow. “Lost and found, found "
             "and lost” flips like a split-flap board.",
        sync="Rings and lamps on kicks; route pins on downbeats; flaps on "
             "word onsets."),
    "But...": dict(
        world="void",
        visual="Everything freezes on “But…”. Colour drains "
               "over one bar. “Round the bend”: headlight glare "
               "resolves into three white-coat silhouettes, then the frame "
               "goes paper-white.",
        type="Her voice; “But…” held alone, centred, for the "
             "whole pause.",
        sync="Freeze frame exactly on the “But” onset."),
    "White Coats III": dict(
        world="institute",
        visual="His file now: the ID photo is his amber contour. “Cut "
               "out his something hard to find”: a dotted scalpel line "
               "is drawn around the heart of his contour. ANYTHING is huge. "
               "“Except…” holds, cursor blinking.",
        type="White-coat voice; ANYTHING set at display size.",
        sync="Scalpel line traces over one bar."),
    "The Cutting": dict(
        world="institute",
        visual="The three threads from Spoken I are cut one per line, each "
               "snip on its “cut” onset, and fall slack and grey. The "
               "tray returns with the gun and bullet. Her stipple figure is "
               "wiped away region by region (arms, legs, eyes, ears) by a "
               "clinical eraser sweep, not gore, leaving a year counter "
               "that keeps ticking. “Agains and agains”: earlier "
               "shots repeat in shrinking nested frames.",
        type="Her voice, colder white. The virtue labels (PROTECTING / "
             "TRUSTING / BELIEVING) grey out as each thread is cut.",
        sync="Snips on word onsets; eraser wipes on kicks; year counter on "
             "eighths."),
    "Refrain II: Stay lost now girl": dict(
        world="her",
        visual="The same big refrain as Refrain I, but the scattered "
               "particles never re-form; they drift into the year counter.",
        type="Her voice at display size, as Refrain I.",
        sync="Scatter on kicks, slower decay than Refrain I."),
    "Verse A reprise": dict(
        world="her",
        visual="Shot-for-shot rhyme with Verse A, aged: gridlines cracked, "
               "calendar pages piled up, her stipple sparser. The amber "
               "voice returns, fainter. Memory or return? It stays open.",
        type="Her voice, as Verse A.",
        sync="Same camera moves as Verse A, on the same bar offsets."),
    "Outro": dict(
        world="void",
        visual="The ECG line flattens into a horizon. The single bullet "
               "sits on the tray in white space. Her point pulses once more "
               "on the last downbeat. The title retypes “lost and”, "
               "the cursor blinks, cut to white.",
        type="Title only.",
        sync="Final pulse on the last downbeat; cursor blinks on eighths "
             "to the end of the audio."),
}

STYLE_FRAMES = [
    ("A_white_coats", "White Coats I", "Scribble scribble, stab stab the charts."),
    ("B_spoken_help", "Spoken I: I can help you find",
     "I can help you find protecting, trusting and believing"),
    ("C_the_run", "The Run", "Through every tunnel, every motel."),
]
