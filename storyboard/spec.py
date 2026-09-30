"""Storyboard spec: one entry per section in analysis/out/grid.json.

Direction: a long exposure of a memory. Nothing is drawn literally.
She is never shown: the film is her memory, and real photographs surface and drift through the haze.
He is never shown: he is warm light leaking in from outside the frame.
The white coats are overexposure, flicker, scratches and film burns.
Time is tally marks (one per bar) and the exposure echo.

world    colour world (drives palette and optics in the renderer)
         void | her | institute | memory | run
"""

WORLDS = {
    "void": ("Bleach", "Fog thinning to white"),
    "her": ("Her room", "Lavender haze, soft padded corner"),
    "institute": ("Institute", "Cold overexposure, flicker, scratches, burns"),
    "memory": ("Memory", "Plum dark, warm light through it"),
    "run": ("The run", "Sodium and neon streaks, fast"),
}

SECTIONS = {
    "Intro": dict(
        world="void",
        visual="Bleached white fog with dust drifting through it. On the "
               "first downbeat a single tally mark scratches into the wall, "
               "then one more every bar for the rest of the film. The title "
               "surfaces out of the fog, and the space after it, where "
               "“found” would go, stays empty.",
        type="Title only, ghosted serif.",
        sync="Tally mark on every downbeat; dust brightens slightly on each "
             "kick."),
    "Verse A: Through eyes": dict(
        world="her",
        visual="Lavender haze and a padded corner you feel more than see. "
               "She is never shown; this is her memory. Real photographs "
               "surface on downbeats and drift through the haze at different "
               "depths: an overcast shore, a curtain, rain on glass, fog "
               "lamps. From \u201ca voice came to her\u201d the memories turn "
               "to night: a figure in a doorway, a car window, streetlights. "
               "On \u201clost inside him\u201d his warmth colours them.",
        type="Her voice, dark ink in the haze. Each word focus-pulls in and "
             "sheds two ghost exposures that drift apart and linger.",
        sync="Ghost exposures swell on kicks; the warm light rises across "
             "the verse and peaks on “inside him”."),
    "There comes a once": dict(
        world="memory",
        visual="Two timelines in one frame. Each “once” flashes the "
               "image to white, and it comes back shifted: the "
               "“no” drifts left, the “yes” drifts right. "
               "“Lost to the world” pulls focus until nothing is "
               "sharp.",
        type="Her voice, white with film halation; the (once...) echoes "
             "sit on the opposite side of the frame.",
        sync="White flashes on the “once” onsets; the focus pull "
             "runs over two bars."),
    "Refrain I: Stay lost now girl": dict(
        world="her",
        visual="The type becomes the weather. The refrain is exposed large "
               "and soft, and its echoes smear sideways on every kick. On "
               "“throw away everything he found” the warm light is "
               "pushed out of frame one pulse at a time.",
        type="Her voice at display size; echoes (forever / again) pile up "
             "as fading exposures.",
        sync="Smear on kicks; “Never again” is held, and the frame "
             "bleaches to white."),
    "Break I": dict(
        world="institute",
        visual="What she foresaw arrives. The corridor that seeped in at the "
               "end of the refrain becomes the room: cold light, a hospital "
               "corridor drifting past with white coats far down it. Then "
               "the music's top end closes and the picture goes soft, dim "
               "and quiet with it, a held breath, until the fluorescent "
               "tubes strike back on in a stutter.",
        type="No lyric. 'Never again...' lets go as the room changes.",
        sync="The hum and scratches creep in bar by bar; the defocus follows "
             "the high end closing (113.0-115.1); the tubes strike on the "
             "hits 115.4, 115.8, 116.1 and hold from the 116.5 downbeat."),
    "White Coats I": dict(
        world="institute",
        visual="Cold, bright, humming. The coats never move: their words are "
               "typed on and stay perfectly still, against her drift. Every "
               "\u201cscribble\u201d tears scratches through the emulsion; "
               "every \u201cstab\u201d burns a cold white hole through the film "
               "behind the typed word. While they say \u201cwith time\u201d her "
               "tally marks surface: their time is her count.",
        type="Typed monospace caps. FOUND / LOST and SAFE / NICE set against "
             "each other; HOW, ANYTHING and FIND held large; \u201cwith time, "
             "with time\u201d typed over itself like a typewriter overstrike.",
        sync="Burns on the exact stab attacks; scratches on the scribbles; "
             "nothing pulses on the beat."),
    "They brought a man": dict(
        world="memory",
        visual="He is never shown. He arrives as warmth: a light leak "
               "creeping in from the frame edge and searching the fog, "
               "softening as it reaches her. On “something VERY hard "
               "to find” it finds the one warm point inside her, and "
               "she glows back.",
        type="Her voice; VERY swells with halation on its onset.",
        sync="The leak advances a step per bar; the flare lands on the "
             "downbeat after “found”."),
    "Spoken I: I can help you find": dict(
        world="memory",
        visual="Near-dark. As he names protecting, trusting and believing, "
               "three shafts of warm light open through the dark, one per "
               "word. They come back in The Cutting.",
        type="His voice: Inter Light caps, wide tracking, rendered as warm "
             "light through a crack. It never moves.",
        sync="Each shaft opens on its word onset; nothing else moves on "
             "the beat."),
    "Break II": dict(
        world="memory",
        visual="The three shafts breathe; dust turns gold inside them.",
        type="No lyric.",
        sync="Shaft brightness on kicks."),
    "White Coats II": dict(
        world="institute",
        visual="Colder, brighter, faster flicker. The etched chant starts to "
               "cover the walls. On “not worth the time” the "
               "exposure clips to pure white for a beat.",
        type="Etched chant, as White Coats I.",
        sync="White clip on the downbeat after “time”; burns on "
             "stabs."),
    "A gun and a bullet": dict(
        world="institute",
        visual="For the first time something is in sharp focus: a small, "
               "cold glint on the floor, crisp in a soft world. On “he "
               "found her” the warm light closes on her twice. The "
               "second time it stops short and goes no further.",
        type="Her voice for the narration; “go and find that "
             "girl” is etched.",
        sync="The glint appears on a downbeat; the light's two approaches "
             "land on the “found” onsets."),
    "Spoken II: I won't lose you": dict(
        world="memory",
        visual="His words as light again. The three shafts re-open around "
               "her.",
        type="His voice, as Spoken I.",
        sync="Shafts re-open on the word onsets."),
    "The Run": dict(
        world="run",
        visual="The only warm, fast section. Sodium and neon smear into "
               "long-exposure streaks across the fog, headlights flare "
               "through, and her ghosts trail sideways like motion blur. "
               "The tally marks stop: time isn't counted here.",
        type="Her voice, warm and hot, smeared along the direction of "
             "travel. “Lost and found, found and lost” swap "
             "places as double exposures.",
        sync="Streak bursts on kicks; a headlight flare on every "
             "downbeat."),
    "But...": dict(
        world="void",
        visual="Everything freezes: the exposure echo holds the last frame. "
               "Over one bar it bleaches to white. The coats come "
               "“round the bend” as a single hard, cold sweep of "
               "light.",
        type="“But…” held alone for the whole pause.",
        sync="The freeze lands exactly on the “But” onset."),
    "White Coats III": dict(
        world="institute",
        visual="Back in the white. The scratches now cut across the spot "
               "where his warmth used to come in.",
        type="Etched chant; ANYTHING etched at display size, then "
             "“Except…” on its own.",
        sync="Burns on stabs; flicker on eighths."),
    "The Cutting": dict(
        world="memory",
        visual="Style frame C. Dark. The three shafts from Spoken I return, "
               "and each “cut” drops a blade through one and it "
               "gutters out. The warmth behind her drains with each cut "
               "until she is a silhouette against nothing. The glint "
               "stays sharp. Then her exposures are taken one by one. For "
               "“eyes” the focus goes. For “ears” the "
               "image stops reacting to the music. The years stay: the "
               "tally marks remain.",
        type="Her voice, white-hot with halation, cooling as the light "
             "goes.",
        sync="Blades on the “cut” onsets; beat-reactivity "
             "switches off on “ears”."),
    "Refrain II: Stay lost now girl": dict(
        world="her",
        visual="The refrain again, but nothing reacts to the beat any more "
               "because she can't hear it. The echoes pile up instead: "
               "her agains, in layers.",
        type="Her voice at display size, with more ghosts than Refrain I.",
        sync="Deliberately unsynced: only the word onsets drive "
             "anything."),
    "Verse A reprise": dict(
        world="her",
        visual="A shot-for-shot rhyme with Verse A, seen through years of "
               "accumulated exposure: overexposed, the wall covered in "
               "tallies, many more ghosts. The warmth returns faintly "
               "behind her shoulder. Memory or return? It stays open.",
        type="Her voice, as Verse A.",
        sync="Same bar offsets as Verse A."),
    "Outro": dict(
        world="void",
        visual="The fog thins toward white. The glint is the last sharp "
               "thing on screen. The final tally mark lands on the last "
               "downbeat, then white.",
        type="Title returns, ghosted, with the empty space after it.",
        sync="Last tally on the last downbeat."),
}

STYLE_FRAMES = [
    ("A_lost_inside_him", "Verse A: Through eyes", "She got lost inside him", 46.3),
    ("B_stab_the_charts", "White Coats I", "Scribble scribble, stab stab the charts.", 125.9),
    ("C_cut_away", "The Cutting", "They cut away his trusting.", 368.3),
]

# Sections built so far: (slug, section name, note shown under the player)
RENDERS = [
    ("white_coats_i", "White Coats I",
     "The coats never move: typed on, perfectly still, blinked out letter by letter, against her drift (no push toward the viewer, nothing on the beat). FOUND / LOST and SAFE / NICE set against each other; HOW, ANYTHING and FIND held large. Each 'stab' burns a cold white hole through the film behind the typed word (the stabs were re-timed to their sung attacks). 'With time, with time' is typed over itself like a typewriter overstrike, and her tally marks surface while they say it. The 'coats' vision in Refrain I is re-shot from this section's 2:05.9.",
    ),
    ("break_i", "Break I",
     "What she foresaw arrives: 'Never again...' lets go and the corridor from her vision becomes the room, with white coats far down it. As the music's top end closes (1:53-1:55) the picture goes soft, dim and quiet with it; the fluorescent tubes strike back on in a stutter on the hits and hold into White Coats I. Includes a second either side for the handoffs. The corridor vision in Refrain I is re-shot from this frame at 1:50.5, so it will update when Refrain I is next rendered.",
     (105.5, 117.8)),
    ("refrain_again", "Refrain I: Stay lost now girl",
     "Timing check: the backing 'again, again' now land on the kicks (96.4 / 97.3), so the rewind snaps and the future flashes fire on the sung words (the second one had been 0.5 s early). The second 'forever' in 'Forever again' now arrives as the voice returns at 103.1 instead of 0.6 s into the silence.",
     (93.5, 106.3)),
    ("refrain_forever", "Refrain I: Stay lost now girl",
     "Revised: the second 'forever' peels off the first as a copy, drifts slowly, outlasts the rest of the line, and is the last thing unwritten, letter by letter, as 'Throw away everything' arrives. Timing corrected: the second 'forever' is sung at 88.1 s and held ~1.1 s (it had been placed 0.65 s late). The full Refrain I clip below has the old version of this moment.",
     (84.15, 92.5)),
    ("refrain_i", "Refrain I: Stay lost now girl",
     "Rebuilt as prophecy: she sees the future and tries to unmake it. Words appear faintly before they are sung; 'stay lost' is unwritten letter by letter; the memories are thrown out; the line rewinds and snaps forward on each 'again' with flashes of the future; 'forever again' recedes in a beat-stepped tunnel; a cold stop on 'Never'. Revised: calmer motion (lines drift and are unwritten, never float toward the viewer), a slow continuous tunnel, tally marks moved left. Web copy at 720p; the 1080p master passed QA and ffprobe."),
    ("verse_a_line6", "Verse A: Through eyes",
     "Revised line: 'A hand, as ever as it was once'. 'ever' and 'once' set against each other on a diagonal; 'ever' keeps opening, 'once' lands. The Verse A clip below still has the old version of this line.",
     (34.5, 44.5)),
    ("verse_a", "Verse A: Through eyes",
     "Kinetic typography (a designed shot per line, hero words, per-letter timing, camera through the type) over drifting memory photographs. Web copy at 720p; the 1080p master passed QA and ffprobe."),
    ("there_comes_a_once", "There comes a once",
     "Two timelines: the 'no' drifts left, the 'yes' drifts right; each 'once' flashes toward white; 'lost to the world' pulls focus until nothing is sharp. Web copy at 720p; the 1080p master passed QA and ffprobe."),
]
