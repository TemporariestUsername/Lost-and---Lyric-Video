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
               "“scribble” scrawls through the emulsion; every "
               "“stab” slams in whole, big and out of line with the "
               "orderly scribbles, the frame jolting and darkening for an "
               "instant; a pencil slash is struck beneath it, left there like "
               "an annotation. While they say “with time” her tally marks "
               "surface: their time is her count.",
        type="Typed monospace caps. FOUND / LOST and SAFE / NICE set against "
             "each other; LOST, ANYTHING and FIND held large; “with time "
             "with time...” typed over itself like a typewriter overstrike.",
        sync="Stabs slam in (jolt, exposure hit) and are slashed on the exact attacks; scrawls on the scribbles; "
             "nothing pulses on the beat."),
    "They brought a man": dict(
        world="memory",
        visual="The coats' white drains into night: he is brought in. He is "
               "never shown; he arrives as warmth, a light leak creeping in "
               "from the right and searching the fog a step per bar. On "
               "“VERY” it finds the one warm point inside her and she "
               "glows back. The title's missing word arrives here: each second "
               "“found...” is written into the empty space after a faint "
               "“lost and”, so for a moment the title reads whole, for "
               "him and then again for her.",
        type="Her voice as light. COULD FIND / HARD TO FIND, the second only "
             "seeping in; HE FOUND apart from \u201cin the end\u201d; VERY; "
             "FOREVER reaching then letting go; EVER typed in the coats' dark grey "
             "monospace and left alone. "
             "“in her” and “in him” echo from the same place.",
        sync="The warmth advances a step per bar; the flare lands on "
             "“VERY”; “lost and” surfaces on “oh he / oh she”."),
    "Spoken I: I can help you find": dict(
        world="memory",
        visual="Near-dark, nothing but his light. As he names protecting, "
               "trusting and believing, three shafts of warm light pour down "
               "through the dark, one per word, and each virtue sits in its "
               "own shaft, lit by it. The same shafts are cut in The Cutting.",
        type="His voice: Inter Light caps, wide tracking, warm light, perfectly "
             "still. PROTECTING, TRUSTING, BELIEVING larger, each in its shaft; "
             "“even if they're very hard to find” apart, small.",
        sync="Each shaft pours open on its word (the words re-timed to the "
             "spoken onsets); nothing moves on the beat."),
    "Break II": dict(
        world="memory",
        visual="His three shafts stay after his words, breathing slowly. Dust "
               "drifts down inside them; when she answers with a wordless held "
               "note it turns gold, still drifting. Just before the coats return, one "
               "flicker of the future: the shafts being cut.",
        type="No lyric. His last line finishes leaving.",
        sync="The shafts breathe over two bars, not on the beat; the gold "
             "swells with her held note (3:24.4-3:28.3); the glimpse lands "
             "on its last onset."),
    "White Coats II": dict(
        world="institute",
        visual="A hard cut from his gold shafts back into the white on "
               "“But”. Colder, brighter, more scratched than White "
               "Coats I. The coats escalate from “too lost” to “too "
               "hard to find” to the verdict; once “we found she isn't "
               "worth the time” has been read, the frame clips to white on its "
               "last word.",
        type="Typed, still. BUT held alone, then “with time with time” "
             "overstruck; LOST; the stab chant slams in and is slashed as in "
             "White Coats I; HARD; FOUND / WORTH (the title's word in their "
             "mouths); FIND.",
        sync="Stabs slammed on their sung attacks (re-timed); her tally surfaces "
             "on “with time”; the white clip lands on “time”."),
    "A gun and a bullet": dict(
        world="institute",
        visual="His finding becomes a hunt. The coats' white drains into the "
               "night as their clock follows him into it. On “bullet” the "
               "first sharp thing in the film appears: the bullet itself, "
               "tumbling slowly end over end down the centre of the frame, top "
               "to bottom, gone just before \u201cHe found her\u201d. On “he found her” his light closes on her; the "
               "second time it stops short, and “found” lands in the "
               "title's space again, behind the same unreadable smudge, with "
               "“her...” after it.",
        type="Until the last line, everything is the coats' cold typed "
             "monospace: their report of the hunt, typed as it is sung, a "
             "cursor waiting after each phrase through the silences. GUN, "
             "BULLET, FIND, DID, then HE DID... seeping in; FOUND. The last "
             "\u201cfound her...\u201d returns to her voice, in the title's space.",
        sync="Every phrase re-timed to its sung onset after the silences; the "
             "bullet starts its fall on “bullet” and leaves the frame after "
             "“he did...”, right before “He found her”."),
    "Spoken II: I won't lose you": dict(
        world="memory",
        visual="His answer to the coats' verdict. Near-dark. On “together” all three of his shafts "
               "pour open at once.",
        type="His voice, still warm light: I WON'T LOSE YOU, then TOGETHER "
             "large in the shafts, and “we'll find you the” TIME, his "
             "time answering their “not worth the time”.",
        sync="The words re-timed to when he says them; the shafts open on "
             "“together”."),
    "The Run": dict(
        world="run",
        visual="The only time they are free: warm and fast, sodium and neon "
               "smearing into long-exposure streaks, motel and diner photographs "
               "rushing past. Speed is sideways, never toward the viewer and "
               "never on the beat. The tally marks stop: time isn't counted "
               "here. 'Lost' turns good (lost in each other), and the title is "
               "sung outright: “lost and found” set plainly in the title's "
               "place, then its words slide past each other into “found and "
               "lost...”.",
        type="Her voice as warm light. Four RANs, no ellipses, scattered "
             "across the frame; FOREVER / EVER; TUNNEL / MOTEL; LOST / LOST; each OTHER; "
             "FOUND with its echo; the title, then its reversal; forever and "
             "ever and ever stepping down.",
        sync="Words re-timed where Whisper ran them together; the bars land on "
             "the “But...” freeze."),
    "But...": dict(
        world="void",
        visual="Her prophecy comes true. The run keeps moving (the band never "
               "stops) but time comes back: the uncounted bars scratch in. The "
               "warm lights go out one by one; in the dark the coats arrive as "
               "cold light round a bend and swell into their white.",
        type="BUT... in her light; FOREVER... lets go and only the coats' END "
             "stays; “when the” COATS “come round the bend”.",
        sync="A mark per eighth note from “But”; a light out on each "
             "sparse hit; the flood swells from “bend” with the band."),
    "White Coats III": dict(
        world="institute",
        visual="White Coats III begins inside the coats' white that flooded "
               "“But...”. Colder and harsher than ever; the chant returns; "
               "they turn on him.",
        type="Their typed, still voice. MAN / LOST; the stabs slam in and are "
             "slashed; CUT OUT his “something hard to find”, her words in "
             "her own serif inside their line, struck through in pencil once "
             "said; LOST; ANYTHING at their darkest; EXCEPT... alone, then "
             "“something hard to find” again in her serif, untouched: the "
             "one thing they cannot make her find.",
        sync="Words re-timed after the pauses; the strike lands on "
             "“find”; her tally surfaces on “with time”."),
    "The Cutting": dict(
        world="memory",
        visual="Out of the coats' white, his three shafts return with "
               "PROTECTING, TRUSTING, BELIEVING still lit inside them where he "
               "set them in Spoken I. Each “cut” is surgery: a scalpel traces "
               "an excision round his word, his light wells out of the incision, "
               "redder, and drips; when she sings the word the wound is sewn shut "
               "over it in thick red thread (a strikeout and sutures at once) and "
               "the shaft drains. GUN and BULLET in the coats' type land in "
               "his wounds where PROTECTING and TRUSTING were. What they take "
               "from her is cut out of her lines, leaving the gaps; taking her "
               "eyes takes the focus, taking her ears stills the world. The years stay: her tally, the one sharp "
               "thing, brighter on each “years”.",
        type="Her voice as light; his virtues in his type; GUN / BULLET and "
             "“with time... with time...” in the coats' type (overstruck, "
             "their phrase now her condition); OF HIM in his light on the right, "
             "and her... on the left.",
        sync="Incisions on the “cut” vowels; each virtue sewn shut as it is "
             "sung; GUN and BULLET land in the wounds on their words; the taken words vanish just after "
             "they are sung; her agains begin to double into Refrain II."),
    "Refrain II: Stay lost now girl": dict(
        world="her",
        visual="Refrain I remembered: the same refrain sung after it has all "
               "happened. Same room and the same layout per line, so it is "
               "recognised, but she is deaf now: nothing answers the beat and the "
               "room stays still. Each line is written over a faint exposure of "
               "how it looked the first time (FIND lands on the old FOUND). "
               "Nothing is fully unwritten any more: every line leaves its trace "
               "and her agains pile up. The visions that flashed past as the "
               "future settle in as still exposures, because they have happened. "
               "The photographs fade over 'Throw away everything he could find'.",
        type="Her voice at display size, as Refrain I; the first time's words "
             "beneath; earlier lines left behind as faint traces.",
        sync="Only the word onsets drive anything; “Never again...” is "
             "held, then everything goes to white and the story starts over "
             "in the Verse A reprise."),
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
    ("refrain_ii", "Refrain II: Stay lost now girl",
     "Refrain I remembered: the same refrain, sung after it has all happened. Same room and the same layout per line as Refrain I, so it is recognised, but she is deaf now: nothing answers the beat (no smear, no rewind, nothing flung) and the room stays still. Each line is written over a faint exposure of how it looked the first time, so FIND lands on the old FOUND ('everything he found' is now 'everything he could find'). Nothing is fully unwritten any more: every line leaves its trace and her agains pile up, thickest under 'Never again...'. The visions that flashed past in Refrain I as the future settle in as still exposures, because they have happened. The photographs fade over 'Throw away everything he could find'. As the last note ends it all goes to white, for the story to start over in the Verse A reprise. Starts a second early, in The Cutting's 'Of him... and her...', to show the hand-off.",
    ),
    ("the_cutting", "The Cutting",
     "What they did to each of them, in her words. Out of the coats' white his three shafts pour back, with PROTECTING, TRUSTING and BELIEVING still lit where he set them in Spoken I (a memory of his offer). Each 'cut' is surgery: a scalpel traces an excision round his word (the frame jolts, as on the stabs), and his light wells out of the incision, redder, and runs down in drips while the word trembles. When she sings the word the wound is sewn shut over it in thick red thread: a strike through it, short uneven stitches across it, a strikeout and sutures at once. His light chokes under them and the shaft drains; the sewn-shut wounds stay until 'They took her arms'. GUN and BULLET in the coats' type land in his wounds: GUN where PROTECTING was sewn shut, BULLET where TRUSTING was, each wound's edge flaring red as it lands while the sewn virtue under it fades; BELIEVING's stays empty. (No drawn bullet here; the falling bullet belongs to A gun and a bullet.) What they take from her is cut out of her lines, leaving the gaps: 'They took her ___ to hug / ___ to run / Her ___ to see and ___ to hear'. Taking her eyes takes the focus; taking her ears stills the world (the drift, the fog, the grain). What they leave her is the years: her tally, sharp through the blur, brightening on each 'years'. 'Alone' in her serif with 'with time... with time...' overstruck in the coats' type: their phrase is now her condition. Her agains begin to double (the deaf ghosts of Refrain II). Then OF HIM... in his light on his side, and her... on hers. The clip starts 3.6 s early, in White Coats III's white, to show the hand-off.",
    ),
    ("white_coats_iii", "White Coats III",
     "They turn on him. It begins inside the coats' white that flooded 'But...' (no cut). Their typed, still voice: MAN / LOST; the stab chant slams in and is slashed; 'We'll have to CUT OUT his' and then 'something hard to find' in her own serif inside their line, struck through in pencil as it is sung; 'with time with time...' overstruck, trailing off as in the earlier White Coats, while her tally surfaces; how LOST this girl can be; ANYTHING in their darkest ink; then EXCEPT... alone, and 'something hard to find' again in her serif, untouched: the one thing they cannot make her find.",
    ),
    ("but", "But...",
     "Rebuilt: the one moment her prophecy comes true. The band never stops, so neither does the run; time comes back instead. On 'But...' her tally faintly returns and the bars the run didn't count scratch back in, one per eighth note. On the sparse hits of 'forever always comes to an end' the run's warm lights go out one by one, leaving the darkest frame in the film; FOREVER and the rest let go and only the coats' END stays. In the dark the coats arrive as cold light swinging in from the right, round a bend (one flicker of the 'coats' vision she saw in Refrain I as it turns toward us), and it swells with the band into their white, so White Coats III begins inside it, her count standing in it.",
    ),
    ("the_run", "The Run",
     "The one time they are free. Warm and fast, but nothing pulses and nothing comes at the viewer: speed is sideways (long-exposure streaks, motel and diner photographs rushing past, light trails behind her words). No tally marks: time isn't counted here. four 'ran's (no ellipses) scattered across the frame; FOREVER / EVER; TUNNEL / MOTEL; 'lost' turns good (LOST / LOST, lost in each OTHER); FOUND with its echo. Then the title is sung outright: 'lost and found' set plainly in the title's place, and its words slide past each other into 'found and lost...'. 'Forever and ever and ever' steps down and lets go before 'But...'.",
    ),
    ("spoken_ii", "Spoken II: I won't lose you",
     "His answer to the coats' verdict. The 'found her...' from the hunt finishes leaving; near-dark. I WON'T LOSE YOU in his still light; on 'together' all three shafts pour open at once (not one per word as in Spoken I), and 'we'll find you the' TIME sits in them: his time answering their 'not worth the time'. Words re-timed to when he says them.",
    ),
    ("gun_and_bullet", "A gun and a bullet",
     "Revised for suspense: until the final 'he found her...', every line is the coats' cold typed monospace, their report of the hunt, typed as it is sung, with a cursor that waits and blinks through each silence. The night darkens as he closes in. On 'bullet' the bullet itself tumbles slowly end over end down the centre of the frame and is gone after 'he did...', right before 'He found her'. The last 'found her...' returns to her voice in the title's space behind the unreadable smudge.",
    ),
    ("white_coats_ii", "White Coats II",
     "Escalation. A hard cut from his gold shafts back to the white on 'But': BUT held alone, then 'with time with time' overstruck. Colder, brighter and more scratched than White Coats I. LOST ('this girl is too lost'); the stab chant slams in and is slashed as before (re-timed to the sung attacks); HARD ('too hard to find'); the verdict FOUND / WORTH (the title's word in their mouths), and once 'we found she isn't worth the time' has been read, the frame swells to near-white on 'time'. Ends on FIND, rhyming with White Coats I.",
    ),
    ("break_ii", "Break II",
     "His three shafts stay after his words and breathe slowly (over two bars, not on the beat). Dust drifts down inside them, one steady drift throughout: her, visible only in his light. When she answers with a wordless held note (3:24.4) it turns gold. Just before the coats return on 'But', one flicker of the future: the shafts being cut.",
    ),
    ("spoken_i", "Spoken I: I can help you find",
     "Near-dark, only his light. Three shafts pour down as he names protecting, trusting and believing, and each virtue sits in its own shaft, lit by it (the same shafts are cut in The Cutting). His words are light and perfectly still: no drift, no ghost copies. 'Even if they're very hard to find' sits apart, small. The spoken words are re-timed to when he actually says them ('I' at 3:04.0, 'trusting' 3:07.45, 'even' 3:11.05).",
    ),
    ("brought_a_man", "They brought a man",
     "Latest: 'in her' diffuses away after it is sung; in 'Her forever And ever in him' everything but EVER lets go (blurs, lifts and fades) as EVER lands, and EVER stays: no ellipsis, in the coats' dark grey ink on a faint patch of their cold light. Also: COULD FIND against HARD TO FIND; HE FOUND / SHE FOUND apart from 'in the end'; the 'lost and' before each completing 'found...' only a blurred smudge; FOREVER reaching in her warm light.",
    ),
    ("white_coats_i", "White Coats I",
     "Latest: no punctuation on screen except ellipses (film-wide from now on). Each 'stab' slams in whole on its attack, big and out of line with the orderly 'scribble scribble', and startles: the frame jolts and the exposure hits down for an instant, with a straight pencil slash struck beneath. 'We've' waits for its real entry (2:00.2); LOST is the held word in 'We'll find out how lost she really is'; 'with time with time...' is typed over itself. The coats never move: typed on, still, blinked out. FOUND / LOST and SAFE / NICE pairs; ANYTHING, FIND; her tally surfaces during 'with time'.",
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
