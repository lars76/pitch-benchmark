from pathlib import Path

RAW = Path("raw")
OUTPUT = Path("output")
CONSENSUS = RAW / "consensus"

SAMPLE_RATE, HOP_SIZE = 16000, 256
VOICED_THRESHOLD = 0.5

CORPUS_FMIN = 65.0
SPEECH_FMAX = 400.0
MUSIC_FMAX = 1200.0

IDENTITY_PANEL = "identity"
FACTORS = ("scene", "room", "mic")
CARRIER = ("level",)
STAGE_ORDER = (*FACTORS, *CARRIER)

PANEL_STAGES = {
    "identity":       (),
    "level":          ("level",),
    "scene":          ("scene", "level"),
    "room":           ("room", "level"),
    "mic":            ("mic", "level"),
    "scene_room":     ("scene", "room", "level"),
    "scene_mic":      ("scene", "mic", "level"),
    "room_mic":       ("room", "mic", "level"),
    "scene_room_mic": ("scene", "room", "mic", "level"),
}
SCORED_PANELS = tuple(panel for panel in PANEL_STAGES if panel != IDENTITY_PANEL)

RENDER_SEED = 42
DESIGN_SEED = 20260826
SALT_SAMPLING = 0
CAL_SHARE = 3
CAL_MIN_GROUPS = 4
CAL_DIR, TEST_DIR = "valid", "test"

MAX_CLIPS, MAX_SECONDS = 40, 10.0

SOURCE_SLOTS = 2
SIR_STRATA = (("heavy", (-6.0, 2.0)), ("moderate", (2.0, 8.0)), ("light", (8.0, 20.0)))
ACTIVE_WIN_S, ACTIVE_HOP_S, ACTIVE_CUT_DB = 0.100, 0.050, 25.0
NOISE_FMIN_HZ = 20.0
NOISE_ALPHA_RANGE = (0.0, 1.0)

SCENE_SOURCES = (
    {"name": "colored", "kind": "colored",
     "point": False, "label_safe": True},
    {"name": "demand", "kind": "segments", "root": "DEMAND",
     "glob": "ch01.wav", "channel": "mix", "seg_seconds": 10.0,
     "point": False, "label_safe": False},
    {"name": "aishell3", "kind": "grouped", "root": "AISHELL3",
     "glob": "*/wav/*/*.wav", "group_re": r"^[^/]+/wav/([^/]+)/",
     "per_group": 10, "select_seed": 11,
     "point": True, "label_safe": False},
    {"name": "mir1k_accomp", "kind": "segments", "root": "MIR1K",
     "subdir": "Wavfile", "glob": "*.wav", "channel": "left", "seg_seconds": 4.0,
     "point": True, "label_safe": False},
)

ROOM_MAX_SECONDS = 8.0
ROOM_MIN_SUPPORT_S = 0.05
ROOM_T60_TARGET_S = 0.4
ROOM_T60_LOG_SIGMA = 0.6
ROOM_DRR_TARGET_DB = 5.0
ROOM_DRR_SIGMA_DB = 8.0
T20_END_DB = 25.0
ROOM_T20_SNR_MARGIN_DB = 10.0
DIRECT_PATH_DB = -20.0
DIRECT_HALF_MS = 2.5
FADE_MS = 5.0
TRUNCATE_ABOVE_FLOOR_DB = 10.0
TRUNCATED_SNR_BIAS_DB = 10.0
NOISE_FLOOR_TAIL_FRAC = 0.9
ENVELOPE_WIN_S = 0.01

SCENE_ROOMS = (
    {"name": "slr28", "kind": "rirs", "root": "RIRS",
     "subdir": "real_rirs_isotropic_noises", "glob": "*_rir_*.wav"},
    {"name": "openair", "kind": "openair", "root": "OPENAIR",
     "per_env": 4, "select_seed": 7,
     "formats": ("b-format", "mono", "stereo", "surround-5-1"),
     "exclude": ("forest-scale-model", "house-of-commons-auralizations",
                 "slinky-ir", "st-marys-abbey-reconstruction",
                 "st-patricks-church-patrington-model",
                 "tvisongur-sound-sculpture-iceland-model",
                 "virtual-membranes", "waveguide-web-example-audio")},
)

F_LO_HZ = 20.0
F_HI_HZ_LOG_RANGE = (3400.0, 7500.0)
HP_ORDER_CHOICES = (2, 4)
LP_ORDER_CHOICES = (4, 8)

LEVEL_PEAK_DBFS = -8.0

DEGRADATION = ("DEMAND", "AISHELL3", "MIR1K", "RIRS", "OPENAIR",
               "LIBRISPEECH", "TAU2019", "MUSDB18")

POOLS = (
    {"name": "colored", "kind": "colored",
     "point": False, "label_safe": True},
    {"name": "librispeech", "kind": "grouped", "root": "LIBRISPEECH",
     "glob": "*/*/*.flac", "group_re": r"^([^/]+)/",
     "per_group": 20, "select_seed": 11, "min_levels": 64,
     "point": True, "label_safe": False},
    {"name": "tau2019", "kind": "grouped", "root": "TAU2019",
     "glob": "*/*.wav", "group_re": r"^([^/]+)/",
     "per_group": 360, "select_seed": 11, "min_levels": 64,
     "point": False, "label_safe": True},
    {"name": "musdb", "kind": "stems", "root": "MUSDB18",
     "glob": "*/*.stem.mp4", "streams": (1, 2, 3), "min_levels": 64,
     "point": True, "label_safe": False},
)

ROOMS = (
    {"name": "sim_rirs", "kind": "rirs", "root": "RIRS",
     "subdir": "simulated_rirs", "glob": "*/*/*.wav"},
)

MEDLEYDB_TRACKS = (
    'A Classic Education - NightOwl',
    'Aimee Norwich - Child',
    'Alexander Ross - Goodbye Bolero',
    'Alexander Ross - Velvet Curtain',
    'Auctioneer - Our Future Faces',
    'AvaLuna - Waterduct',
    'BigTroubles - Phantom',
    'Celestial Shore - Die For Us',
    'Clara Berry And Wooldog - Air Traffic',
    'Clara Berry And Wooldog - Stella',
    'Clara Berry And Wooldog - Waltz For My Victims',
    'Creepoid - OldTree',
    'Dreamers Of The Ghetto - Heavy Love',
    'Faces On Film - Waiting For Ga',
    'Grants - PunchDrunk',
    'Helado Negro - Mitad Del Mundo',
    'Hezekiah Jones - Borrowed Heart',
    'Hop Along - Sister Cities',
    'Invisible Familiars - Disturbing Wildlife',
    'Lushlife - Toynbee Suite',
    'Matthew Entwistle - Dont You Ever',
    'Meaxic - Take A Step',
    'Meaxic - You Listen',
    'Music Delta - 80s Rock',
    'Music Delta - Beatles',
    'Music Delta - Britpop',
    'Music Delta - Country1',
    'Music Delta - Country2',
    'Music Delta - Disco',
    'Music Delta - Gospel',
    'Music Delta - Grunge',
    'Music Delta - Hendrix',
    'Music Delta - Punk',
    'Music Delta - Reggae',
    'Music Delta - Rock',
    'Music Delta - Rockabilly',
    'Night Panther - Fire',
    'Port St Willow - Stay Even',
    'Secret Mountains - High Horse',
    'Snowmine - Curfews',
    'Steven Clark - Bounty',
    'Strand Of Oaks - Spacestation',
    'Sweet Lights - You Let Me Down',
    'The Districts - Vermont',
    'The Scarlet Brand - Les Fleurs Du Mal',
    'The So So Glos - Emergency',
)

AUDIO_SUBTYPE = "PCM_16"
POOL_SUBTYPE = "PCM_16"
DATASET_NAME = "dataset.json"
POOLS_NAME = "pools.json"
RENDERS_NAME = "renders.json"
CLIPS_NAME = "clips.json"
POOL_AUDIO_NAME = "pool.flac"
POOL_INDEX_NAME = "index.json"


POWER_FLOOR = 1e-12
RMS_FLOOR = 1e-9
