import zlib

import numpy as np


def corpus_uid(name):
    return zlib.crc32(str(name).encode())


def item_seed(seed, *ids):
    return int(
        np.random.SeedSequence([int(seed), *(int(i) for i in ids)])
        .generate_state(1, dtype=np.uint32)[0]
    )


def item_rng(seed, *ids):
    return np.random.default_rng(item_seed(seed, *ids))
