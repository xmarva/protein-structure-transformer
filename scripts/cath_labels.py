"""The ten CATH architectures used as classes.

Keys are CATH (class, architecture) numbers; a domain's label is the position of its key here.
"""

ARCHITECTURES = {
    (1, 10): "Mainly Alpha: Orthogonal Bundle",
    (1, 20): "Mainly Alpha: Up-down Bundle",
    (2, 30): "Mainly Beta: Roll",
    (2, 40): "Mainly Beta: Beta Barrel",
    (2, 60): "Mainly Beta: Sandwich",
    (3, 10): "Alpha Beta: Roll",
    (3, 20): "Alpha Beta: Alpha-Beta Barrel",
    (3, 30): "Alpha Beta: 2-Layer Sandwich",
    (3, 40): "Alpha Beta: 3-Layer(aba) Sandwich",
    (3, 90): "Alpha Beta: Alpha-Beta Complex",
}

LABELS = {key: label for label, key in enumerate(ARCHITECTURES)}
