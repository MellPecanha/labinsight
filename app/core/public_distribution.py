"""Non-clinical comparison bands derived from the public BIOPRO_L dataset."""

# Unweighted 5th and 95th percentiles calculated from non-missing BIOPRO_L
# values in the LabInsight feature panel. These are descriptive public-data
# bands, not laboratory reference intervals or clinical decision limits.
NHANES_PUBLIC_BANDS = {
    "LBXSATSI": (8.0, 46.0),
    "LBDSALSI": (35.0, 46.0),
    "LBXSAPSI": (51.0, 175.0),
    "LBXSASSI": (13.0, 38.0),
    "LBXSC3SI": (21.0, 28.0),
    "LBDSBUSI": (2.9, 8.6),
    "LBXSCLSI": (97.0, 106.0),
    "LBDSCRSI": (48.62, 113.2),
    "LBXSGTSI": (8.0, 71.0),
    "LBDSIRSI": (7.0, 28.0),
    "LBXMAGN": (1.7, 2.2),
    "LBDSPHSI": (0.9, 1.45),
    "LBXSKSI": (3.6, 4.6),
    "LBXSNASI": (136.0, 143.0),
    "LBDSTBSI": (3.42, 18.81),
    "LBDSCASI": (2.2, 2.5),
    "LBDSTPSI": (65.0, 79.0),
    "LBDSUASI": (184.4, 452.0),
}

PUBLIC_BAND_LABEL = "NHANES BIOPRO_L, percentis não ponderados 5–95"
