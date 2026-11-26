"""Candidate universes. Pairs are only searched *within* a sector: an economic
reason for two stocks to co-move makes a statistical finding far more credible.

Caveat (survivorship bias): these are today's large caps. Companies that
shrank or were delisted since the start date aren't here, which flatters
backtests. Mention this in the write-up.
"""

UNIVERSES = {
    "banks": ["JPM", "BAC", "C", "WFC", "GS", "MS", "USB", "PNC", "TFC", "SCHW"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "HAL"],
    "staples": ["KO", "PEP", "PG", "CL", "KMB", "GIS", "MDLZ", "HSY", "KHC", "CLX"],
    "semis": ["NVDA", "AMD", "INTC", "TXN", "QCOM", "AVGO", "MU", "ADI", "MCHP", "NXPI"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC", "ES"],
}

# Wider universes for the walk-forward backtest (stage 3): mid caps included,
# because heavily traded mega-cap pairs are the most likely to be arbitraged away.
# Same survivorship caveat applies, and is stronger here: companies acquired or
# delisted before today (e.g. regional banks lost in 2023) are missing.
WIDE_UNIVERSES = {
    "banks_wide": ["JPM", "BAC", "C", "WFC", "USB", "PNC", "TFC", "MTB", "FITB", "RF",
                   "HBAN", "KEY", "CFG", "ZION", "WAL", "FHN", "SNV", "CFR", "WBS", "EWBC"],
    "utilities_wide": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "XEL", "ED", "WEC", "ES",
                       "PEG", "EIX", "ETR", "DTE", "AEE", "CMS", "PPL", "CNP", "NI", "LNT"],
    "energy_wide": ["XOM", "CVX", "COP", "EOG", "OXY", "DVN", "APA", "FANG", "OVV", "MUR",
                    "PSX", "MPC", "VLO", "HAL", "SLB", "BKR", "EQT", "RRC", "SM", "CHRD"],
    "reits_wide": ["O", "SPG", "PLD", "AVB", "EQR", "ESS", "UDR", "MAA", "CPT", "PSA",
                   "EXR", "VTR", "WELL", "BXP", "VNO", "KIM", "REG", "FRT", "HST", "ARE"],
    "staples_wide": ["KO", "PEP", "PG", "CL", "KMB", "GIS", "MDLZ", "HSY", "KHC", "CLX",
                     "CPB", "CAG", "SJM", "HRL", "MKC", "CHD", "TSN", "KDP", "MO", "PM"],
    "country_etfs": ["EWA", "EWC", "EWG", "EWU", "EWJ", "EWQ", "EWL", "EWN", "EWD", "EWH",
                     "EWS", "EWZ", "EWW", "EWI", "EWP", "EWK", "EWO", "EWY", "EWT", "EZA"],
}
UNIVERSES.update(WIDE_UNIVERSES)
