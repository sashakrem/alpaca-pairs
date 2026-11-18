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
