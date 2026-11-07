from enum import StrEnum

class RiskLevel(StrEnum):
    SAFE       = "safe"        # read-only, auto-approve
    MUTATING   = "mutating"    # changes files, ask
    DANGEROUS  = "dangerous"   # runs arbitrary things, ask (louder)