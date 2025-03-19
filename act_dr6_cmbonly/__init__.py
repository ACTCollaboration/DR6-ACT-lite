__author__ = "Hidde T. Jense"
__url__ = "https://github.com/ACTCollaboration/DR6-ACT-lite"
__version__ = "1.0.0"

try:
    from .act_dr6_cmbonly import ACTDR6CMBonly  # noqa: F401
    from .PlanckActCut import PlanckActCut  # noqa: F401

    def get_cobaya_class():
        return ACTDR6CMBonly
except ImportError:
    pass
