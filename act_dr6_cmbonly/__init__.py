__author__ = "Hidde T. Jense"
__url__ = "https://github.com/ACTCollaboration/dr6-cmbonly"
__version__ = "0.1.4"

try:
    from .act_dr6_cmbonly import ACTDR6CMBonly  # noqa: F401
    from .PlanckActCut import PlanckActCut  # noqa: F401
    from .PlanckNPIPEActCut import PlanckNPIPEActCut  # noqa: F401
except ImportError:
    pass

try:
    from .act_dr6_jaxlike import ACTDR6jax  # noqa: F401
except ImportError:
    ACTDR6jax = None  # noqa: F401
