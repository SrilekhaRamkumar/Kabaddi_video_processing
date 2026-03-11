"""
Interaction Module
Handles interaction proposals, graph construction, and temporal validation.
"""
from .graph import InteractionProposalEngine, ActiveFactorGraphNetwork
from .logic import build_player_states, process_interactions
from .temporal import TemporalInteractionCandidateManager

__all__ = [
    "InteractionProposalEngine",
    "ActiveFactorGraphNetwork",
    "build_player_states",
    "process_interactions",
    "TemporalInteractionCandidateManager",
]

