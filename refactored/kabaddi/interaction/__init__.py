"""
Interaction Module
Handles interaction proposals, graph construction, and temporal validation.
"""
from .graph import InteractionProposalEngine, ActiveFactorGraphNetwork, render_graph_panel
from .logic import build_player_states, process_interactions
from .temporal import TemporalInteractionCandidateManager
from .classifier_bridge import ConfirmedWindowClassifierBridge

__all__ = [
    "InteractionProposalEngine",
    "ActiveFactorGraphNetwork",
    "render_graph_panel",
    "build_player_states",
    "process_interactions",
    "TemporalInteractionCandidateManager",
    "ConfirmedWindowClassifierBridge",
]

