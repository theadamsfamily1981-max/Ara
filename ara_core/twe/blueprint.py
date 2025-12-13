#!/usr/bin/env python3
"""
TWE Blueprint - 4D Organ Design Specification
==============================================

The blueprint is the complete specification for an organ:
    B(x,y,z,t) = {material, cell_type, vascular_state, schedule}

Components:
    - S(x,y,z,t): Scaffold/material field (ECM, hydrogels)
    - C(x,y,z,t): Cell-type field (parenchymal, endothelial, stromal)
    - V: Vascular graph (nodes, edges, diameters, flow targets)
    - Γ(t): Temporal schedule (dissolution, perfusion, maturation)

GUTC Integration:
    Blueprints are designed to maintain λ near-criticality during print,
    with Π fields controlling precision on structure vs. viability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto
import numpy as np


# =============================================================================
# Material and Cell Types
# =============================================================================

class MaterialType(Enum):
    """Scaffold material types."""
    COLLAGEN_I = auto()       # Primary structural ECM
    COLLAGEN_IV = auto()      # Basement membrane
    FIBRIN = auto()           # Temporary scaffold
    PEG_HYDROGEL = auto()     # Synthetic, tunable stiffness
    ALGINATE = auto()         # Rapid gelation
    GELATIN_MA = auto()       # Photo-crosslinkable
    SACRIFICIAL = auto()      # Thermo-reversible (Pluronic)
    EMPTY = auto()            # No material (void/lumen)


class CellType(Enum):
    """Cell populations for organ construction."""
    PARENCHYMAL = auto()      # Functional cells (hepatocytes, nephrons, etc.)
    ENDOTHELIAL = auto()      # Vascular lining
    SMOOTH_MUSCLE = auto()    # Vessel wall support
    PERICYTE = auto()         # Capillary support
    FIBROBLAST = auto()       # Stromal/support
    IMMUNE_RESIDENT = auto()  # Tissue-resident immune cells
    STEM_PROGENITOR = auto()  # Undifferentiated reserve
    NONE = auto()             # No cells


class VascularState(Enum):
    """Vascular channel states."""
    SOLID = auto()            # Filled with scaffold
    SACRIFICIAL = auto()      # Sacrificial material (to be removed)
    LUMEN = auto()            # Open channel
    PERFUSED = auto()         # Active flow
    ANASTOMOSED = auto()      # Connected to main circulation


# =============================================================================
# Voxel and Field Definitions
# =============================================================================

@dataclass
class Voxel:
    """Single voxel in the 4D blueprint."""
    material: MaterialType = MaterialType.EMPTY
    cell_type: CellType = CellType.NONE
    cell_density: float = 0.0        # cells/mm³
    vascular_state: VascularState = VascularState.SOLID
    stiffness_kpa: float = 1.0       # Mechanical property
    degradation_rate: float = 0.0    # 1/day

    # GUTC-relevant fields
    pi_structure: float = 1.0        # Precision on structural fidelity
    pi_viability: float = 1.0        # Precision on viability signals


@dataclass
class ScaffoldField:
    """
    S(x,y,z,t) - Scaffold/material field.

    Defines the mechanical and ECM properties at each point.
    """
    shape: Tuple[int, int, int]      # (nx, ny, nz) voxels
    resolution_um: float = 50.0       # Voxel size in microns

    # 3D arrays (initialized lazily)
    material: np.ndarray = field(default=None, repr=False)
    stiffness: np.ndarray = field(default=None, repr=False)
    degradation: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        nx, ny, nz = self.shape
        if self.material is None:
            self.material = np.full((nx, ny, nz), MaterialType.EMPTY.value, dtype=np.int8)
        if self.stiffness is None:
            self.stiffness = np.ones((nx, ny, nz), dtype=np.float32)
        if self.degradation is None:
            self.degradation = np.zeros((nx, ny, nz), dtype=np.float32)

    def set_region(self, bounds: Tuple[slice, slice, slice],
                   material: MaterialType, stiffness: float = 1.0):
        """Set material in a rectangular region."""
        self.material[bounds] = material.value
        self.stiffness[bounds] = stiffness


@dataclass
class CellField:
    """
    C(x,y,z,t) - Cell-type field.

    Defines cell populations and densities at each point.
    """
    shape: Tuple[int, int, int]
    resolution_um: float = 50.0

    cell_type: np.ndarray = field(default=None, repr=False)
    density: np.ndarray = field(default=None, repr=False)  # cells/mm³

    def __post_init__(self):
        nx, ny, nz = self.shape
        if self.cell_type is None:
            self.cell_type = np.full((nx, ny, nz), CellType.NONE.value, dtype=np.int8)
        if self.density is None:
            self.density = np.zeros((nx, ny, nz), dtype=np.float32)

    def set_region(self, bounds: Tuple[slice, slice, slice],
                   cell_type: CellType, density: float):
        """Set cell population in a region."""
        self.cell_type[bounds] = cell_type.value
        self.density[bounds] = density


# =============================================================================
# Vascular Graph
# =============================================================================

@dataclass
class VascularNode:
    """Node in the vascular tree."""
    node_id: str
    position: Tuple[float, float, float]  # (x, y, z) in mm
    node_type: str = "junction"           # inlet, outlet, junction, capillary
    diameter_um: float = 100.0
    target_flow_ul_min: float = 0.0       # Target perfusion rate

    # Connection state
    anastomosed: bool = False
    perfused: bool = False


@dataclass
class VascularEdge:
    """Edge (vessel segment) in the vascular tree."""
    edge_id: str
    source_id: str
    target_id: str
    diameter_um: float = 50.0
    length_mm: float = 1.0
    wall_thickness_um: float = 10.0

    # Printing state
    state: VascularState = VascularState.SACRIFICIAL

    # Path waypoints for non-linear vessels
    waypoints: List[Tuple[float, float, float]] = field(default_factory=list)


@dataclass
class VascularGraph:
    """
    V = (V_nodes, V_edges) - Vascular network specification.

    Defines the complete vascular tree including:
    - Arterial tree (high-pressure inlets)
    - Venous tree (low-pressure outlets)
    - Capillary beds (gas exchange regions)
    """
    nodes: Dict[str, VascularNode] = field(default_factory=dict)
    edges: Dict[str, VascularEdge] = field(default_factory=dict)

    # Topology metadata
    arterial_inlets: List[str] = field(default_factory=list)
    venous_outlets: List[str] = field(default_factory=list)

    def add_node(self, node: VascularNode):
        self.nodes[node.node_id] = node

    def add_edge(self, edge: VascularEdge):
        self.edges[edge.edge_id] = edge

    def get_downstream(self, node_id: str) -> List[str]:
        """Get all downstream edges from a node."""
        return [e.edge_id for e in self.edges.values() if e.source_id == node_id]

    def get_upstream(self, node_id: str) -> List[str]:
        """Get all upstream edges to a node."""
        return [e.edge_id for e in self.edges.values() if e.target_id == node_id]

    def total_vessel_length_mm(self) -> float:
        return sum(e.length_mm for e in self.edges.values())

    def n_capillary_beds(self) -> int:
        return sum(1 for n in self.nodes.values() if n.node_type == "capillary")


# =============================================================================
# Temporal Schedule
# =============================================================================

class ScheduleEventType(Enum):
    """Types of scheduled events during print/maturation."""
    START_PERFUSION = auto()
    REMOVE_SACRIFICIAL = auto()
    APPLY_GROWTH_FACTOR = auto()
    MECHANICAL_CONDITIONING = auto()
    TEMPERATURE_SHIFT = auto()
    LIGHT_EXPOSURE = auto()
    ANASTOMOSIS = auto()
    QUALITY_CHECKPOINT = auto()


@dataclass
class ScheduleEvent:
    """A scheduled event in the temporal program."""
    event_id: str
    event_type: ScheduleEventType
    time_hours: float              # When to trigger
    duration_hours: float = 0.0    # How long it lasts
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Dependencies
    requires_events: List[str] = field(default_factory=list)

    # State
    completed: bool = False


@dataclass
class TemporalSchedule:
    """
    Γ(t) - Temporal program for the organ.

    Defines the sequence of:
    - Scaffold dissolution steps
    - Perfusion initiation
    - Growth factor applications
    - Mechanical conditioning
    - Quality checkpoints
    """
    events: Dict[str, ScheduleEvent] = field(default_factory=dict)
    total_duration_hours: float = 168.0  # Default: 1 week

    def add_event(self, event: ScheduleEvent):
        self.events[event.event_id] = event

    def get_events_at_time(self, t_hours: float, tolerance: float = 0.5) -> List[ScheduleEvent]:
        """Get events scheduled around time t."""
        return [
            e for e in self.events.values()
            if abs(e.time_hours - t_hours) < tolerance and not e.completed
        ]

    def get_pending_events(self) -> List[ScheduleEvent]:
        """Get all incomplete events, sorted by time."""
        pending = [e for e in self.events.values() if not e.completed]
        return sorted(pending, key=lambda e: e.time_hours)

    def mark_complete(self, event_id: str):
        if event_id in self.events:
            self.events[event_id].completed = True


# =============================================================================
# Complete Blueprint
# =============================================================================

@dataclass
class OrganBlueprint:
    """
    Complete 4D organ blueprint: B = (S, C, V, Γ)

    This is the full specification passed from Meta-Brain to TWE.
    """
    # Identity
    blueprint_id: str
    organ_type: str                 # kidney, liver, heart_patch, etc.
    patient_id: Optional[str] = None

    # 3D Fields
    scaffold: ScaffoldField = None
    cells: CellField = None

    # Vascular network
    vasculature: VascularGraph = field(default_factory=VascularGraph)

    # Temporal program
    schedule: TemporalSchedule = field(default_factory=TemporalSchedule)

    # Metadata
    created_at: float = field(default_factory=time.time)
    version: str = "1.0"

    # GUTC control parameters
    target_lambda: float = 1.0      # Target criticality during print
    lambda_tolerance: float = 0.2   # Acceptable deviation

    # Robustness metrics (filled by Meta-Brain simulations)
    stress_margin: float = 0.0      # Safety factor on mechanical failure
    perfusion_tolerance: float = 0.0  # Margin on oxygen delivery
    viability_score: float = 0.0    # Overall predicted viability

    def __post_init__(self):
        if self.scaffold is None and hasattr(self, '_default_shape'):
            self.scaffold = ScaffoldField(self._default_shape)
        if self.cells is None and hasattr(self, '_default_shape'):
            self.cells = CellField(self._default_shape)

    @classmethod
    def create_empty(cls, blueprint_id: str, organ_type: str,
                     shape: Tuple[int, int, int],
                     resolution_um: float = 50.0) -> 'OrganBlueprint':
        """Create an empty blueprint with specified dimensions."""
        bp = cls(
            blueprint_id=blueprint_id,
            organ_type=organ_type,
        )
        bp.scaffold = ScaffoldField(shape, resolution_um)
        bp.cells = CellField(shape, resolution_um)
        return bp

    def n_voxels(self) -> int:
        if self.scaffold:
            return np.prod(self.scaffold.shape)
        return 0

    def volume_mm3(self) -> float:
        if self.scaffold:
            voxel_vol = (self.scaffold.resolution_um / 1000.0) ** 3
            return self.n_voxels() * voxel_vol
        return 0.0

    def layers(self) -> List['PrintLayer']:
        """Generate print layers (z-slices) for the printer."""
        if self.scaffold is None:
            return []

        nz = self.scaffold.shape[2]
        layers = []
        for z in range(nz):
            layer = PrintLayer(
                z_index=z,
                z_position_mm=z * self.scaffold.resolution_um / 1000.0,
                scaffold_slice=self.scaffold.material[:, :, z],
                cell_slice=self.cells.cell_type[:, :, z] if self.cells else None,
                density_slice=self.cells.density[:, :, z] if self.cells else None,
            )
            layers.append(layer)
        return layers

    def to_dict(self) -> Dict[str, Any]:
        """Serialize blueprint metadata (not full arrays)."""
        return {
            "blueprint_id": self.blueprint_id,
            "organ_type": self.organ_type,
            "patient_id": self.patient_id,
            "shape": self.scaffold.shape if self.scaffold else None,
            "resolution_um": self.scaffold.resolution_um if self.scaffold else None,
            "n_voxels": self.n_voxels(),
            "volume_mm3": self.volume_mm3(),
            "n_vessels": len(self.vasculature.edges),
            "n_schedule_events": len(self.schedule.events),
            "target_lambda": self.target_lambda,
            "stress_margin": self.stress_margin,
            "perfusion_tolerance": self.perfusion_tolerance,
            "viability_score": self.viability_score,
            "created_at": self.created_at,
        }


@dataclass
class PrintLayer:
    """A single z-layer for printing."""
    z_index: int
    z_position_mm: float
    scaffold_slice: np.ndarray      # 2D material array
    cell_slice: Optional[np.ndarray] = None
    density_slice: Optional[np.ndarray] = None

    # Layer-specific control hints
    print_speed_modifier: float = 1.0
    extra_dwell_ms: float = 0.0


# =============================================================================
# Blueprint Factory (Demo)
# =============================================================================

def create_demo_vascular_patch(size_mm: float = 10.0,
                                resolution_um: float = 100.0) -> OrganBlueprint:
    """
    Create a demo vascularized tissue patch blueprint.

    This is a simple test case: a cube of tissue with a single
    arterial inlet, branching tree, and venous outlet.
    """
    n_voxels = int(size_mm * 1000 / resolution_um)
    shape = (n_voxels, n_voxels, n_voxels)

    bp = OrganBlueprint.create_empty(
        blueprint_id=f"demo_patch_{int(time.time())}",
        organ_type="vascular_patch",
        shape=shape,
        resolution_um=resolution_um,
    )

    # Fill with collagen scaffold
    bp.scaffold.set_region(
        (slice(None), slice(None), slice(None)),
        MaterialType.COLLAGEN_I,
        stiffness=2.0,
    )

    # Add fibroblasts throughout
    bp.cells.set_region(
        (slice(None), slice(None), slice(None)),
        CellType.FIBROBLAST,
        density=1e5,
    )

    # Create simple vascular tree
    # Inlet at top center
    inlet = VascularNode(
        node_id="inlet_0",
        position=(size_mm/2, size_mm/2, size_mm),
        node_type="inlet",
        diameter_um=500,
        target_flow_ul_min=100,
    )
    bp.vasculature.add_node(inlet)
    bp.vasculature.arterial_inlets.append("inlet_0")

    # Branch point in center
    branch = VascularNode(
        node_id="branch_0",
        position=(size_mm/2, size_mm/2, size_mm/2),
        node_type="junction",
        diameter_um=300,
    )
    bp.vasculature.add_node(branch)

    # Two capillary beds
    for i, (dx, dy) in enumerate([(-2, 0), (2, 0)]):
        cap = VascularNode(
            node_id=f"cap_{i}",
            position=(size_mm/2 + dx, size_mm/2 + dy, size_mm/4),
            node_type="capillary",
            diameter_um=50,
        )
        bp.vasculature.add_node(cap)

    # Outlet at bottom
    outlet = VascularNode(
        node_id="outlet_0",
        position=(size_mm/2, size_mm/2, 0),
        node_type="outlet",
        diameter_um=400,
    )
    bp.vasculature.add_node(outlet)
    bp.vasculature.venous_outlets.append("outlet_0")

    # Connect with edges
    bp.vasculature.add_edge(VascularEdge(
        edge_id="e_inlet_branch",
        source_id="inlet_0",
        target_id="branch_0",
        diameter_um=400,
        length_mm=size_mm/2,
    ))

    for i in range(2):
        bp.vasculature.add_edge(VascularEdge(
            edge_id=f"e_branch_cap_{i}",
            source_id="branch_0",
            target_id=f"cap_{i}",
            diameter_um=100,
            length_mm=3.0,
        ))
        bp.vasculature.add_edge(VascularEdge(
            edge_id=f"e_cap_outlet_{i}",
            source_id=f"cap_{i}",
            target_id="outlet_0",
            diameter_um=80,
            length_mm=3.0,
        ))

    # Simple schedule
    bp.schedule.add_event(ScheduleEvent(
        event_id="start_perfusion",
        event_type=ScheduleEventType.START_PERFUSION,
        time_hours=24.0,
        parameters={"flow_rate_ul_min": 50},
    ))

    bp.schedule.add_event(ScheduleEvent(
        event_id="remove_sacrificial",
        event_type=ScheduleEventType.REMOVE_SACRIFICIAL,
        time_hours=12.0,
        parameters={"temperature_c": 4},
        requires_events=[],
    ))

    bp.schedule.add_event(ScheduleEvent(
        event_id="quality_check_1",
        event_type=ScheduleEventType.QUALITY_CHECKPOINT,
        time_hours=48.0,
        requires_events=["start_perfusion"],
    ))

    # Set robustness scores (would be computed by Meta-Brain)
    bp.stress_margin = 0.3
    bp.perfusion_tolerance = 0.25
    bp.viability_score = 0.85

    return bp
