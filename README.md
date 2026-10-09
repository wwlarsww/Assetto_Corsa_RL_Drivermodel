## Project Goal

The goal of this project is to build a modular reinforcement learning
platform for autonomous driving in Assetto Corsa.

The system is developed incrementally with a focus on clear software
architecture, testability and separation between simulator integration
and learning logic.


## System Overview

Assetto Corsa
    ↓ Telemetry
Python
    ↓ Vehicle State
RL Environment
    ↓ Observation
Agent
    ↓ Vehicle Actions
RL Environment
    ↓ Steuerbefehl
Assetto Corsa


## Current Scope

The current implementation reads live telemetry and static information,
loads AI spline geometry, and projects vehicle positions onto the track.

Controller integration, Gymnasium and reinforcement learning
will be added later.

## Development

From the repository root, using the project virtual environment:

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
```

Projection and geometry tests use synthetic tracks and do not require
Assetto Corsa. See [architecture](docs/architecture.md) for conventions and
limitations.

The simulator-independent projection API is:

```python
from assetto_corsa_ai.track_projection import TrackProjector

# track_geometry: TrackGeometry; position: (world_x, world_y, world_z)
projector = TrackProjector(track_geometry, closed=True)
projection = projector.project(position)
print(projection.lateral_offset, projection.normalized_track_position)
```

With Assetto Corsa running on Monza, the validation tool shows the spline,
road edges, vehicle, tyre contact points and projection. A purple section
marks progress from the spline start to the projected point; a dashed line
connects the actual vehicle position to its projection. The readout shows
progress in metres and percent, signed lateral margins and Assetto Corsa's
telemetry progress for comparison. Geometric progress can differ from the
simulator's value (see the architecture notes).

```powershell
python tools/validate_track_alignment.py
```

Adjust `track_file` in the tool to match your local installation and track.
The plot follows the vehicle with a 60 m by 60 m view. Adjust
`VIEW_RADIUS_METRES` in the tool for a closer or wider view; smaller values
zoom in. The vehicle stays centred with its front pointing up; the track
rotates with the vehicle's orientation. Orientation comes from the front
and rear tyre contact points, so stopping or reversing does not flip the
view. If the contact points cannot define an orientation, the last valid
orientation is retained (initially world -Z). Only the display is transformed;
telemetry and projection calculations keep native world coordinates.

To inspect damage while driving manually:

```powershell
python tools/validate_damage.py
```

The tool prints all five raw damage channels, their signed changes since the
previous sample and the largest positive channel change observed during the run.
The first sample establishes a baseline. Compare normal driving, light contacts
and strong impacts to calibrate a future collision threshold; no threshold is
applied yet. Stop with `Ctrl+C`.
