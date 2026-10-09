# Software Architecture

## Overview

The project separates Assetto Corsa integration from simulator-independent
geometry and future reinforcement learning logic.

The current scope covers telemetry, track geometry and projection, observation
features, rewards, termination checks, vehicle control and session launch.
Gymnasium integration and training will be added incrementally.

The session manager prepares and launches AC; the simulator controller handles
runtime commands through CSP. Telemetry remains a separate read interface.
See [vehicle control](control.md) for the start/reset concepts and live validation.


## Coordinates and Visualization

Telemetry and geometry retain Assetto Corsa's native world coordinates
`(x, y, z)`. Projection operates in 3D; coordinate transformations belong
at explicit system boundaries.

[`tools/validate_track_alignment.py`](../tools/validate_track_alignment.py)
visually compares the vehicle position with its spline projection and
displays progress and edge margins. Its zoomed X/Z view follows the vehicle,
with the vehicle's front always pointing up.

Translation and rotation affect only the display. They do not change
telemetry, track geometry or projection results. Implementation details and
configuration are documented in the respective source files.
