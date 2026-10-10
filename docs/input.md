# Input / Read Layer

The AC adapters translate shared-memory pages and track files into typed domain
data. Observations, geometry and reward calculations consume those models
without parsing simulator formats themselves.

```mermaid
flowchart TD

    AC[Assetto Corsa<br/>Simulation / Game]

    subgraph DS["Data Sources"]
        Physics[Physics Shared Memory]
        Graphics[Graphics Shared Memory]
        Static[Static Shared Memory]
        AI[fast_lane.ai]
    end

    subgraph ACA["Assetto Corsa Adapters"]
        TR[Telemetry Reader<br/>telemetry_reader.py]
        SR[Static Info Reader<br/>static_info_reader.py]
        TrackReader[Track Reader<br/>track_reader.py]
    end

    subgraph DD["Domain Data"]
        TF[Telemetry Frame<br/>telemetry.py]
        SI[Static Info<br/>static_info.py]
        TG[Track Geometry<br/>track_geometry.py]
    end

    AC --> Physics
    AC --> Graphics
    AC --> Static
    AC --> AI

    Physics --> TR
    Graphics --> TR
    Static --> SR
    AI --> TrackReader

    TR --> TF
    SR --> SI
    TrackReader --> TG
```

## Data Sources and Adapters

| Source | Purpose | Adapter |
| --- | --- | --- |
| Physics and graphics shared memory | Runtime vehicle state and world position. | [TelemetryReader](../src/assetto_corsa_ai/assetto_corsa/telemetry_reader.py) |
| Static shared memory | Vehicle, track and session metadata. | [StaticInfoReader](../src/assetto_corsa_ai/assetto_corsa/static_info_reader.py) |
| `fast_lane.ai` | Sampled reference line and track widths. | [AssettoCorsaTrackReader](../src/assetto_corsa_ai/assetto_corsa/track_reader.py) |

## Domain Data

[TelemetryFrame](../src/assetto_corsa_ai/telemetry.py) represents the current
vehicle state. [SimulatorStaticInfo](../src/assetto_corsa_ai/static_info.py)
holds session metadata. [TrackGeometry](../src/assetto_corsa_ai/track_geometry.py)
stores the sampled reference line and boundary information.

Memory layouts, field mappings and file parsing remain in the linked adapters.
[TrackProjector](../src/assetto_corsa_ai/track_projection.py) derives progress
and lateral offset from geometry; [observation construction](observation.md)
then combines vehicle and track features.
