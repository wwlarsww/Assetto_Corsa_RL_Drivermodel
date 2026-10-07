## Input/Read Layer

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

### Data Sources

Provide raw simulator and track data.

- **Physics Shared Memory**  
  Runtime vehicle physics such as speed, steering, throttle, brake, acceleration and wheel data.

- **Graphics Shared Memory**  
  Runtime session and vehicle position information.

- **Static Shared Memory**  
  Simulator and vehicle metadata that remains mostly constant during a session.

- **`fast_lane.ai`**  
  Track geometry source used to reconstruct the reference spline and track boundaries.

### Assetto Corsa Adapters

Translate Assetto Corsa specific data formats into internal project models.

- **[`TelemetryReader`](../src/assetto_corsa_ai/assetto_corsa/telemetry_reader.py)**  
  Reads physics and graphics shared memory and creates a `TelemetryFrame`.

- **[`StaticInfoReader`](../src/assetto_corsa_ai/assetto_corsa/static_info_reader.py)**  
  Reads static shared memory and creates `StaticInfo`.

- **[`TrackReader`](../src/assetto_corsa_ai/assetto_corsa/track_reader.py)**  
  Parses `fast_lane.ai` and creates `TrackGeometry`.

### Domain Data

Simulator-independent representations used by the rest of the application.

- **[`TelemetryFrame`](../src/assetto_corsa_ai/telemetry.py)**  
  Represents the current vehicle state.

- **[`StaticInfo`](../src/assetto_corsa_ai/static_info.py)**  
  Contains vehicle, track and simulator metadata.

- **[`TrackGeometry`](../src/assetto_corsa_ai/track_geometry.py)**  
  Represents the sampled track geometry including centerline and boundaries.
   
