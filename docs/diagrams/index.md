# Class diagrams

These diagrams are generated from the Python source at every documentation
build using **Pyreverse**, with additional class-symbol references from the
same static syntax trees. They cover public classes in `src/` and `tools/`.
No simulator session, model or Qt application is started.

## Explore dependencies

Use the generated class index below to select a **class** or **module**.
Each view shows both its direct dependencies and the classes depending on it.
The selected class or module's classes are highlighted.
Follow the class links below a diagram to move to another class's neighborhood,
or open its API reference for methods, signatures and source code.

Start with:

- [RacingEnvironment](classes/assetto_corsa_ai.RacingEnvironment.RacingEnvironment.md):
  simulator adapters, observations, reward and episode lifecycle.
- [PPOAgent](classes/assetto_corsa_ai.agent.PPOAgent.md):
  environment, PPO configuration and learning components.
- [ObservationBuilder](classes/assetto_corsa_ai.observation_builder.ObservationBuilder.md):
  vehicle state, tyre dynamics and track perception.
- [TrainingConfig](classes/assetto_corsa_ai.training_config.TrainingConfig.md):
  the experiment configuration sections.
- [LiveSource](classes/tools.diagnostics.sources.LiveSource.md):
  independent live telemetry, model inference and diagnostic state.

## Reading the arrows

| Relationship | Meaning |
| --- | --- |
| `Child --\|> Parent` | Child inherits from Parent. |
| `Owner --> Type` | An associated field refers to Type. |
| `Owner o-- Type` | Pyreverse infers a received object stored by Owner. |
| `Owner *-- Type` | Pyreverse infers an object created and stored by Owner. |
| `User ..> Type` | User directly references the class symbol in its body or annotations. |

Diamonds are on the owner side. Field labels identify stored attributes,
including private fields such as `_reward_calculator`. The relationship table
always lists the dependent class first, independently of the arrowhead's shape.

Use **Zoom in**, **Zoom out**, **Fit** and **100%** to explore large diagrams.
Scroll inside the diagram to reach off-screen classes. **Export SVG** saves a
vector copy for sharing.

Class boxes show names only to keep dependency views readable. Attribute and
method details remain in the linked API reference. A view contains one hop;
it does not expand every neighbor transitively.

## Scope and limitations

This is a static view of code relationships, not a runtime call graph or a
proof of object lifetime. Inferred ownership can be incomplete. Factory
callbacks, dynamically selected classes and unannotated objects may hide
dependencies. Unused imports alone do not create class relationships.
External library classes and private helper classes are omitted.

New public classes are discovered automatically. Generated pages are build
artifacts; improve annotations and source code rather than editing diagrams.
See [documentation and publishing](../documentation.md) for the build process.
