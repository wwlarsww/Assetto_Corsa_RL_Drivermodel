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

The current implementation focuses only on reading telemetry
from Assetto Corsa.

Controller integration, Gymnasium and reinforcement learning
will be added later.