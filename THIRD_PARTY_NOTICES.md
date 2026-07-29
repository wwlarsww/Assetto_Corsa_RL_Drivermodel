# Third-party notices

## AssettoCorsaGym

The optional downloader retrieves a subset of the AssettoCorsaGym dataset by
Marconato et al., *A Simulation Benchmark for Autonomous Racing with Large-Scale
Human Data*. The dataset is distributed under CC BY 4.0. Track configuration
files are retrieved from the accompanying MIT-licensed source repository.
Neither the downloaded data nor generated reference artefacts are included in
this Git repository.

- Dataset: <https://huggingface.co/datasets/dasgringuen/assettoCorsaGym>
- Source: <https://github.com/dasGringuen/assetto_corsa_gym>
- Project: <https://assetto-corsa-gym.github.io/>

## MoTeC LD format interoperability

The read-only parser in this project is an original, reduced implementation of
the binary-format facts needed for the ACTI recordings. The public
`gotzl/ldparser` project and the AssettoCorsaGym loader were used as
interoperability references. Their code is not bundled as a dependency or
copied into this repository.

- Format reference: <https://github.com/gotzl/ldparser> (GPL-3.0)

MoTeC and Assetto Corsa are trademarks of their respective owners. This project
is independent and is not endorsed by them.
