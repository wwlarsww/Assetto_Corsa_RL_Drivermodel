# Local data

Downloaded telemetry and generated dataset artefacts live below this directory
and are excluded from Git. Reproduce the initial dataset with:

```bash
python -m pip install -e ".[data]"
python -m assetto_corsa_rl download-data
```

The default download is deliberately narrow:

- track: `ks_barcelona-layout_gp`
- car: `dallara_f317`
- human MoTeC recordings (`.ld`/`.ldx`) and lap metadata (`.csv`) only
- the 1.0 m Barcelona track configuration

SAC rollouts, replay buffers, checkpoints, and preprocessed Python pickle
telemetry are not downloaded. The command writes a content-hash manifest to
`data/manifests/`.

The source dataset is
[AssettoCorsaGym](https://huggingface.co/datasets/dasgringuen/assettoCorsaGym)
by Marconato et al. and is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Keep its attribution
when publishing derived datasets or results. Assetto Corsa game assets are not
part of this repository.
