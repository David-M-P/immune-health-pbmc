# TRIPSO Provenance

The repository contains complete imported working trees from two third-party upstream repositories. The imports are ordinary files, not submodules, subtrees, or nested repositories.

| Repository | Upstream | Original branch | Imported commit | Commit date | Vendored path | License | Original tracked files | Original size |
|---|---|---|---|---|---|---|---:|---:|
| TRIPSO | https://github.com/Lotfollahi-lab/tripso | `main` | `5d19c88081b1a0c497fb6dc4637df063e7782a3a` | `2026-03-26T15:36:08Z` | `tripso_code/tripso` | MIT, `tripso_code/tripso/LICENSE` | 85 | 96 MiB including Git metadata; 50.55 MiB working tree |
| TRIPSO reproducibility | https://github.com/Lotfollahi-lab/tripso_reproducibility | `main` | `26d4f90d8474f4baa8511ef5ce12a0f2624eb06a` | `2026-05-15T12:10:51+01:00` | `tripso_code/tripso_reproducibility` | MIT, `tripso_code/tripso_reproducibility/LICENSE` | 79 | 203 MiB including Git metadata; 123.02 MiB working tree |

Both working trees were imported on `2026-07-10`. Their original Git metadata is preserved locally, outside this repository, at:

- `../.repo_setup_backups/20260710_110919/tripso.git`
- `../.repo_setup_backups/20260710_110919/tripso_reproducibility.git`

TRIPSO and the TRIPSO reproducibility repository are third-party projects. Vendoring their source does not transfer ownership, and their upstream licenses continue to apply. The original LICENSE files remain inside their vendored directories.

The nested upstream Git histories are not embedded in the parent repository. Future modifications made in this project will be recorded in the main repository history. The upstream origin URLs and exact imported commits are preserved here for reproducibility.
