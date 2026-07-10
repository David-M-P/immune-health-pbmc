# Development Workflow

## Repository flow

```text
Local computer ─┐
                ├── GitHub ── Gefion GitLab mirror ── Gefion clone
AU HPC ─────────┘
```

GitHub `main` is the canonical source of truth. The laptop and Aarhus University HPC both use GitHub as `origin`. Gefion clones from the internal Gefion GitLab mirror, which follows GitHub in one direction. Gefion is normally an execution environment rather than a development source.

Changes should normally be developed and committed locally or on the AU HPC, then pushed to GitHub. Begin each work session by updating the branch:

```bash
git switch main
git pull --ff-only
git switch -c feature/descriptive-name
```

Use short-lived feature or fix branches for meaningful work. Do not maintain permanent machine-specific branches. Machine differences belong in local configuration files, environment definitions, and separate Slurm scripts under `slurm/au/` and `slurm/gefion/`.

Review and publish a change with commands such as:

```bash
git status
git diff
git add .
git commit -m "Describe the change"
git push -u origin feature/descriptive-name
```

TRIPSO is vendored as ordinary files, so any necessary TRIPSO modifications are committed as ordinary changes in the main repository. Do not edit the same branch simultaneously on multiple machines without first pulling and coordinating the changes.
