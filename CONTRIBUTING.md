# Contributing

## Branch flow

```
feat/*  chore/*  fix/*  ──►  dev  ──►  main
```

- `main` holds released, deployable code. Only `dev` merges into it.
- `dev` is the integration branch. Nothing lands here except a merge from a
  topic branch.
- Work happens on short-lived branches cut from `dev`, one per change:

```bash
git checkout dev && git pull
git checkout -b feat/short-description
# ... work, commit ...
git push -u origin feat/short-description
```

Open a pull request into `dev`. The default template asks for a short
summary, the reason, and how to test. Merge it, then delete the branch.

Prefixes: `feat/` for behavior, `fix/` for defects, `chore/` for tooling and
documentation. Keep one concern per branch.

Releasing means a pull request from `dev` into `main`. Use the release
template by adding `?template=release.md` to the compare URL, and list the
branches it ships.

CI runs on pushes and pull requests targeting both `dev` and `main`. The
`Deploy` workflow runs only from `main`, is bound to the protected `production`
environment and waits for a required reviewer before it touches Cloud Run.
Branch protection blocks direct pushes to `main`, so a release is always a pull
request from `dev`. Every action in these workflows is pinned to a full commit
SHA, and the repository setting `sha_pinning_required` rejects any workflow that
uses a tag instead.

## Commits

Conventional Commits, imperative mood, body explaining what changed and why:

```
fix: answer a refused model with 503 instead of 500
```

## Local checks

```bash
uv sync
uv run pre-commit install          # once per clone
uv run pre-commit run --all-files
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy src
uv run --with pip-audit pip-audit  # dependency audit against OSV.dev
```

`detect-secrets` blocks credentials from being committed. If it flags a false
positive, add the finding to `.secrets.baseline`:

```bash
uvx detect-secrets scan --baseline .secrets.baseline
uvx detect-secrets audit .secrets.baseline
```

If a real secret ever reaches a commit, treat it as compromised: rotate it
first, then rewrite history. Deleting the commit is not enough.

## Secrets and configuration

Credentials live in the environment, never in the repository. Settings are read
from environment variables (see the README table); `config.yaml`-style files are
not used here, and `.env` is git-ignored for local development. CI and deploy
use repository secrets, and `GITHUB_TOKEN` stays scoped to the minimum
permissions each workflow declares.

## Model changes

- Training registers a new immutable model version. Serving refuses floating
  aliases (`latest`, `staging`) unless `ALLOW_UNPINNED_MODEL=1` is set locally.
- Promote a release by setting `MODEL_VERSION` to the version printed by
  training, not by moving aliases.
