## What changed

<!-- One paragraph: what behavior changed and why. -->

## Branch flow

- [ ] Cut from `dev`
- [ ] Targets `dev` (never `main` directly)

## Checks

- [ ] `uv run pre-commit run --all-files`
- [ ] `uv run pytest`
- [ ] `uv run mypy src`

## Security

- [ ] No credentials, tokens or private keys added
- [ ] Inputs validated and errors do not echo internals
- [ ] Serving changes keep the model pinned (`MODEL_VERSION`)
