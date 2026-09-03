# srinivasan.dev

Personal site + **Papers for Dummies** — ML papers explained for working engineers.

Plain HTML/CSS. No build step, no dependencies. Push to `main` and GitHub Pages serves it.

## Local preview

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Adding a paper

1. `mkdir papers/<slug>`
2. Copy `papers/attention/index.html` as a template
3. Add a card to `papers/index.html` and `index.html`

No config to update. The nav is hand-written on purpose.
