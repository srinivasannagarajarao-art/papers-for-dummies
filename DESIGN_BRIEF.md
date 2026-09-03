# Design Brief — Papers for Dummies

You are a **frontend designer and engineer** specializing in editorial, content-first sites for technical audiences.

I have an existing static site: **Papers for Dummies**, a personal knowledge-sharing site where an AI Engineer explains research papers to other engineers — programmer-framing, not academic framing. It currently has one paper live (*Attention Is All You Need*), a homepage, and a papers index. Plain HTML/CSS, no build step, no backend, hosted on GitHub Pages. It grows one paper page at a time, written by hand.

Redesign the visual language of the **pages that exist today**. Do not invent product surfaces this site doesn't have — no accounts, no database-backed search, no citation graphs. This is a personal essay site with excellent typography, not a research platform.

---

## 1. WHAT THIS SITE ACTUALLY IS

A single author explaining ML papers as runnable code, for engineers who'd rather debug something than read a proof. Every paper page follows the same shape: the idea as a metaphor a programmer already knows, the real formula, real printed output from running code (not illustrative numbers), a "break this on purpose" table, and a closing note on what maths is actually needed.

The audience is one person: a working engineer, self-taught, skeptical of hype, who wants the idea distilled honestly. Not a research audience. Not a lead-gen audience. Not a SaaS audience.

Pages that exist:
- **Homepage** — bio + list of papers (1 live, N "coming soon")
- **Papers index** — same list, standalone
- **Paper page** (Attention) — the essay, in English/Tamil/Hindi via in-page tabs, runnable Python script alongside

Design the system so a new paper page is easy to add by copying the template — no CMS, no data layer, just a new HTML file.

---

## 2. DESIGN PHILOSOPHY

The site should feel like:

> "A programmer's notebook, typeset properly."

Avoid:
- Dashboard chrome (stat tiles, sidebars, nav rails) — there's nothing to dashboard
- Card grids pretending to be a catalog of hundreds of items when there are 3
- Generic SaaS gradients or glassmorphism
- Academic-portal density (dense tables, citation counts, venue metadata) — this isn't indexing a corpus
- Anything implying scale or a team ("Trending," "Collections," "42 emerging topics") that would be fabricated for a one-author site with one paper

Prefer:
- Generous whitespace, strong type hierarchy, restrained color
- The paper list as a short editorial index, not a marketplace grid
- Code blocks and terminal-style "real output" blocks that are visually distinct from each other — the proof-by-running-it is the whole pedagogical point
- Clear treatment for the three language tabs (EN/TA/HI) — Tamil and Devanagari need real font stacks, not fallback boxes
- Motion only where it clarifies (tab switching, hover on the one or two link types) — nothing decorative

Think **"personal essay site meets well-typeset developer docs."**

---

## 3. VISUAL LANGUAGE

**Color** — one neutral base, one accent, both light and dark mode:
- Background: warm off-white (light) / near-black (dark), not pure white/black
- Text: near-black / soft white; secondary text muted gray in both
- Borders: extremely subtle
- One accent color, used sparingly (links, the active language tab, headings) — pick one, don't add "category colors," there's no category system
- No gradients as a design element (the current draft uses a gradient hero band — fine to keep or drop, but don't multiply it elsewhere)

**Typography** — this matters more than any other decision:
- One premium sans for UI and body (Inter, Geist, IBM Plex Sans, or similar)
- Monospace for all code and terminal-output blocks — and the terminal-output blocks should look distinct from source-code blocks (they're evidence, not source)
- A serif accent is optional and should be used sparingly if at all — this is a technical essay, not a magazine
- Tamil needs `Noto Sans Tamil` (or equivalent) with real fallbacks; Hindi needs `Noto Sans Devanagari` — load both, test both render, don't let them silently box on Windows

---

## 4. HOMEPAGE

- Short bio, first person, honest about being self-taught ("I'm not a researcher, I'm a programmer")
- One list: the papers, each entry showing title, one-line pitch, and which languages it's available in
- "Coming soon" entries stay visually muted/disabled — don't fake availability
- No search bar (3 items don't need search), no filters, no "explore by discipline" — there's one discipline so far

---

## 5. PAPER LIST (papers index)

Same list as the homepage, as its own page for direct linking. Each entry:

- Paper name + authors/year
- One-sentence pitch in plain language (not an abstract)
- Which languages it has
- A single word for status: live or coming soon

No citation counts, no venue, no reading-time estimate unless it's true and you'd actually compute it (word count / 200wpm is fine; don't invent "18 min read").

---

## 6. PAPER PAGE TEMPLATE

This is the core surface. Structure, in order:

1. **Title + one-line framing** of why this paper, why this angle
2. **Language switcher** — tabs, not separate URLs, so a reader can flip mid-read; each language is a full independent write, not a partial translation
3. **The metaphor** — the paper's core idea recast as something a programmer already knows
4. **The real formula/notation**, presented plainly, not hidden behind more metaphor
5. **Runnable code** — the actual working implementation, downloadable
6. **Real output blocks** — actual printed results from running the code, styled distinctly from source code so "this is evidence" reads at a glance
7. **"Break this on purpose" table** — specific things to change in the code and what breaks, because that's how the reader actually learns it
8. **Honest scope note** — what maths this does and doesn't require, said plainly

No sidebar "about this paper" panel, no related-papers rail (nothing to relate it to yet), no AI chat panel bolted onto the page — if an AI explain-this-paper feature is ever built, it should be a clearly separate, clearly optional add-on, not baked into the reading experience by default.

---

## 7. WHAT NOT TO ADD

Explicitly out of scope until there's real content or infrastructure to justify it:
- Global/command-palette search (nothing to search yet)
- User accounts, saving, bookmarking
- Category taxonomy and category landing pages
- Citation counts, venue metadata, author institution badges
- "Trending," "Most cited," "Emerging topics" — all imply a corpus and a team; this is one person
- A papers database or CMS — stay static HTML until the number of papers makes hand-editing genuinely painful (that's likely 15–20+ pages, not 3)

If any of these become real later (more papers, real need for search), redesign for them then, with real content driving the design — not before.

---

## 8. TECHNICAL CONSTRAINTS

- Stay plain HTML/CSS, no build step, no framework — this is a deliberate choice for a personal site with one maintainer
- One shared stylesheet (`assets/style.css`), no per-page style forking
- Every page must work with relative paths (site may be hosted at a domain root or a project subpath)
- New paper page = copy the template file, no other config to touch
- Fast, accessible, semantic HTML — no JS framework needed; the only interactive JS is the language-tab switcher

---

## 9. GOAL

A reader should think: *"this person actually explains things clearly, and I trust what they show me because I can run it myself."*

Not: *"this looks like a funded startup's research platform."*

When in doubt, cut a feature rather than fake the scale to justify it. Three honest papers beat a UI built for three hundred.
