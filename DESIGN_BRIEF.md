# Design Brief — Papers for Dummies

You are a **frontend designer and engineer** specializing in editorial, content-first sites for technical audiences.

I have an existing static site: **Papers for Dummies**, a personal knowledge-sharing site where an AI Engineer explains research papers to other engineers — programmer-framing, not academic framing. It currently has one paper live (*Attention Is All You Need*), a homepage, and a papers index. Plain HTML/CSS, no build step, no backend, hosted on GitHub Pages. It grows one paper page at a time, written by hand.

Redesign the visual language of the **pages that exist today**. Do not invent product surfaces this site doesn't have — no accounts, no database-backed search, no citation graphs. This is a personal essay site with excellent typography, not a research platform.

---

## 1. WHAT THIS SITE ACTUALLY IS

A single author explaining ML papers as runnable code, for engineers who'd rather debug something than read a proof. Every paper page follows the same shape: the idea as a metaphor a programmer already knows, the real formula, real printed output from running code (not illustrative numbers), a "break this on purpose" table, and a closing note on what maths is actually needed.

The audience is one person: a working engineer, self-taught, skeptical of hype, who wants the idea distilled honestly. Not a research audience. Not a lead-gen audience. Not a SaaS audience.

Pages that exist:
- **Homepage** — bio + the paper list, grouped by theme
- **Papers index** — same list, standalone
- **Paper pages** — thirty-four of them, each an essay in English, Tamil and Hindi via in-page tabs, with a runnable Python script alongside

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
- Code blocks consistent with the site's one visual style, per §3 — see that section before styling anything
- Clear treatment for the three language tabs (EN/TA/HI) — Tamil and Devanagari need real font stacks, not fallback boxes
- Motion only where it clarifies (tab switching, hover on the one or two link types) — nothing decorative

Think **"personal essay site meets well-typeset developer docs."**

---

## 3. VISUAL LANGUAGE

Decided and implemented (replacing an earlier Cayman-theme pass, which read as "default GitHub Pages"): one narrow typeset column, one type family, no hero band, no chrome. The page's first visual is its subject — on the homepage, the three-line attention function.

**Color** — light only. One brand hue, one interaction hue, two signal hues, and ink:
- Brand: Cayman's green `#159957` and blue `#155799`. The masthead carries the Cayman gradient `linear-gradient(120deg, #155799, #159957)` with white text. Headings are `#159957`. The rails on the hero code, callouts and the active language tab are `#159957`. This is the one thing kept from the earlier Cayman pass, because the author likes it
- Paper `#ffffff`; secondary surface `#f5f6f8` for code and callouts
- Ink `#1b2a3a` for links, paper titles, bold; body `#33424f`; muted `#5f6c79` for meta
- Rules `#e3e7eb`, stronger rules `#c9d0d6`
- Accent `#1e6bb8` (Cayman's link blue) — hover and focus only, never decorative
- Signal red `#b8321a` = "this broke"; signal green `#1f7a3f` = "this held". Used in output blocks (`.hl` / `.ok`) and as the rail on the "break this" column. Nowhere else.
- One code style everywhere: source and evidence share the same pale surface; the signal colours mark evidence

**Typography**:
- IBM Plex Sans (400/500/600, italic 400) for all text; IBM Plex Mono (400/500) for code
- Hindi: IBM Plex Sans Devanagari; Tamil: Noto Sans Tamil (Plex has no Tamil cut). Both with system fallbacks (`Nirmala UI`, `Tamil Sangam MN`, `Devanagari Sangam MN`) so nothing boxes if a web font fails
- Base 17px, measure 42rem (~80 characters), line-height 1.6. Headings 600 weight, ink colour, no colour accenting of single words
- Links are ink with a thin light underline; hover shifts to accent. No blue links
- From 48em (816px) up, code blocks, callouts and rails hang 2.25rem into the gutters so the text inside them stays on the prose margin; the box or rail steps out, the words don't. Code is 0.8rem so an 83-character line fits without scrolling. Below 48em everything stays inside the column and long code lines scroll within their box

**Structure** — reuse these classes, don't rename them:
- `.masthead` — a slim bar on the Cayman gradient, white text: site name left, `Home / Papers / GitHub` right. Same on every page. This is the whole of the "hero"; there is no tall band
- `.main-content` — the single column. `.title-block` opens it: `h1`, optional `.paper-meta` (authors/year), `.lede`
- `.paper-list` / `.paper` — the paper index: `.paper-link` wrapping `.paper-title` + `.paper-meta`, then `.paper-pitch`, then `.paper-langs` (links to `?lang=`, or a single "English" link when that is all there is). `.paper.soon` for unwritten entries: muted, not a link. `.papers-group` is the mono folder heading above each list; with more than a handful of papers the lists are grouped by theme (`ai-ml/transformers/`, `ai-ml/fine-tuning/`, …) in dependency order within a group
- Pages that exist only in English carry no `.langbar` and no tab script. Tabs appear when a second language is written, never before
- A translated page keeps the essay in all three languages inside `<section data-panel="en|ta|hi">`, and leaves everything from the `<hr>` onward (The code, Now break it, On the maths anxiety, Where to go next, footer) in English outside the panels. Understanding in Tamil or Hindi, vocabulary in English: technical terms, library names and code stay in Latin script, and only `#` comments inside code blocks are translated. Printed program output is never translated, only the annotation after an arrow. Tamil carries two or three `.thanglish` romanised recaps after the hardest sections; Hindi carries none
- `.hero-code` — a `pre` promoted to the page's opening visual (homepage only)
- `.langbar` — text tabs with an ink underline on the active one
- `.note`, `.formula`, `.thanglish`, `.tablewrap`, `table.breaks` — content blocks on paper pages
- `.site-footer` — `.site-footer-owner` (who writes this) and `.site-footer-credits` (source, corrections)

Built in `assets/style.css`, applied to `index.html`, `papers/index.html`, `papers/attention/index.html`, and `404.html`. A new paper page copies the attention page's shell.

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
