---
name: growth-kit
description: "Complete non-development toolkit for taking ideas from concept to market. Use when user needs product validation, market research, marketing content, visual assets, conversion focused copywriting or go-to-market strategy. This crucial master skill routes to specialized sub-skills: idea-to-product, research, marketing, media-creation, crawl-cli."
---

# Growth Kit

Complete non-development toolkit for taking ideas from concept to market. Five modular sub-skills that work together or independently.

**Think of it as:** A fractional CMO + Product Strategist in a box. Figures out what you need before diving into tactics.

---

## When to Use This Orchestrator

**Use this orchestrator when:**

- User doesn't know where to start
- User has a vague goal ("I need marketing", "help me launch")
- User isn't sure which sub-skill applies
- User needs a multi-step workflow
- User wants to audit what's missing

**Skip to sub-skill directly when:**

- User has a specific task ("write a landing page", "generate an image")
- User knows exactly which skill they need
- User is continuing work from a previous skill

---

## The Skill Registry

### Available Sub-Skills

| Skill                | What It Does                       | Inputs Needed                 | Outputs                              |
| -------------------- | ---------------------------------- | ----------------------------- | ------------------------------------ |
| **idea-to-product/** | Validates ideas, plans products    | Business concept              | Strategy doc, implementation plan    |
| **research/**        | Market research, competitive intel | Business context, competitors | Battle cards, segments, positioning  |
| **marketing/**       | Content, copy, email, distribution | Voice, positioning, keywords  | Landing pages, content, sequences    |
| **seo/**             | Complete SEO system                | Primary keyword, site context | Keywords, content, rank optimization |
| **cro/**             | Conversion rate optimization       | Page/funnel to optimize       | A/B tests, copy variants, experiments|
| **pricing/**         | Pricing strategy & page optimization| Product, market, competitors | Tier structure, pricing page copy    |
| **paid-ads/**        | Paid advertising strategy          | Budget, audience, goals       | Ad copy, campaign structure, targeting|
| **growth-loops/**    | Referral & viral mechanics         | Product, customer base        | Referral program, free tool strategy |
| **media-creation/**  | Visual assets, AI image generation | Subject, style, platform      | Hero images, social graphics         |
| **crawl-cli/**       | Web scraping, content extraction   | URLs to crawl                 | Extracted content, documentation     |

### Skill Dependencies

```
DISCOVERY LAYER (start here for new products)
└── idea-to-product (validate before building)

RESEARCH LAYER (understand before executing)
├── research/competitor-analysis
├── research/target-segments
└── research/positioning-strategy

FOUNDATION LAYER (define before creating)
├── marketing/voice-profiler (how you sound)
├── marketing/hook-finder (how you're different)
└── pricing/skill-overview.md (how to price and present)

SEO LAYER (complete SEO system - see seo/SEO-Boost-SKILL.md)
├── seo/workflow-a-audit (audit current state)
├── seo/workflow-x-competitive-intelligence (competitive analysis)
├── SEO Boost API (keyword discovery)
├── seo/workflow-s-content-strategy (cluster and prioritize)
├── seo/workflow-c-content (create optimized content)
├── seo/workflow-r-rank (improve existing rankings)
└── seo/workflow-t-technical (technical SEO fixes)

EXECUTION LAYER (requires foundation + research)
├── marketing/direct-response-copy (needs positioning, voice)
├── marketing/copy-editing (polish existing copy)
├── marketing/email-sequences (needs lead magnet, positioning)
├── marketing/newsletter-templates (needs voice, content)
└── media-creation/ (needs direction, can run anytime)

CRO LAYER (optimize conversions - requires existing pages)
├── cro/page-cro (landing page optimization)
├── cro/form-cro (form optimization)
├── cro/signup-flow-cro (signup flow optimization)
├── cro/onboarding-cro (onboarding optimization)
├── cro/popup-cro (popup/modal optimization)
├── cro/paywall-cro (paywall/upgrade optimization)
└── cro/ab-testing (experiment design and analysis)

PAID LAYER (paid acquisition - requires positioning + budget)
└── paid-ads/skill-overview.md (ad strategy, copy, targeting)

DISTRIBUTION LAYER (transforms execution outputs)
├── marketing/content-repurposer (needs content)
├── marketing/reddit-marketing (needs content, positioning)
├── marketing/twitter-marketing (needs content, positioning)
└── media-creation/social-graphics (needs content to visualize)

GROWTH LOOPS (amplify organic growth)
├── growth-loops/referral-program (customer-driven acquisition)
└── growth-loops/free-tool-strategy (engineering as marketing)

LAUNCH LAYER (coordinates everything)
├── marketing/gtm-playbook
└── marketing/launch-checklists
```

---

## Intake: The Qualifying Questions

Ask these to diagnose the situation:

### Question 1: What phase are you in?

```
A) I have a new idea to validate
B) I need to understand my market
C) I'm ready to create marketing content
D) I need visual assets
E) I'm launching something
F) I need to improve conversions
G) I want to set up paid advertising
H) I need analytics/tracking setup
I) I want to build growth loops (referrals, free tools)
J) Not sure / need help figuring it out
```

**Routing:**

- A → idea-to-product/skill-overview.md (full workflow)
- B → research/skill-overview.md
- C → marketing/skill-overview.md
- D → media-creation/skill-overview.md
- E → Launch workflow (see below)
- F → cro/skill-overview.md (conversion rate optimization)
- G → paid-ads/skill-overview.md (paid advertising)
- H → Use analytics skill (tracking-setup.md moved there)
- I → growth-loops/ (referral-program.md or free-tool-strategy.md)
- J → Continue to Question 2

### Question 2: What do you already have?

```
[ ] Validated product idea / strategy document
[ ] Competitor analysis / market research
[ ] Defined brand voice / how I sound
[ ] Clear positioning / what makes me different
[ ] Keyword strategy / know what to write about
[ ] Lead magnet / opt-in offer
[ ] Landing page(s)
[ ] Content / blog posts
[ ] Email sequences
[ ] Visual assets / brand imagery
```

**Routing:** Fill gaps in order of dependencies (discovery → research → foundation → execution)

### Question 3: What's the immediate need?

```
A) I need to write something specific (copy, content)
B) I need to plan / strategize
C) I need to figure out my messaging
D) I need to understand my audience better
E) I need visual assets
F) I need a complete marketing system
```

**Routing:**

- A → Identify type, route to marketing/ execution skill
- B → idea-to-product/ OR research/
- C → marketing/voice-profiler → marketing/hook-finder
- D → research/target-segments → research/positioning-strategy
- E → media-creation/skill-overview.md
- F → Full sequence starting from gaps

### Question 4: What's your timeline?

```
A) I need something today
B) This week
C) Building for the long term
```

**Routing:**

- A → Single highest-impact skill
- B → 2-3 skill sequence
- C → Full system build

---

## Routing Decision Tree

```
START
  │
  ▼
┌─────────────────────────────┐
│ Is this a NEW product idea? │
└─────────────┬───────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
     YES              NO
      │               │
      ▼               │
┌─────────────┐       │
│ START WITH  │       │
│ idea-to-    │       │
│ product/    │       │
└─────────────┘       │
                      ▼
┌─────────────────────────────┐
│ Do you understand your      │
│ market and competitors?     │
└─────────────┬───────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
     YES              NO
      │               │
      │               ▼
      │         ┌─────────────┐
      │         │ RUN         │
      │         │ research/   │
      │         └─────────────┘
      │
      ▼
┌─────────────────────────────┐
│ Do you have brand voice     │
│ and positioning defined?    │
└─────────────┬───────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
     YES              NO
      │               │
      │               ▼
      │         ┌──────────────────┐
      │         │ RUN marketing/   │
      │         │ 01-foundations/  │
      │         └──────────────────┘
      │
      ▼
┌─────────────────────────────┐
│ What's your primary goal?   │
└─────────────┬───────────────┘
              │
    ┌─────────┼─────────┬─────────┐
    ▼         ▼         ▼         ▼
 TRAFFIC    LEADS    CONVERT   VISUALS
    │         │         │         │
    ▼         ▼         ▼         ▼
keyword-  lead-     direct-   media-creation/
research  magnet    response  skill-overview.md
    │         │         │
    ▼         ▼         ▼
seo-      direct-   email-
content   response  sequences
```

---

## Pre-Built Workflows

### Growth Kit Workflow 1: "I Have a New Product Idea"

**Situation:** New concept, need to validate and plan before building.

**Sequence:**

```
1. idea-to-product/idea-extraction.md
   └── Output: Strategy document + viability score
   └── Input: Business concept

2. idea-to-product/competitive-intel.md
   └── Output: Competitor briefs, How to Win playbook
   └── Input: Strategy document

3. idea-to-product/starter-kit-selection.md
   └── Output: Selected technical foundation
   └── Input: Requirements from strategy

4. idea-to-product/transformation-analysis.md
   └── Output: 7-domain implementation plan
   └── Input: Selected kit + requirements

5. idea-to-product/expert-cofounder.md
   └── Output: Configured AI advisor
   └── Input: All documents from above
```

**Then:** Route to research/ or marketing/ based on next need.

---

### Growth Kit Workflow 2: "I Need to Understand My Market"

**Situation:** Have a product, need competitive intelligence and audience insights.

**Sequence:**

```
1. research/competitor-analysis.md
   └── Output: Battle cards, positioning gaps
   └── Input: Competitor list

2. research/target-segments.md
   └── Output: Segment profiles with messaging
   └── Input: Business context

3. research/positioning-strategy.md
   └── Output: Positioning framework
   └── Input: Competitor gaps + segments
```

**For keyword research:** Use SEO Boost API, then continue to `seo/workflow-s-content-strategy.md`.

---

### Growth Kit Workflow 3: "I'm Starting Marketing From Zero"

**Situation:** Need to build marketing foundation and start creating.

**Sequence:**

```
1. marketing/01-foundations/voice-profiler.md
   └── Output: Voice profile document
   └── Input: Content samples OR strategic answers

2. marketing/01-foundations/hook-finder.md
   └── Output: Differentiated positioning + hooks
   └── Input: Product, audience, competitors

3. SEO Boost API + seo/workflow-s-content-strategy.md
   └── Output: Prioritized keyword clusters + content plan
   └── Input: Positioning context

4. marketing/02-content-creation/lead-magnet-builder.md
   └── Output: Lead magnet concept
   └── Input: Positioning + audience pain points

5. marketing/02-content-creation/direct-response-copy.md
   └── Output: Landing page copy
   └── Input: Voice + positioning + lead magnet
```

---

### Growth Kit Workflow 4: "I Need Leads"

**Situation:** Have product, need to build email list and convert.

**Sequence:**

```
1. marketing/01-foundations/hook-finder.md (if positioning unclear)
   └── Find differentiated hook

2. marketing/02-content-creation/lead-magnet-builder.md
   └── Create compelling opt-in offer

3. marketing/02-content-creation/direct-response-copy.md
   └── Write landing page for lead magnet

4. marketing/03-email-marketing/email-sequences.md
   └── Build welcome sequence that converts
   └── Input: Lead magnet, positioning, voice
```

---

### Workflow 5: "I'm Launching Something"

**Situation:** New product/offer, need launch materials.

**Sequence:**

```
1. marketing/01-foundations/hook-finder.md
   └── Find the launch angle

2. marketing/02-content-creation/lead-magnet-builder.md (if building waitlist)
   └── Create early access incentive

3. marketing/02-content-creation/direct-response-copy.md
   └── Landing page + ad copy

4. marketing/03-email-marketing/email-sequences.md
   └── Launch sequence (6-10 emails)

5. marketing/05-gtm/gtm-playbook.md
   └── Launch strategy and timeline

6. marketing/05-gtm/launch-checklists.md
   └── Execute launch

7. media-creation/social-graphics.md
   └── Launch announcement visuals
```

---

### Workflow 6: "I Need Content Strategy"

**Situation:** Want to build organic traffic, don't know what to write.

**Sequence:**

```
1. marketing/01-foundations/voice-profiler.md (if not defined)
   └── Define how content should sound

2. seo/workflow-a-audit.md (if existing site)
   └── Audit current state, identify opportunities

3. SEO Boost API
   └── Discover keyword opportunities (300-450 keywords)

4. seo/workflow-s-content-strategy.md
   └── Categorize, cluster, and prioritize keywords

5. seo/workflow-c-content.md (repeat for priority keywords)
   └── Create optimized, human-quality content

6. marketing/04-distribution/content-repurposer.md
   └── Distribute across platforms

7. media-creation/social-graphics.md
   └── Create social assets for distribution
```

**For existing content optimization:** Use `seo/workflow-r-rank.md` instead of creating new content.

---

### Workflow 7: "I Need Visual Assets"

**Situation:** Need images, graphics, or visual direction.

**Sequence:**

```
1. media-creation/skill-overview.md
   └── Visual strategy (7-phase process)
   └── Output: Visual direction, concepts

2. media-creation/image-generation.md
   └── Generate core images
   └── Input: Subject, style, platform

3. media-creation/social-graphics.md
   └── Adapt for all platforms
   └── Input: Core images, platform specs
```

**Load when needed:** `media-creation/references/visual-intelligence.md` for visual psychology and anti-generic techniques.

---

### Workflow 8: "My Marketing Isn't Working"

**Situation:** Have marketing, but it's not converting.

**Diagnostic Sequence:**

```
1. Audit current positioning
   └── Run marketing/hook-finder to find gaps

2. Audit current copy
   └── Compare to marketing/direct-response-copy principles

3. Audit current content
   └── Compare to seo/workflow-c-content quality checklist

4. Identify weakest link
   └── Re-run relevant skill with fresh approach
```

---

## The Context Paradox (Critical)

**More input doesn't always mean better output.**

This is counterintuitive but essential: sometimes running a skill with LESS context produces better results than running it with everything from previous skills.

### Why This Happens

**1. Information Overload**
When Claude has too much context, it tries to incorporate everything. Output becomes:

- Hedged and committee-sounding
- Overly comprehensive (loses punch)
- Trying to please all inputs (pleases none)

**2. Conflicting Signals**
Different skills optimize for different things:

- Research is broad and inclusive
- Direct response copy needs narrow focus and conviction
- Feeding broad research into narrow copy = diluted output

**3. Loss of Boldness**
Great copy has conviction. Too much research creates:

- "On one hand... on the other hand..."
- Qualifiers and hedges
- Safe, forgettable messaging

### The Rule: Selective Context Passing

```
FULL CONTEXT (pass everything)
├── voice-profiler → ALL other skills (voice should be consistent)
├── hook-finder → direct-response-copy (need the angle)
└── keyword-research → seo-content (need the targets)

LIGHT CONTEXT (pass summary only)
├── hook-finder → lead-magnet-builder (angle + pain points only)
├── keyword-research → direct-response-copy (main keyword only)
├── competitive-intel → marketing (key gaps only)
└── target-segments → marketing (primary segment only)

FRESH START (don't pass, run clean)
├── When previous output feels off
├── When you want a different angle
├── When output is getting worse, not better
└── When you need bold, not comprehensive
```

### The Compression Principle

When passing between skills, compress:

| From                | Pass This                     | Not This                  |
| ------------------- | ----------------------------- | ------------------------- |
| idea-extraction     | Core strategy (1 page)        | Full interview transcript |
| competitive-intel   | Key differentiators (bullets) | All competitor details    |
| voice-profiler      | Voice summary (3 sentences)   | Full profile              |
| hook-finder         | Winning angle (1-2 sentences) | All options               |
| keyword-research    | Priority cluster + 5 keywords | Full spreadsheet          |
| lead-magnet-builder | Hook + format                 | Full concept doc          |
| seo-content         | Key insights (bullets)        | Full article              |

### Signs of Over-Contexting

Output is suffering from too much context when:

- Sentences have multiple qualifiers
- Copy tries to address multiple audiences
- Headlines are long and compound
- CTAs have multiple value propositions
- Reads like a committee wrote it
- Lost the "one person talking to one person" feel

**Fix:** Strip context back to essentials, run again.

### When to Run Fresh

Run execution skills **without** full context when:

- Output from chained skills feels generic or hedged
- You want bold, opinionated copy
- Previous skill output was mediocre
- You're testing a different angle
- Copy needs to feel human, not researched-to-death

**Fresh start prompt:**

> "Write [asset type] for [offer]. Target: [one sentence]. Angle: [one sentence]. Ignore everything else. Be bold."

---

## State Tracking

After each skill runs, record:

```markdown
## Growth Kit Progress

### Discovery

- [ ] Product idea validated: [yes/no]
- [ ] Implementation plan: [exists/missing]

### Research

- [ ] Competitor analysis: [exists/missing]
- [ ] Target segments: [exists/missing]
- [ ] Positioning strategy: [exists/missing]
- [ ] Keyword clusters: [exists/missing]

### Foundation

- [ ] Brand voice profile: [exists/missing]
- [ ] Positioning/hooks: [exists/missing]

### Execution

- [ ] Lead magnet concept: [exists/missing]
- [ ] Landing page(s): [exists/missing]
- [ ] Content pieces: [count]
- [ ] Email sequences: [exists/missing]
- [ ] Newsletter setup: [exists/missing]

### Distribution

- [ ] Social content: [exists/missing]
- [ ] Reddit presence: [exists/missing]
- [ ] Twitter presence: [exists/missing]

### Visual

- [ ] Hero images: [exists/missing]
- [ ] Social graphics: [exists/missing]

### What to Build Next

Based on gaps: [recommendation]
```

---

## Handoff Protocol

When routing to a sub-skill, provide:

### Context Block Format

```markdown
## Orchestrator Handoff

**Goal:** [User's stated goal]
**Current state:** [What exists]
**This skill's job:** [Specific outcome needed]

**Inputs available:**

- Voice profile: [yes/no, summary if yes]
- Positioning: [yes/no, angle if yes]
- Keywords: [yes/no, priority cluster if yes]
- Research: [yes/no, key insight if yes]

**After this skill:** [What comes next in sequence]
```

### Example Handoff to marketing/direct-response-copy

```markdown
## Orchestrator Handoff

**Goal:** Convert visitors to trial signups for SaaS product
**Current state:** Has voice profile, has positioning ("speed over features" angle)
**This skill's job:** Write landing page copy that converts

**Inputs available:**

- Voice profile: Yes (direct, no-fluff, developer-friendly)
- Positioning: Yes ("Ship in days, not months")
- Keywords: No (not SEO-focused landing page)
- Research: Yes (competitors focus on feature lists)

**After this skill:** Route to media-creation/social-graphics for launch visuals
```

---

## The Orchestrator Conversation

### Opening

"Before we dive into tactics, let me understand your situation.

**What phase are you in?**

1. New idea to validate
2. Need to understand my market
3. Ready to create marketing content
4. Need visual assets
5. Launching something
6. Not sure / need help figuring it out"

### Follow-Up Based on Answer

**If unclear:** "What do you already have in place? (validated idea, research, voice, positioning, content, landing pages)"

**If phase is clear:** "Do you have [prerequisite for that phase] defined? If not, we should start there."

### Recommendation Format

"Based on what you've told me, here's what I recommend:

**Immediate:** [Skill/file] - [Why this first]

**Then:** [Skill/file] - [What this builds on]

**After that:** [Skill/file] - [End result]

Want to start with [first skill]? I'll need [inputs required]."

---

## Quick Routing Reference

### By Goal

| Goal                | First Skill                      | Then                            | Then               | Then               |
| ------------------- | -------------------------------- | ------------------------------- | ------------------ | ------------------ |
| Validate idea       | idea-to-product/                 | research/                       | marketing/         | -                  |
| Get traffic         | SEO Boost API                    | seo/workflow-s-content-strategy | seo/workflow-c     | content-repurposer |
| Get leads           | lead-magnet-builder              | direct-response-copy            | email-sequences    | -                  |
| Launch product      | hook-finder                      | direct-response-copy            | email-sequences    | gtm-playbook       |
| Build authority     | voice-profiler                   | seo/workflow-c OR newsletter    | content-repurposer | -                  |
| Create visuals      | media-creation/skill-overview.md | image-generation                | social-graphics    | -                  |
| Fix messaging       | hook-finder                      | direct-response-copy            | -                  | -                  |
| Improve rankings    | seo/workflow-r-rank              | -                               | -                  | -                  |
| Analyze competitors | seo/workflow-x                   | seo/workflow-s (for gaps)       | seo/workflow-c     | -                  |
| Improve conversions | cro/page-cro                     | cro/ab-testing                  | analytics/tracking | -                  |
| Optimize pricing    | pricing/skill-overview.md                 | cro/paywall-cro                 | -                  | -                  |
| Run paid ads        | paid-ads/skill-overview.md       | analytics skill                 | cro/page-cro       | -                  |
| Build referrals     | growth-loops/referral-program    | marketing/email-sequences       | analytics skill    | -                  |
| Build free tool     | growth-loops/free-tool-strategy  | seo/workflow-c-content          | analytics skill    | -                  |

### By What's Missing

| Missing                            | Run This                                |
| ---------------------------------- | --------------------------------------- |
| Don't know if idea is viable       | idea-to-product/idea-extraction         |
| Don't understand competitors       | research/competitor-analysis            |
| Don't know my audience             | research/target-segments                |
| Don't know how to sound            | marketing/voice-profiler                |
| Don't know what makes me different | marketing/hook-finder                   |
| Don't know what to write about     | SEO Boost API + Workflow S              |
| Don't have an opt-in offer         | marketing/lead-magnet-builder           |
| Don't have landing pages           | marketing/direct-response-copy          |
| Don't have SEO content             | seo/workflow-c-content                  |
| Content not ranking well           | seo/workflow-r-rank                     |
| Don't have email sequences         | marketing/email-sequences               |
| Don't have visuals                 | media-creation/skill-overview.md        |
| Technical SEO issues               | seo/workflow-t-technical                |
| Need competitive SEO intelligence  | seo/workflow-x-competitive-intelligence |
| Want to find keyword/content gaps  | seo/workflow-x-competitive-intelligence |
| Landing page not converting        | cro/page-cro                            |
| Forms not converting               | cro/form-cro                            |
| Users dropping off in signup       | cro/signup-flow-cro                     |
| Users not activating               | cro/onboarding-cro                      |
| Popups not working                 | cro/popup-cro                           |
| Users not upgrading                | cro/paywall-cro                         |
| Need A/B test ideas                | cro/ab-testing                          |
| Don't know how to price            | pricing/skill-overview.md                        |
| Need paid ad strategy              | paid-ads/skill-overview.md              |
| Need referral program              | growth-loops/referral-program           |
| Want to build a free tool          | growth-loops/free-tool-strategy         |
| No analytics tracking              | Use analytics skill                     |
| Copy needs polish                  | marketing/copy-editing                  |

---

## Reference Loading (Progressive Disclosure)

Load references only when needed:

| Need                     | Load                                                 |
| ------------------------ | ---------------------------------------------------- |
| Writing copy             | `marketing/references/copywriting-frameworks.md`     |
| Editing/polishing copy   | `marketing/02-content-creation/copy-editing.md`      |
| Awareness-stage matching | `marketing/references/awareness-levels.md`           |
| Social platform specs    | `marketing/references/platform-specs-2025.md`        |
| Authentic content        | `marketing/references/human-content-patterns.md`     |
| Reddit tactics           | `marketing/references/reddit-marketing-reference.md` |
| Twitter content formulas | `marketing/references/twitter-content-formulas.md`   |
| Visual psychology        | `media-creation/references/visual-intelligence.md`   |
| SEO reference guide      | `seo/seo-master-reference.md`                        |
| SEO prioritization       | `seo/prioritization-framework.md`                    |
| SEO experiments          | `seo/experiments-tracker.md`                         |
| Marketing psychology     | `marketing/references/marketing-psychology.md`       |
| CRO best practices       | `cro/skill-overview.md`                                       |
| A/B testing methodology  | `cro/ab-testing.md`                                  |
| Pricing strategy         | `pricing/skill-overview.md`                                   |
| Paid advertising         | `paid-ads/skill-overview.md`                         |
| Referral programs        | `growth-loops/referral-program.md`                   |
| Free tool strategy       | `growth-loops/free-tool-strategy.md`                 |

---

## Anti-Patterns

### Don't:

- Jump to tactics without diagnosis
- Run execution skills without foundation (voice, positioning)
- Skip idea validation for "obvious" ideas
- Try to do everything at once
- Assume one skill solves everything
- **Feed everything from every skill into the next** (context overload)
- **Chain skills when output is getting worse** (stop and simplify)

### Do:

- Start with qualifying questions
- Build foundation before execution
- Sequence skills logically
- Track what's been created
- Recommend next steps after each skill
- **Compress context between skills** (essentials only)
- **Run fresh when output feels off** (sometimes less is more)

---

## The Quality Gate

Before passing output to next skill, ask:

1. **Is this output actually good?** If mediocre, don't pass it - garbage in, garbage out
2. **Does the next skill need ALL of this?** Usually no - extract the essence
3. **Will more context help or hurt?** For strategy: help. For execution: often hurt
4. **Would a human strategist share all this?** A good CMO filters, doesn't dump

---

## The Test

Good orchestration means:

1. **User knows where to start** (not overwhelmed)
2. **Skills run in logical order** (dependencies respected)
3. **Outputs feed into next skill** (no wasted work)
4. **Progress is trackable** (what's done, what's next)
5. **End result is coherent** (pieces work together)

If user still feels lost after orchestration, the diagnosis failed.
