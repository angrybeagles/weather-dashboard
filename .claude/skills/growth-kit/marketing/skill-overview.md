---
name: growth-kit-marketing
description: "Marketing execution toolkit. Use when user needs landing page copy, SEO content, email sequences, lead magnets, newsletters, Reddit marketing, Twitter threads, or go-to-market strategy. Transforms research into conversion-optimized content."
---

# Marketing Skill Pack

A complete marketing execution skill pack for Claude Code. Transform research into conversion-optimized copy, multi-channel content, and go-to-market execution.

---

## What This Is

A modular system for marketing execution:

- **Foundation** - Brand voice and positioning angles
- **Content** - SEO, direct response, lead magnets
- **Email** - Sequences and newsletters
- **Distribution** - Social repurposing, Reddit, Twitter
- **GTM** - Launch playbooks and checklists

**Think of it as:** A fractional CMO in your Claude Code. Executes marketing tactics based on strategic research.

**Related Skills:**

- `idea-to-product` - For product discovery and validation
- `research` - For competitor analysis, segments, positioning, and keywords
- `media-creation` - For visual assets, brand direction, and image generation
- `crawl-cli` - For web crawling and content extraction

---

## Skill Directory

### 01-foundations/

| Skill              | What It Does                    | Inputs                               | Outputs                        |
| ------------------ | ------------------------------- | ------------------------------------ | ------------------------------ |
| **voice-profiler** | Defines how you sound           | Content samples OR strategic answers | Voice profile document         |
| **hook-finder**    | Finds your differentiated angle | Product, audience, competitors       | Positioning options with hooks |

### 02-content-creation/

| Skill                    | What It Does                | Inputs                   | Outputs                         |
| ------------------------ | --------------------------- | ------------------------ | ------------------------------- |
| **seo-content-planning** | Plans SEO-optimized content | Target keyword, research | Content outline, structure      |
| **seo-content-writing**  | Writes content that ranks   | Planning output          | Publication-ready article       |
| **direct-response-copy** | Writes conversion copy      | Offer, audience, voice   | Landing page/email/ad copy      |
| **lead-magnet-builder**  | Creates opt-in concepts     | Audience, pain points    | Lead magnet concepts with hooks |
| **content-types**        | Content type specifications | Content need             | Type-specific templates         |

### 03-email-marketing/

| Skill                    | What It Does                | Inputs                      | Outputs                              |
| ------------------------ | --------------------------- | --------------------------- | ------------------------------------ |
| **email-sequences**      | Builds converting sequences | Lead magnet, offer, voice   | Welcome/nurture/conversion sequences |
| **newsletter-templates** | Creates newsletter editions | Content, format type, voice | Publication-ready newsletter         |

### 04-distribution/

| Skill                  | What It Does                | Inputs                     | Outputs                          |
| ---------------------- | --------------------------- | -------------------------- | -------------------------------- |
| **content-repurposer** | Turns 1 piece into many     | Blog, newsletter, video    | Platform-native social assets    |
| **reddit-marketing**   | Reddit strategy and tactics | Product, target subreddits | Posts, comments, engagement plan |
| **twitter-marketing**  | Twitter/X viral content     | Product, positioning       | Threads, tweets, engagement plan |

### 05-gtm/

| Skill                 | What It Does            | Inputs            | Outputs                  |
| --------------------- | ----------------------- | ----------------- | ------------------------ |
| **gtm-playbook**      | Launch strategy         | Product, timeline | Launch plan with tactics |
| **launch-checklists** | Ready-to-use checklists | Launch type       | Executable checklist     |

### references/

Supporting reference material - load these when working on related skills:

| Reference                      | Contains                                                        | Use With                                                   |
| ------------------------------ | --------------------------------------------------------------- | ---------------------------------------------------------- |
| **copywriting-frameworks**     | PAS, AIDA, PASTOR, headline formulas, 6 angle discovery methods | direct-response-copy, email-sequences, lead-magnet-builder |
| **awareness-levels**           | Five awareness levels, copy matching by stage                   | seo-content-writing, direct-response-copy, email-sequences |
| **platform-specs-2025**        | Character limits, image sizes, best practices                   | content-repurposer, reddit-marketing                       |
| **human-content-patterns**     | 20 E-E-A-T writing examples, authenticity patterns              | seo-content-writing, newsletter-templates                  |
| **reddit-marketing-reference** | Ban recovery, SEO/GEO tactics, advanced subreddit research      | reddit-marketing                                           |
| **twitter-content-formulas**   | 5 viral formulas, thread structures, power words                | twitter-marketing                                          |

**When to load references:**

- Writing copy? Load `copywriting-frameworks` + `awareness-levels`
- Creating SEO content? Load `awareness-levels` + `human-content-patterns`
- Repurposing content? Load `platform-specs-2025`
- Reddit marketing? Load `reddit-marketing-reference` for advanced tactics
- Twitter marketing? Load `twitter-content-formulas` for viral formulas

---

## Skill Dependencies

```
FOUNDATION LAYER (do these first if missing)
├── voice-profiler (how you sound)
└── hook-finder (how you're different)

RESEARCH LAYER (use research skill)
├── See research skill for:
│   ├── competitor-analysis
│   ├── target-segments
│   ├── positioning-strategy
│   └── keyword-research

EXECUTION LAYER (requires foundation + research)
├── seo-content-planning + seo-content-writing (needs keywords from research)
├── direct-response-copy (needs positioning, voice)
├── lead-magnet-builder (needs segments from research, positioning)
├── newsletter-templates (needs voice, content)
└── email-sequences (needs lead magnet, positioning, voice)

DISTRIBUTION LAYER (transforms execution outputs)
├── content-repurposer (needs content to repurpose)
├── reddit-marketing (needs content, positioning)
└── twitter-marketing (needs content, positioning)

LAUNCH LAYER (coordinates everything)
├── gtm-playbook (needs product, positioning)
└── launch-checklists (needs launch type)
```

**Cross-Skill Integration:**

- Start with `idea-to-product` for new products
- Use `research` skill for competitor analysis, segments, and keywords
- Then execute with this marketing skill

---

## Integration with Research Skill

The marketing skill depends on outputs from the research skill:

| From research        | To marketing         | What transfers      |
| -------------------- | -------------------- | ------------------- |
| competitor-analysis  | direct-response-copy | Competitive claims  |
| target-segments      | lead-magnet-builder  | Segment pain points |
| positioning-strategy | hook-finder          | Positioning angles  |
| keyword-research     | seo-content-planning | Priority keywords   |

**Workflow:** Research first, then marketing execution.

---

## Quick Start: Routing by Goal

### "I'm Starting From Zero"

**Sequence:**

```
1. voice-profiler
   Output: Voice profile

2. hook-finder
   Output: Differentiated positioning

3. (research skill) keyword-research
   Output: Prioritized content plan

4. lead-magnet-builder
   Output: Lead magnet concept

5. direct-response-copy
   Output: Landing page copy
```

---

### "I Need Leads"

**Sequence:**

```
1. hook-finder (if not clear)
   Find differentiated hook

2. lead-magnet-builder
   Create compelling opt-in offer

3. direct-response-copy
   Write landing page for lead magnet

4. email-sequences
   Build welcome sequence that converts
```

---

### "I Need Content Strategy"

**Sequence:**

```
1. voice-profiler (if not defined)
   Define how content should sound

2. (research skill) keyword-research
   Identify priority topics and clusters

3. seo-content-planning + seo-content-writing (repeat)
   Create optimized content pieces

4. content-repurposer
   Distribute across platforms
```

---

### "I'm Launching Something"

**Sequence:**

```
1. hook-finder
   Find the launch angle

2. lead-magnet-builder (if building waitlist)
   Create early access incentive

3. direct-response-copy
   Landing page + ad copy

4. email-sequences
   Launch sequence (6-10 emails)

5. gtm-playbook + launch-checklists
   Execute launch
```

---

### "I Want to Start a Newsletter"

**Sequence:**

```
1. voice-profiler
   Define newsletter voice

2. hook-finder
   Find unique angle

3. newsletter-templates
   Choose format + create template + first editions
```

---

### "My Marketing Isn't Working"

**Diagnostic Sequence:**

```
1. Audit positioning
   Run hook-finder to find gaps

2. Audit copy
   Compare to direct-response-copy principles

3. Audit content
   Compare to seo-content quality checklist

4. Identify weakest link
   Re-run relevant skill with fresh approach
```

---

## Quick Routing Reference

### By Goal

| Goal                  | First Skill         | Then                      | Then               | Then         |
| --------------------- | ------------------- | ------------------------- | ------------------ | ------------ |
| Get traffic           | (research) keywords | seo-content-\*            | content-repurposer | -            |
| Get leads             | lead-magnet-builder | direct-response-copy      | email-sequences    | -            |
| Launch product        | hook-finder         | direct-response-copy      | email-sequences    | gtm-playbook |
| Build authority       | voice-profiler      | seo-content OR newsletter | content-repurposer | -            |
| Start newsletter      | voice-profiler      | newsletter-templates      | -                  | -            |
| Fix messaging         | hook-finder         | direct-response-copy      | -                  | -            |
| Convert subscribers   | email-sequences     | -                         | -                  | -            |
| Maximize distribution | content-repurposer  | reddit-marketing          | -                  | -            |

### By What's Missing

| Missing                            | Run This                       |
| ---------------------------------- | ------------------------------ |
| Don't know how to sound            | voice-profiler                 |
| Don't know what makes me different | hook-finder                    |
| Don't know what to write about     | (research) keyword-research    |
| Don't have an opt-in offer         | lead-magnet-builder            |
| Don't have landing pages           | direct-response-copy           |
| Don't have content                 | seo-content-planning + writing |
| Don't have email editions          | newsletter-templates           |
| Don't have email sequences         | email-sequences                |
| Don't have social distribution     | content-repurposer             |
| Not on Reddit                      | reddit-marketing               |

---

## Context Passing Between Skills

### The Rule: Selective Context

Not all information should flow between skills. Too much context produces generic, hedged output.

```
FULL CONTEXT (pass everything)
├── voice-profiler → ALL other skills (voice should be consistent)
├── hook-finder → direct-response-copy (need the angle)
└── (research) keyword-research → seo-content-* (need the targets)

LIGHT CONTEXT (pass summary only)
├── hook-finder → lead-magnet-builder (angle + pain points only)
├── (research) keyword-research → direct-response-copy (main keyword only)
└── lead-magnet-builder → newsletter (concept only)

FRESH START (don't pass, run clean)
├── When previous output feels off
├── When you want a different angle
├── When output is getting worse
└── When you need bold, not comprehensive
```

### Compression Principle

When passing between skills, compress:

| From                | Pass This                     | Not This         |
| ------------------- | ----------------------------- | ---------------- |
| voice-profiler      | Voice summary (3 sentences)   | Full profile     |
| hook-finder         | Winning angle (1-2 sentences) | All options      |
| (research) keywords | Priority cluster + 5 keywords | Full spreadsheet |
| lead-magnet-builder | Hook + format                 | Full concept doc |
| seo-content-writing | Key insights (bullets)        | Full article     |

---

## State Tracking

After each skill runs, record:

```markdown
## Marketing Assets Status

### Foundation

- [ ] Brand voice profile: [exists/missing]
- [ ] Positioning/differentiation: [exists/missing]

### Research (from research skill)

- [ ] Competitor analysis: [exists/missing]
- [ ] Target segments: [exists/missing]
- [ ] Keyword clusters: [exists/missing]

### Execution

- [ ] Lead magnet concept: [exists/missing]
- [ ] Landing page(s): [exists/missing]
- [ ] Content pieces: [count]
- [ ] Newsletter setup: [exists/missing]
- [ ] Email sequences: [exists/missing]

### Distribution

- [ ] Social content: [exists/missing]
- [ ] Reddit presence: [exists/missing]

### What to Build Next

Based on gaps: [recommendation]
```

---

## Handoff Protocol

When routing to a skill, provide:

```markdown
## Skill Handoff

**Goal:** [User's stated goal]
**Current state:** [What exists]
**This skill's job:** [Specific outcome needed]

**Inputs available:**

- Brand voice: [yes/no, summary if yes]
- Positioning: [yes/no, angle if yes]
- Keywords: [yes/no, priority cluster if yes]

**After this skill:** [What comes next in sequence]
```

---

## Anti-Patterns to Avoid

### Don't:

- Jump to tactics without diagnosis
- Run execution skills without foundation (voice, positioning)
- Skip the research skill when you need data
- Try to do everything at once
- Assume one skill solves everything
- Feed everything from every skill into the next (context overload)
- Chain skills when output is getting worse (stop and simplify)

### Do:

- Start with qualifying questions
- Build foundation before execution
- Use research skill for data needs
- Sequence skills logically
- Track what's been created
- Recommend next steps after each skill
- Compress context between skills (essentials only)
- Run fresh when output feels off (sometimes less is more)

---

## The Test

Good orchestration means:

1. **User knows where to start** (not overwhelmed)
2. **Skills run in logical order** (dependencies respected)
3. **Outputs feed into next skill** (no wasted work)
4. **Progress is trackable** (what's done, what's next)
5. **End result is coherent** (pieces work together)

If you still feel lost after using this skill pack, start with the diagnostic questions at the top of this file.
