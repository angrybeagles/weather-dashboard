# Onboarding CRO

## Overview

Optimize post-signup onboarding, user activation, and first-run experience to help users reach their "aha moment" quickly and establish habits for long-term retention.

## When to Use

- "Onboarding flow", "activation rate", "user activation"
- "First-run experience", "empty states"
- "Onboarding checklist", "aha moment"
- "New user experience", "time-to-value"
- For signup/registration, use `signup-flow-cro.md`
- For ongoing email sequences, use `marketing/03-email-marketing/email-sequences.md`

## Inputs Required

- What happens immediately after signup
- Definition of "aha moment" / activation event
- Current activation rate
- Where users currently drop off

---

## Core Principles

### 1. Time-to-Value Is Everything

- How quickly can someone experience core value?
- Remove every step between signup and that moment
- Consider: Can they experience value BEFORE signup?

### 2. One Goal Per Session

- Don't try to teach everything at once
- Focus first session on one successful outcome
- Save advanced features for later

### 3. Do, Don't Show

- Interactive > Tutorial
- Doing the thing > Learning about the thing
- Show UI in context of real tasks

### 4. Progress Creates Motivation

- Show advancement
- Celebrate completions
- Make the path visible

---

## Defining Activation

### Find Your Aha Moment

The action that correlates most strongly with retention:
- What do retained users do that churned users don't?
- What's the earliest indicator of future engagement?
- What action demonstrates they "got it"?

**Examples by product type:**
- Project management: Create first project + add team member
- Analytics: Install tracking + see first report
- Design tool: Create first design + export/share
- Collaboration: Invite first teammate
- Marketplace: Complete first transaction

### Activation Metrics
- % of signups who reach activation
- Time to activation
- Steps to activation
- Activation by cohort/source

---

## Onboarding Flow Design

### Immediate Post-Signup (First 30 Seconds)

**Options:**

1. **Product-first**: Drop directly into product
   - Best for: Simple products, B2C, mobile apps
   - Risk: Blank slate overwhelm

2. **Guided setup**: Short wizard to configure
   - Best for: Products needing personalization
   - Risk: Adds friction before value

3. **Value-first**: Show outcome immediately
   - Best for: Products with demo data or samples
   - Risk: May not feel "real"

**Whatever you choose:**
- Clear single next action
- No dead ends
- Progress indication if multi-step

### Onboarding Checklist Pattern

**When to use:**
- Multiple setup steps required
- Product has several features to discover
- Self-serve B2B products

**Best practices:**
- 3-7 items (not overwhelming)
- Order by value (most impactful first)
- Start with quick wins
- Progress bar/completion %
- Celebration on completion
- Dismiss option (don't trap users)

**Checklist item structure:**
```
[ ] Connect your first data source (2 min)
    Get real-time insights from your existing tools
    [Connect Now]
```

### Empty States

Empty states are onboarding opportunities, not dead ends.

**Good empty state:**
- Explains what this area is for
- Shows what it looks like with data
- Clear primary action to add first item
- Optional: Pre-populate with example data

**Structure:**
1. Illustration or preview
2. Brief explanation of value
3. Primary CTA to add first item
4. Optional: Secondary action (import, template)

### Tooltips and Guided Tours

**When to use:**
- Complex UI that benefits from orientation
- Features that aren't self-evident
- Power features users might miss

**Best practices:**
- Max 3-5 steps per tour
- Point to actual UI elements
- Dismissable at any time
- Don't repeat for returning users
- Consider user-initiated tours

### Progress Indicators

**Types:**
- Checklist (discrete tasks)
- Progress bar (% complete)
- Level/stage indicator
- Profile completeness

**Best practices:**
- Show early progress (start at 20%, not 0%)
- Quick early wins (first items easy)
- Clear benefit of completing
- Don't block features behind completion

---

## Multi-Channel Onboarding

### Email + In-App Coordination

**Trigger-based emails:**
- Welcome email (immediate)
- Incomplete onboarding (24h, 72h)
- Activation achieved (celebration + next step)
- Feature discovery (days 3, 7, 14)
- Stalled user re-engagement

**Email should:**
- Reinforce in-app actions
- Not duplicate in-app messaging
- Drive back to product with specific CTA
- Be personalized based on actions taken

---

## Engagement Loops

### Building Habits

- What regular action should users take?
- What trigger can prompt return?
- What reward reinforces the behavior?

**Loop structure:**
Trigger > Action > Variable Reward > Investment

**Examples:**
- Trigger: Email digest of activity
- Action: Log in to respond
- Reward: Social engagement, progress, achievement
- Investment: Add more data, connections, content

### Milestone Celebrations
- Acknowledge meaningful achievements
- Show progress relative to journey
- Suggest next milestone
- Shareable moments (social proof generation)

---

## Handling Stalled Users

### Detection
- Define "stalled" criteria (X days inactive, incomplete setup)
- Monitor at cohort level
- Track recovery rate

### Re-engagement Tactics

1. **Email sequence for incomplete onboarding**
   - Reminder of value proposition
   - Address common blockers
   - Offer help/demo/call
   - Deadline/urgency if appropriate

2. **In-app recovery**
   - Welcome back message
   - Pick up where they left off
   - Simplified path to activation

3. **Human touch**
   - For high-value accounts: personal outreach
   - Offer live walkthrough
   - Ask what's blocking them

---

## Common Patterns by Product Type

### B2B SaaS Tool
1. Short setup wizard (use case selection)
2. First value-generating action
3. Team invitation prompt
4. Checklist for deeper setup

### Marketplace/Platform
1. Complete profile
2. First search/browse
3. First transaction
4. Repeat engagement loop

### Mobile App
1. Permission requests (strategic timing)
2. Quick win in first session
3. Push notification setup
4. Habit loop establishment

### Content/Social Platform
1. Follow/customize feed
2. First content consumption
3. First content creation
4. Social connection/engagement

---

## Experiment Ideas

### Flow Simplification
- Add/remove email verification during onboarding
- Empty states vs. pre-populated dummy data
- Pre-filled templates to accelerate setup
- OAuth options for faster account linking
- Reduce required onboarding steps

### Step Sequencing
- Different ordering of onboarding steps
- Lead with highest-value features first
- Move friction-heavy steps later
- Required vs. optional step balance

### Progress & Motivation
- Add progress bars or completion percentages
- Onboarding checklists (3-5 items vs. 5-7 items)
- Gamify milestones with badges or rewards
- "X% complete" messaging

### Guided Experience
- Interactive product tours (Navattic, Storylane)
- Tooltip-based guidance vs. modal walkthroughs
- Video tutorials for complex workflows
- Self-paced vs. guided tour options

### Personalization
- Segment by role to show relevant features
- Segment by goal to customize path
- Role-specific dashboards
- Industry-specific examples and templates

### Quick Wins & Engagement
- Highlight quick wins early
- Success messages after key actions
- Progress celebrations at milestones
- Suggest next steps after completion

---

## Measurement

### Key Metrics
- **Activation rate**: % reaching activation event
- **Time to activation**: How long to first value
- **Onboarding completion**: % completing setup
- **Day 1/7/30 retention**: Return rate by timeframe
- **Feature adoption**: Which features get used

### Funnel Analysis
Track drop-off at each step:
```
Signup > Step 1 > Step 2 > Activation > Retention
100%      80%       60%       40%         25%
```
Identify biggest drops and focus there.

---

## Output Format

### Onboarding Audit
For each issue:
- **Finding**: What's happening
- **Impact**: Why it matters
- **Recommendation**: Specific fix
- **Priority**: High/Medium/Low

### Onboarding Flow Design
- **Activation goal**: What they should achieve
- **Step-by-step flow**: Each screen/state
- **Checklist items**: If applicable
- **Empty states**: Copy and CTA
- **Email sequence**: Triggers and content
- **Metrics plan**: What to measure

### Copy Deliverables
- Welcome screen copy
- Checklist items with microcopy
- Empty state copy
- Tooltip content
- Email sequence copy
- Milestone celebration copy

## Related Skills

- `cro/signup-flow-cro.md` - For optimizing signup before onboarding
- `marketing/03-email-marketing/email-sequences.md` - For onboarding email series
- `cro/paywall-cro.md` - For converting to paid during/after onboarding
- `cro/ab-testing.md` - For testing onboarding changes
