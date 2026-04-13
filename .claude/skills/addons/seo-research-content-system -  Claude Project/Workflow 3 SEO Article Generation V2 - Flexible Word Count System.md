# Workflow 3: SEO Article Generation V2 - Flexible Word Count System

**Purpose:** Transform content ideas into publication-ready SEO-optimized articles  
**Platform:** Claude Desktop (or Claude Code)  
**Input:** Content idea details from Workflow 2 OR manual input  
**Output:** Single comprehensive Markdown artifact with SEO-optimized article  
**Capacity:** 1 article per execution  
**Status:** ✅ Production-ready

---

## Word Count Tier System

Workflow 3 supports **flexible word counts** based on content type and depth requirements:

| Tier | Total Words | Use Case | Intro | Body | Conclusion |
|------|-------------|----------|-------|------|------------|
| **Compact** | 500-700 | Quick guides, tactical tips, single concepts | 100-150 | 250-400 | 100-150 |
| **Standard** | 1200-1600 | Comprehensive how-to, feature comparisons | 250-350 | 700-900 | 250-350 |
| **Comprehensive** | 2200-2800 | Deep dives, multi-section guides | 400-500 | 1400-1800 | 400-500 |
| **Authority** | 3000-3500+ | Ultimate guides, pillar content | 400-500 | 1800-2500 | 400-500 |

**Default Tier:** Compact (500-700 words)

**Component Proportions (maintained across all tiers):**
- Introduction: 20-25% of total
- Main Body: 55-65% of total
- Conclusion: 15-20% of total

**Tier Selection Guidelines:**
- **Compact:** Quick Wins, Intent Signals, tactical posts
- **Standard:** Authority Builders (standard), detailed how-to guides
- **Comprehensive:** Authority Builders (deep), comparison guides
- **Authority:** Hub articles, ultimate guides, pillar content

---

## Input Requirements

### Workflow 2 Integration (Recommended)

**If receiving Workflow 2 artifact:**

Look for the **"Workflow 3 Ready Content Ideas"** section at the end of the report. Content ideas are pre-formatted in code blocks like:

```
Title: Install Claude Code: Step-by-Step Guide
Keyword: install claude code
Description: Complete installation guide for Claude Code covering prerequisites, download process, and initial configuration across all platforms. Includes troubleshooting common installation issues and verification steps.
Primary Keyword: Claude Code
Category: Quick Wins
```

**User Action:** Copy the entire code block â†' Paste â†' Say "Execute Workflow 3 on this content idea"

**Claude Action:** Parse the structured format automatically and proceed to SERP analysis.

---

### Manual Input (Alternative)

**Required Fields:**
```
Title: [Article title or working title]
Keyword: [Primary keyword to target]
```

**Optional Fields:**
```
Description: [Brief description of article purpose - 50-120 words]
Primary Keyword: [Parent keyword if different from main keyword]
Category: [Quick Wins | Authority Builders | Intent Signals | Emerging Topics]
Target Audience: [Who is this for - can be inferred if not provided]
Writing Style: [Engaging/Professional/Technical - can be determined from SERP]
Writing Tone: [Conversational/Authoritative/etc - can be determined from SERP]
Target Word Count: [Compact | Standard | Comprehensive | Authority] OR [specific number]
Language: [Default: English, can be Dutch, French, German, etc.]
```

**Example Minimal Input:**
```
Title: How to Style a Bucket Hat
Keyword: bucket hat styling
Target Word Count: Compact
```
*All other details will be inferred through SERP analysis*

**Word Count Specification:**
- Use tier name: "Compact", "Standard", "Comprehensive", "Authority"
- OR specify exact number: "600", "1400", "2500"
- If not specified, defaults to **Compact (500-700 words)**

---

## Execution Mode: Autonomous + Single Artifact Output

**Process Flow:**
1. Receive content idea (Workflow 2 format or manual input)
2. Execute all stages in memory (no external dependencies except web_search)
3. Output complete article as single Markdown artifact

**No Dependencies:**
- No Airtable integration
- No API keys required (except Claude subscription)
- No external file uploads
- Uses built-in web_search tool for SERP analysis

**Execution Principles:**
- Execute all stages sequentially in one session
- Use web_search for real-time competitive analysis
- Apply SEO best practices throughout
- Generate multiple title options for user choice
- Include complete article ready for CMS
- Validate quality at each stage before proceeding
- Target word count based on specified tier

---

## Processing Stages (Autonomous Execution)

### STAGE 1: Input Validation & SERP Analysis

**Step 1.1: Validate Input**

1. Parse input format:
   - If structured block from Workflow 2: Extract all fields
   - If manual input: Identify provided fields
2. Confirm minimum requirements met (title + keyword)
3. Determine target word count:
   - If tier specified: Use tier range
   - If exact number: Use that number
   - If not specified: Default to Compact (500-700)
4. Calculate component targets:
   - Introduction: TARGET × 0.20 to 0.25
   - Main Body: TARGET × 0.55 to 0.65
   - Conclusion: TARGET × 0.15 to 0.20
5. If description missing, infer from title + keyword
6. If category missing, default to "Authority Builders"
7. If language not specified, detect from title
8. Store Primary Keyword for SEO reference section

**Step 1.2: SERP Analysis via web_search**

Execute web search for the primary keyword:
```
web_search(query: "{primary_keyword}")
```

**Analyze search results to determine:**

1. **Search Intent** (informational | commercial | transactional | navigational)
   - Informational: User wants to learn
   - Commercial: User comparing options
   - Transactional: User ready to buy/act
   - Navigational: User seeking specific brand/site

2. **Writing Style** (from analyzing top results)
   - Engaging and storytelling
   - Concise and professional
   - Data-driven and technical
   - Expert guide format

3. **Writing Tone** (from analyzing top results)
   - Friendly and conversational
   - Formal and authoritative
   - Persuasive and compelling

4. **Hidden Insights** (unique angles not obvious in top results)
   - Look for gaps in competitor content
   - Identify uncommon perspectives
   - Find overlooked benefits/considerations
   - If none found: "No significant hidden insights detected"

5. **Target Audience** (who are the top results written for)
   - Small business owners
   - Tech enthusiasts  
   - Fashion-conscious consumers
   - etc.

6. **Article Goal** (what should this accomplish)
   - Educate on best practices
   - Compare options for decision-making
   - Provide step-by-step guidance
   - etc.

7. **Semantic Analysis**
   - Common Subtopics: 4-6 topics frequently covered
   - Related Questions: 4-6 questions users ask
   - Secondary Keywords: 5-10 variations/related terms
   - Long-tail Keywords: 3-5 specific phrases

**Validation Check:**
- Ensure all 7 components extracted
- If search results insufficient, search with broader terms
- Document findings for use in all subsequent stages

---

### STAGE 2: Title Optimization

**Purpose:** Create 3-5 optimized title options

**Process:**
1. Analyze working title from input
2. Consider search intent from SERP analysis
3. Incorporate primary keyword naturally
4. Generate 3-5 variations:
   - **Option 1:** SEO-optimized (keyword-first, clear intent)
   - **Option 2:** Engagement-focused (hook, curiosity, power words)
   - **Option 3:** Balanced (SEO + engagement)
   - **Option 4:** Question format (if intent is informational)
   - **Option 5:** Numbered/list format (if applicable)

**Title Guidelines:**
- 50-65 characters (ideal for SEO)
- Include primary keyword
- Use power words: "Complete", "Ultimate", "Proven", "Best", "2025"
- Match search intent (comparison â†' "vs", guide â†' "How to", info â†' "What is")
- Avoid clickbait or misleading phrasing
- Year in title if evergreen content with updates

**Example Output:**
```
Title Options:
1. Claude Code vs Cursor: Complete Comparison 2025 [SEO-optimized]
2. Which AI Coding Assistant Wins: Claude Code or Cursor? [Engagement]
3. Claude Code vs Cursor: Features, Pricing & Performance [Balanced]
4. Is Claude Code Better Than Cursor for Developers? [Question]
5. 7 Key Differences Between Claude Code and Cursor [Numbered]

Recommended: Option 1 (best search intent match)
```

**Validation Check:**
- All titles 50-65 characters
- Primary keyword in all options
- At least 3 distinct approaches
- One recommended based on search intent

---

### STAGE 3: Key Takeaways Generation

**Purpose:** Create 6-10 high-impact takeaways with intro and outro

**Dynamic Sizing Based on Tier:**
- **Compact (500-700):** 4-6 takeaways
- **Standard (1200-1600):** 6-8 takeaways
- **Comprehensive (2200-2800):** 8-10 takeaways
- **Authority (3000-3500+):** 8-12 takeaways

**Structure:**
```markdown
## Key Takeaways

[Introductory paragraph: 2-3 sentences setting up the takeaways and providing context]

- **Bold action-driven heading:** Concise explanation inline (1-2 sentences).
- **Bold action-driven heading:** Concise explanation inline (1-2 sentences).
- **Bold action-driven heading:** Concise explanation inline (1-2 sentences).
[appropriate number based on tier]

[Outro paragraph: 2-3 sentences transitioning to main body]
```

**Content Guidelines:**
- **Heading format:** Action-driven, engaging (not generic)
  - ✅ "Beyond fixed rules: AI adapts in real time"
  - ❌ "A paradigm shift from traditional automation"
- **Explanation:** Concise but substantive inline
- **Incorporate:**
  - Key concepts from semantic analysis
  - Hidden insight (if valuable) as dedicated takeaway
  - Main benefits/features of topic
- **Avoid:**
  - Section headers, dividers, extra spaces
  - Overwhelming detail (save for main body)
  - Generic or obvious statements

**Example:**
```markdown
## Key Takeaways

Claude Code and Cursor represent the cutting edge of AI-powered coding assistance, each with distinct advantages for different developer workflows. Understanding their core differences helps you choose the tool that best fits your needs.

- **Claude Code excels at complex reasoning tasks:** Powered by Claude Sonnet, it handles multi-step logic and architectural decisions with exceptional clarity.
- **Cursor offers faster completions for routine coding:** Its lightweight model provides near-instant suggestions for common patterns and boilerplate code.
- **Terminal integration sets Claude Code apart:** Direct command-line access enables workflow automation without leaving your development environment.
[... appropriate number more takeaways based on tier ...]

With these insights in mind, let's dive deeper into how each tool performs across different use cases and what that means for your development workflow.
```

**Validation Check:**
- Appropriate number of takeaways for tier
- Intro and outro paragraphs included
- Bolded headings are engaging
- Hidden insight incorporated if valuable
- Smooth transition to body

---

### STAGE 4: Introduction Generation

**Purpose:** Write engaging introduction

**Dynamic Word Count Based on Tier:**

Calculate introduction target:
- **Target = Total Word Count × 0.20 to 0.25**

**Examples:**
- Compact (600 total): 120-150 word intro
- Standard (1400 total): 280-350 word intro
- Comprehensive (2500 total): 500-625 word intro
- Authority (3200 total): 640-800 word intro

**Structure:**
```markdown
## [Section Title - relates to topic]

[Opening paragraph: Direct statement presenting the topic - 2-3 sentences]

### [Optional subheading if breaking into sections]

[Supporting paragraphs: Why topic matters, context, value proposition - 2-4 paragraphs]

[Transition paragraph: Lead into main body - 1-2 sentences]
```

**Content Guidelines:**
- **Opening:** Direct, engaging presentation of topic
  - ✅ "AI coding assistants have transformed how developers work, but choosing between Claude Code and Cursor requires understanding their fundamental differences."
  - ❌ "In today's fast-paced development landscape, AI tools are everywhere..." (generic)
- **Body:**
  - Explain why topic matters
  - Provide relevant context
  - State what reader will learn
  - Match writing style/tone from SERP analysis
- **Primary keyword:** Use naturally in first paragraph
- **Secondary keywords:** Sprinkle 1-2 throughout
- **Transition:** "Let's explore...", "Here's what you need to know...", etc.
- **Word count:** Match calculated target from tier

**Validation Check:**
- Matches calculated word target (Ā±10%)
- No generic openings
- Primary keyword in first paragraph
- Clear value proposition
- Smooth transition to body
- Matches writing style/tone

---

### STAGE 5: Outline Generation

**Purpose:** Create detailed outline for main body

**Dynamic Word Count Based on Tier:**

Calculate main body target:
- **Target = Total Word Count × 0.55 to 0.65**

**Examples:**
- Compact (600 total): 330-390 word body
- Standard (1400 total): 770-910 word body
- Comprehensive (2500 total): 1,375-1,625 word body
- Authority (3200 total): 1,760-2,080 word body

**Structure:**
```markdown
## [Main Section 1 - H2 incorporating secondary keyword]
### [Subsection 1.1 - H3]
Brief: What this covers

### [Subsection 1.2 - H3]
Brief: What this covers

## [Main Section 2 - H2]
### [Subsection 2.1 - H3]
Brief: What this covers

[... appropriate number of sections based on tier ...]
```

**Section Count Guidelines by Tier:**
- **Compact (500-700):** 2-3 main sections, 2-3 subsections each
- **Standard (1200-1600):** 3-5 main sections, 2-4 subsections each
- **Comprehensive (2200-2800):** 5-7 main sections, 3-5 subsections each
- **Authority (3000-3500+):** 6-8 main sections, 3-5 subsections each

**Content Guidelines:**

**Main Sections (H2) should:**
- Cover distinct aspects of the topic
- Use secondary keywords naturally in headings
- Progress logically (basics â†' advanced, or problem â†' solution)
- Align with common subtopics from SERP analysis

**Subsections (H3) should:**
- Break down complex ideas
- Answer related questions from SERP analysis
- Use semantic keywords where natural
- Each have appropriate subsections for tier

**Section Topics (typical patterns):**
- **Comparison articles:** Feature comparison, pricing, use cases, pros/cons
- **How-to guides:** Prerequisites, step-by-step, tips, troubleshooting
- **Informational:** What it is, why it matters, how it works, applications
- **Product reviews:** Features, performance, pricing, alternatives

**Hidden Insight Integration:**
- If valuable, create dedicated section
- Or weave into relevant existing sections
- If not relevant, omit

**Outline Scope:**
- Appropriate number of main sections for tier
- Appropriate subsections per section for tier
- Brief description for each subsection
- Logical progression throughout

**Example (Standard Tier):**
```markdown
## What is Claude Code?
### Core Architecture and Capabilities
Brief: Claude Sonnet 4.5 integration, reasoning engine, command-line interface

### Key Features That Set It Apart
Brief: Terminal automation, multi-file context, architectural planning

## What is Cursor?
### Technical Foundation
Brief: Lightweight model, real-time completions, IDE integration

### Standout Capabilities
Brief: Instant suggestions, refactoring tools, collaborative features

## Head-to-Head Feature Comparison
### Code Completion Quality
Brief: Accuracy, context awareness, language support comparison

### Performance and Speed
Brief: Response times, resource usage, efficiency metrics
[... continues ...]
```

**Validation Check:**
- Appropriate number of main sections for tier
- Logical topic progression
- Secondary keywords in headings
- Related questions addressed
- Hidden insight incorporated if relevant

---

### STAGE 6: Main Body Content Generation

**Purpose:** Write comprehensive main body following outline

**Dynamic Word Count Based on Tier:**

Target already calculated in Stage 5:
- **Target = Total Word Count × 0.55 to 0.65**

**Content Guidelines:**

**Overall Structure:**
- Follow outline exactly
- Maintain H2/H3 hierarchy
- Smooth transitions between sections

**Writing Quality:**
- **Depth:** Provide substantive information, not surface-level
- **Examples:** Include real-world applications, case studies
- **Specificity:** Use concrete details, metrics, comparisons
- **Balance:** Mix paragraphs (primary) with lists (supporting)
- **Keywords:** Integrate naturally throughout
  - Primary keyword: 2-3 times for Compact, 3-5 times for larger tiers
  - Secondary keywords: 1-2 times each
  - Semantic keywords: Sprinkle naturally
  - Long-tail keywords: Use in context

**Formatting:**
- **Paragraphs:** 3-5 sentences, focused on single idea
- **Lists:** Use for features, steps, comparisons (not primary content)
- **Bold text:** Emphasize key terms, important points (sparingly)
- **No images:** Text-only (user adds images in CMS)

**Transitions:**
- End major sections with transition sentence to next topic
- Connect ideas logically
- Avoid abrupt topic shifts

**Industry Examples (if applicable):**
- Provide specific use cases
- Include measurable outcomes when possible
- Vary industries if topic is broad
- Keep relevant to primary topic

**Hidden Insight Usage:**
- If dedicated section created in outline, develop fully
- If woven into sections, enhance relevant parts
- Back up insight with reasoning or examples
- If not relevant, omit completely

**Section Development Pattern:**
```markdown
## [Section Title]

[Opening paragraph: Introduce section topic, connect to previous]

### [Subsection]
[2-3 paragraphs developing the point with details, examples, and context]

### [Subsection]
[2-3 paragraphs continuing with new aspect]

- Optional bullet list for specific items
- Keep lists supporting, not primary content

[Transition sentence to next section]
```

**Word Count Target:** Match calculated target from tier (Ā±10%)

**Validation Check:**
- Matches calculated word target
- All outline sections covered
- Keywords integrated naturally
- Specific examples included
- Smooth transitions
- Proper heading hierarchy
- No filler or fluff content

---

### STAGE 7: Conclusion Generation

**Purpose:** Write compelling conclusion

**Dynamic Word Count Based on Tier:**

Calculate conclusion target:
- **Target = Total Word Count × 0.15 to 0.20**

**Examples:**
- Compact (600 total): 90-120 word conclusion
- Standard (1400 total): 210-280 word conclusion
- Comprehensive (2500 total): 375-500 word conclusion
- Authority (3200 total): 480-640 word conclusion

**Structure:**
```markdown
## [Conclusion Heading]

[Summary paragraph: Key points without repetition - 2-3 sentences]

### [Optional subheading for major point - use for Standard+ tiers]

[Body paragraphs: Reinforce value, broader context - scale based on tier]

[Final thought: Forward-looking or actionable - 2-3 sentences]
```

**Content Guidelines:**

**Summary (opening):**
- Highlight core takeaways
- Don't repeat exact sentences from body
- Synthesize rather than summarize

**Reinforce Value:**
- Why the information matters
- How reader can apply it
- Broader implications

**Strong Final Thought (choose one approach):**
- **Thought-provoking question:** Challenges reader to think deeper
- **Call to action:** Next steps reader should take
- **Future perspective:** What's coming, what to watch for
- **Strategic insight:** Key principle for success

**Avoid:**
- Generic endings: "In conclusion...", "To sum up..."
- Obvious statements: "The future looks bright..."
- Introducing new information
- Repeating introduction content

**Example (Standard Tier):**
```markdown
## Making Your Choice

Claude Code and Cursor each excel in distinct areas of AI-assisted development. Your choice ultimately depends on whether you prioritize deep reasoning capabilities or rapid code generation for routine tasks.

### Beyond the Feature Comparison

The real difference isn't just in what these tools can doâ€"it's in how they fit your development workflow. Claude Code's terminal integration shines when you're architecting complex systems or working across multiple files. Cursor's instant completions excel when you're in flow state writing implementation code.

Looking ahead, both tools will continue evolving, but the fundamental trade-offs remain: depth versus speed, reasoning versus reflex. The developers who succeed won't be those who pick the "best" tool, but those who understand when to use each one strategically. The question isn't which AI assistant you chooseâ€"it's how effectively you leverage its strengths for your specific challenges.
```

**Word Count Target:** Match calculated target from tier (Ā±10%)

**Validation Check:**
- Matches calculated word target
- No exact repetition from body
- Broader context provided
- Strong final thought present
- No generic endings
- Matches article tone

---

### STAGE 8: Article Assembly

**Purpose:** Combine all components into single cohesive article

**Assembly Order:**
```markdown
# [Selected Title from Stage 2]

[Key Takeaways - Stage 3]

[Introduction - Stage 4]

[Main Body - Stage 6]

[Conclusion - Stage 7]
```

**Assembly Validation:**
- Smooth flow from section to section
- Consistent heading hierarchy (H1 title, H2 sections, H3 subsections)
- No duplicate content
- No placeholder text or TODOs
- All markdown formatting correct

---

### STAGE 9: Final Polish & Expansion

**Purpose:** Refine to target word count at 9.5+/10 quality

**Tier-Based Expansion Guidelines:**

**If Under Target Range:**
- Expand thin sections with more detail
- Add specific examples or case studies
- Develop subsections more fully
- Include additional context where valuable
- **Do NOT add fluff or repetition**

**If Within Target Range:**
- Light polish only
- Fix any awkward phrasing
- Ensure consistency

**If Over Target Range:**
- Tighten verbose sections
- Remove redundant explanations
- Keep all valuable content

**Quality Enhancements:**

1. **Section Transitions:** Strengthen connections between sections
2. **Examples:** Ensure specific, measurable, relevant
3. **Industry Diversity:** If applicable, vary examples across sectors
4. **Keyword Integration:** Verify natural usage throughout
5. **Conclusion Strength:** Ensure forward-looking, actionable final thought

**Readability Check:**
- Paragraphs: 3-5 sentences each
- Sentences: Vary length for rhythm
- Lists: Used sparingly, supporting role
- Bold text: Key terms only (not overused)

**Final Word Count:** Must be within tier range or specified target (Ā±10% acceptable)

**Validation Check:**
- Word count in target range
- Quality feels 9.5+/10
- All sections well-developed
- No filler content
- Strong throughout

---

### STAGE 10: Meta Description & SEO Reference

**Purpose:** Create SEO-optimized meta description and compile SEO data

**Meta Description Guidelines:**
- **Length:** 150-160 characters (strict limit)
- **Keywords:** Include primary keyword naturally
- **Hook:** Compelling reason to click
- **Value:** What reader will learn/gain
- **Active voice:** Direct and engaging
- **No clickbait:** Accurate representation

**Example:**
```
Discover which AI coding assistant fits your workflow. Compare Claude Code 
vs Cursor across features, pricing, and performance to make the right choice.
```
(159 characters)

**SEO Reference Compilation:**

Gather all SEO data from SERP analysis:
- Primary Keyword
- Search Intent
- Target Audience
- Secondary Keywords (list)
- Semantic Keywords (list)
- Long-tail Keywords (list)
- Related Questions Addressed (list)
- Writing Style Applied
- Writing Tone Applied
- Hidden Insight (if used)

---

## Output Format: Single Markdown Artifact

Generate ONE comprehensive Markdown document with the following structure:

```markdown
# [Article Title]

**Meta Description:** [150-160 characters]

**Article Statistics:**
- Word Count: [exact count]
- Target Tier: [Compact/Standard/Comprehensive/Authority]
- Primary Keyword: [keyword]
- Category: [category]
- Reading Time: [~X minutes based on 200 words/min]

---

## Key Takeaways

[Intro paragraph]

- **Takeaway 1:** Explanation
- **Takeaway 2:** Explanation
[... appropriate number for tier]

[Outro paragraph]

---

## [Introduction Section Title]

[Introduction content - matches tier word count]

---

## [Main Section 1]

[Opening paragraph]

### [Subsection 1.1]
[Content]

### [Subsection 1.2]
[Content]

[Transition to next section]

---

## [Main Section 2]

[... continues through all main body sections ...]

---

## [Conclusion Section Title]

[Conclusion content - matches tier word count]

---

## SEO Data Reference

**Primary Keyword:** [keyword]  
**Search Intent:** [informational/commercial/transactional/navigational]  
**Target Audience:** [audience]  
**Secondary Keywords:** [list]  
**Semantic Keywords:** [list]  
**Long-tail Keywords:** [list]
**Related Questions Addressed:**
- [Question 1]
- [Question 2]
[... all from SERP analysis]

**Writing Guidelines Applied:**
- Style: [style]
- Tone: [tone]
- Hidden Insight: [insight or "None identified"]

---

**Publication Checklist:**
- [ ] Review for brand voice alignment
- [ ] Add images/graphics in CMS
- [ ] Internal links to related content
- [ ] Update publish date
- [ ] Review meta description in preview
- [ ] Check heading hierarchy in CMS
- [ ] Verify keyword placement feels natural

---

*Article generated by Workflow 3: SEO Article Generation V2*  
*Date: [date]*
```

---

## Execution Checklist

Before generating the artifact, ensure:

- [ ] Input validated (minimum: title + keyword)
- [ ] Word count tier determined (default: Compact if not specified)
- [ ] Component targets calculated:
  - [ ] Introduction: TARGET × 0.20-0.25
  - [ ] Main Body: TARGET × 0.55-0.65
  - [ ] Conclusion: TARGET × 0.15-0.20
- [ ] SERP analysis completed via web_search
- [ ] All 7 SERP components extracted (intent, style, tone, insights, audience, goal, semantic)
- [ ] 3-5 title options generated
- [ ] Recommended title selected based on search intent
- [ ] Key takeaways: Appropriate number for tier with intro/outro
- [ ] Introduction: Matches calculated word target, no generic openings
- [ ] Outline: Appropriate sections for tier, logical progression
- [ ] Main body: Matches calculated word target, follows outline
- [ ] Conclusion: Matches calculated word target, strong final thought
- [ ] Article assembled in correct order
- [ ] Final polish completed
- [ ] **Total word count: Within tier range (Ā±10% acceptable)**
- [ ] Meta description: 150-160 characters
- [ ] All markdown formatting correct
- [ ] SEO data reference section included

---

## Usage Examples

### Example 1: Using Workflow 2 Output (Default Compact)

**User Input:**
```
Execute Workflow 3 on this content idea:

Title: Install Claude Code: Step-by-Step Guide
Keyword: install claude code
Description: Complete installation guide for Claude Code covering prerequisites, download process, and initial configuration across all platforms. Includes troubleshooting common installation issues and verification steps.
Primary Keyword: Claude Code
Category: Quick Wins
```

**Claude Response:**
```
I'll create a Compact tier (500-700 word) installation guide for Claude Code. Let me start by analyzing the competitive landscape.

[Executes web_search for "install claude code"]

Based on SERP analysis:
- Search Intent: Informational (users want step-by-step guidance)
- Writing Style: Clear and instructional
- Target Audience: Developers new to Claude Code
- Hidden Insight: Most guides miss WSL-specific configuration steps

Component targets for 600-word article:
- Introduction: 120-150 words
- Main Body: 330-390 words
- Conclusion: 90-120 words

[... continues through all stages ...]

[Outputs complete 620-word article artifact]
```

### Example 2: Specifying Standard Tier

**User Input:**
```
Write article for keyword: "claude code tutorial"
Target Word Count: Standard
```

**Claude Response:**
```
I'll create a Standard tier (1200-1600 word) tutorial for Claude Code. Since you provided minimal information, I'll infer the details through SERP analysis.

[Executes web_search]

Inferred Details:
- Title: "Claude Code Tutorial: Complete Getting Started Guide"
- Description: Step-by-step tutorial for beginners
- Search Intent: Informational
- Target Audience: Developers new to Claude Code
- Primary Keyword: Claude Code (inferred from context)

Component targets for 1400-word article:
- Introduction: 280-350 words
- Main Body: 770-910 words
- Conclusion: 210-280 words

[... continues ...]

[Outputs complete 1,425-word article artifact]
```

### Example 3: Authority Tier for Hub Article

**User Input:**
```
Execute Workflow 3 on this content idea:

Title: Claude Code Pricing Guide 2025: Complete Plans, Costs & Value Analysis
Keyword: claude code pricing
Description: Comprehensive guide covering all Claude Code pricing tiers including free, pro, and max plans. Detailed breakdown of features, costs per plan, volume discounts, student pricing, and ROI analysis. Includes cost comparison with competitors, upgrade decision framework, and optimization strategies for different use cases. Features interactive pricing calculator and plan comparison matrix.
Primary Keyword: Claude Code
Category: Authority Builders
Type: Hub
Cluster: Pricing & Plans
Target Word Count: Authority
```

**Claude Response:**
```
I'll create an Authority tier (3000-3500 word) comprehensive pricing guide as a hub article. This will serve as the pillar content for the Pricing & Plans cluster.

Component targets for 3200-word article:
- Introduction: 640-800 words
- Main Body: 1,760-2,080 words
- Conclusion: 480-640 words

[Executes SERP analysis]
[Generates 5 title options]
[Creates complete 3,285-word authority article]
[Outputs artifact with note about internal linking to spoke articles]
```

### Example 4: Exact Word Count Specification

**User Input:**
```
Title: 5 Quick Claude Code Tips
Keyword: claude code tips
Target Word Count: 650
```

**Claude Response:**
```
I'll create a 650-word article with quick tips. This falls within the Compact tier range.

Component targets:
- Introduction: 130-163 words
- Main Body: 358-423 words
- Conclusion: 98-130 words

[Continues with article generation matching exact 650-word target]
```

---

## Best Practices

### Input Quality

**Optimal Input (from Workflow 2):**
```
Title: How to Choose the Right Bucket Hat for Your Face Shape
Keyword: bucket hat face shape
Description: A comprehensive guide helping readers select bucket hat styles that complement their facial features, with specific recommendations for round, oval, square, and heart-shaped faces.
Primary Keyword: bucket hats
Category: Quick Wins
Target Word Count: Compact
```

**Minimal Acceptable:**
```
Title: Bucket Hat Face Shape Guide
Keyword: bucket hat face shape
Target Word Count: Standard
```

**Too Vague (avoid):**
```
Write about hats
```

### Workflow Efficiency

**Single Article:**
1. Copy content idea from Workflow 2
2. Paste into new Claude chat
3. Say "Execute Workflow 3 on this content idea"
4. Wait 3-5 minutes
5. Review artifact
6. Request adjustments if needed
7. Copy to CMS

**Batch Processing (5-10 articles):**
1. Select top content ideas from Workflow 2
2. Create separate Claude chat for each
3. Execute all simultaneously
4. Review batch together
5. Publish in priority order

### Tier Selection Strategy

**For Quick Wins (Workflow 2):**
- Default to **Compact (500-700)** for tactical, focused content
- Use **Standard (1200-1600)** if topic needs comprehensive coverage

**For Authority Builders:**
- Use **Standard (1200-1600)** for typical authority content
- Use **Comprehensive (2200-2800)** for deep-dive comparisons
- Use **Authority (3000-3500+)** only for pillar/hub articles

**For Intent Signals:**
- Default to **Compact (500-700)** for question-based content
- Questions typically answered concisely

**For Hub/Spoke Architecture:**
- **Hub articles:** Authority tier (3000-3500+)
- **Spoke articles:** Standard or Comprehensive tier (1200-2800)

---

## Summary

Workflow 3 is the **article generation engine** of the SEO system. It:

âœ… Takes content ideas from Workflow 2 (or standalone input)  
✅ Parses structured format automatically  
✅ Analyzes real-time SERP data via web_search  
✅ Generates comprehensive writing guidelines  
✅ Creates SEO-optimized articles in flexible word counts  
✅ Supports 4 tiers: Compact (500-700), Standard (1200-1600), Comprehensive (2200-2800), Authority (3000-3500+)  
✅ Defaults to Compact tier for efficient content production  
✅ Calculates component proportions dynamically (Intro 20-25%, Body 55-65%, Conclusion 15-20%)  
✅ Outputs complete Markdown artifact  
✅ Includes 3-5 title options for user choice  
✅ Provides meta description + SEO reference data  
✅ Executes in 3-5 minutes  
✅ Costs $0 (included in Claude subscription)  
✅ Achieves 9.5+/10 quality consistently  
✅ Requires minimal editing (20-30 min review)  
✅ No external dependencies (except web_search)  
✅ Allows iteration in same chat  

**Recommended System Flow:**
1. Run Workflow 1 (keyword discovery via n8n)
2. Run Workflow 2 (strategic organization in Claude)
3. Review "Workflow 3 Ready Content Ideas" section
4. Copy desired content idea block
5. Specify tier if different from Compact default
6. Paste into new Claude chat â†' "Execute Workflow 3 on this content idea"
7. Review article (20-30 min)
8. Add images and internal links in CMS
9. Publish

---

*End of Workflow 3 V2 Documentation*