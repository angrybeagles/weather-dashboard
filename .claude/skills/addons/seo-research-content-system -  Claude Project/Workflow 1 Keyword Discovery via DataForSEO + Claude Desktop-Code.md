# Workflow 1: Keyword Discovery via DataForSEO MCP - Claude Desktop / Code

**Purpose:** Generate comprehensive keyword variations from a primary keyword using DataForSEO APIs  
**Platform:** Claude Desktop or Claude Code with custom DataForSEO MCP server  
**Output:** Markdown artifact with 300-450 keyword variations in table format  
**Status:** ✅ Production-ready (MCP-optimized)

---

## Overview

Workflow 1 is the foundation of the SEO research system. It takes a single primary keyword (e.g., "Claude Code") and automatically discovers 300-450 related keyword variations using 5 DataForSEO API endpoints executed in parallel through a custom MCP server.

**Major Architectural Improvement:**  
The custom `dataforseo-claudefast` MCP server executes all 5 API calls in parallel and returns optimized, filtered results with **90%+ token reduction** compared to raw API responses. This allows the entire keyword discovery process to happen directly in Claude Desktop/Code with results fitting comfortably in the context window.

---

## Architecture Revolution

### Old Approach (n8n + Airtable)
```
User → Airtable → n8n Webhook → 6 API Calls → Data Transform → Airtable Storage → CSV Export → Workflow 2
Time: 2-3 minutes | Dependencies: 2 external systems | Token overhead: Massive
```

### New Approach (Claude Desktop/Code + MCP)
```
User → Claude Desktop/Code → Single MCP Tool Call (5 parallel APIs) → Markdown Artifact → Workflow 2
Time: <60 seconds | Dependencies: 0 external systems | Token overhead: Minimal (90%+ reduction)
```

**Key Benefits:**
- ✅ **Single step execution** - One tool call, immediate results
- ✅ **90%+ token reduction** - Optimized filtering and formatting
- ✅ **No external dependencies** - No n8n, no Airtable, no CSV exports
- ✅ **Parallel execution** - All 5 APIs run simultaneously
- ✅ **Direct artifact output** - Ready for Workflow 2 immediately
- ✅ **Execution in <60 seconds** - Dramatically faster than old approach

---

## The All-in-One MCP Tool

### Tool Name
`dataforseo-claudefast:claudefast_keyword_all_in_one`

### What It Does
Executes 5 DataForSEO API endpoints in parallel and returns unified, optimized results:

1. **Keyword Ideas** - Category-based keyword expansion
2. **Keyword Suggestions** - Keywords containing the target phrase
3. **Related Keywords** - Semantically related keywords (depth-first search)
4. **Autocomplete** - Google autocomplete suggestions
5. **SERP Analysis** - Top organic results + People Also Ask questions

**Response Format:** Compact, filtered JSON with only essential data (keyword, MSV, difficulty, intent, competition, CPC, type)

**Token Efficiency:** Returns 300-450 keywords in ~11-20K tokens (vs. 130K+ tokens for raw API responses)

### Tool Parameters

| Parameter | Type | Required | Description | Default/Example |
|-----------|------|----------|-------------|-----------------|
| `keyword` | string | ✅ Yes | Primary keyword to research | "Claude Code" |
| `location_name` | string | ✅ Yes | Geographic location for search data | "United States" |
| `language_code` | string | ✅ Yes | Language code for search data | "en" |
| `ideas_limit` | number | ✅ Yes | Max keyword ideas (20-1000) | 200 |
| `suggestions_limit` | number | ✅ Yes | Max keyword suggestions (20-1000) | 200 |
| `related_limit` | number | ✅ Yes | Max related keywords (20-1000) | 200 |
| `related_depth` | number | ✅ Yes | Related keywords search depth (1-4) | 2 |
| `serp_depth` | number | ✅ Yes | SERP analysis depth (10-700) | 50 |

### Standard Configuration

**Recommended Settings:**
```json
{
  "keyword": "[your keyword]",
  "location_name": "United States",
  "language_code": "en",
  "ideas_limit": 200,
  "suggestions_limit": 200,
  "related_limit": 200,
  "related_depth": 2,
  "serp_depth": 50
}
```

**Why These Values:**
- **200 limit per endpoint** - Optimal balance of coverage and processing efficiency
- **Depth 2** - Balances semantic discovery with API cost
- **50 SERP depth** - Captures top results plus PAA questions without excess

### Location Format Examples

| Format | Example |
|--------|---------|
| Country only | "United States" |
| State, Country | "California,United States" |
| City, State, Country | "San Francisco,California,United States" |
| International | "United Kingdom", "Germany", "Canada" |

### Language Codes

| Language | Code |
|----------|------|
| English | en |
| Spanish | es |
| French | fr |
| German | de |
| Dutch | nl |
| Italian | it |
| Portuguese | pt |

---

## Execution Instructions

### Step 1: Gather Input Parameters

**Required Information:**
- Primary keyword (e.g., "Claude Code")
- Location (e.g., "United States")
- Language code (e.g., "en")

**Optional Customization:**
- Adjust limits if you want more/fewer results
- Increase depth for deeper semantic exploration
- Increase SERP depth for more competitive intelligence

### Step 2: Execute Tool Call

Call the all-in-one tool with your parameters:

```xml
<function_calls>
<invoke name="dataforseo-claudefast:claudefast_keyword_all_in_one">
<parameter name="keyword">Claude Code</parameter>
<parameter name="location_name">United States</parameter>
<parameter name="language_code">en</parameter>
<parameter name="ideas_limit">200</parameter>
<parameter name="suggestions_limit">200</parameter>
<parameter name="related_limit">200</parameter>
<parameter name="related_depth">2</parameter>
<parameter name="serp_depth">50</parameter>
</invoke>
</function_calls>
```

**Expected Response Time:** 45-60 seconds  
**Expected Data:** 300-450 keywords + PAA questions + subtopics + SERP results

### Step 3: Receive and Validate Response

The MCP tool returns a JSON object with this structure:

```json
{
  "primary_keyword": "string",
  "execution_time": "string",
  "total_keywords": number,
  "total_paa_questions": number,
  "total_subtopics": number,
  "total_serp_results": number,
  "keywords_by_type": {
    "KW Related": number,
    "KW Suggestion": number,
    "KW Idea": number
  },
  "keywords": [
    ["keyword", msv, difficulty, "intent", "competition", cpc, "type"],
    ...
  ],
  "paa_questions": ["question 1", "question 2", ...],
  "subtopics": ["subtopic 1", "subtopic 2", ...],
  "serp_results": [
    ["url", "domain", "title", "description", "type"],
    ...
  ]
}
```

**Validation:**
- Confirm total_keywords > 0
- Verify all three keyword types present
- Note execution time

### Step 4: Generate Markdown Artifact

**CRITICAL PRINCIPLE:** Workflow 1 is pure data collection. Your only job is to format the JSON response into a clean markdown table. NO analysis, NO interpretation, NO filtering, NO commentary.

---

## Output Format: Markdown Artifact Template

```markdown
# Keyword Research Results: [Primary Keyword]

**Execution Date:** [current date]  
**Location:** [location from input]  
**Language:** [language code from input]  
**Execution Time:** [execution_time from response]

---

## Summary Statistics

- **Total Keywords:** [total_keywords]
- **PAA Questions:** [total_paa_questions]
- **Subtopics:** [total_subtopics]
- **SERP Results:** [total_serp_results]

**Keywords by Type:**
- KW Related: [keywords_by_type.KW Related]
- KW Suggestion: [keywords_by_type.KW Suggestion]
- KW Idea: [keywords_by_type.KW Idea]

---

## Keywords Table

| Keyword | MSV | Difficulty | Intent | Competition | CPC | Type |
|---------|-----|------------|--------|-------------|-----|------|
| [row[0]] | [row[1]] | [row[2]] | [row[3]] | [row[4]] | [row[5]] | [row[6]] |

*[Include ALL keywords from the response - no filtering, no sorting]*

---

## People Also Ask Questions

1. [paa_questions[0]]
2. [paa_questions[1]]
3. [paa_questions[2]]

*[Include all PAA questions from response]*

---

## Subtopics

- [subtopics[0]]
- [subtopics[1]]
- [subtopics[2]]

*[Include all subtopics from response]*

---

## SERP Results

| URL | Domain | Title | Description |
|-----|--------|-------|-------------|
| [row[0]] | [row[1]] | [row[2]] | [row[3]] |

*[Include ALL SERP results from response]*

---

**Next Step:** Use this artifact as input for Workflow 2 Strategic Keyword Organization

*Generated by Workflow 1: Keyword Discovery via DataForSEO MCP*
```

---

## Formatting Guidelines

### For Keywords Table

**Rules:**
- Include ALL keywords from response (typically 300-450 rows)
- Each row is: `[keyword, msv, difficulty, intent, competition, cpc, type]`
- Display null values as empty cells or "null"
- NO filtering based on relevance (Workflow 2 handles that)
- NO sorting or reordering
- NO categorization or grouping
- Just format the data as-is into markdown table

**Example Row:**
```
| claude code | 90500 | 38 | informational | LOW | 4.26 | KW Related |
```

**Handling Null Values:**
```
| claude code vs cursor | 2900 | null | informational | LOW | 8.86 | KW Related |
```
OR
```
| claude code vs cursor | 2900 | | informational | LOW | 8.86 | KW Related |
```

### For Summary Statistics

**Rules:**
- Use exact counts from response
- Simple bullet list format
- NO commentary or interpretation
- NO recommendations

**Example:**
```markdown
## Summary Statistics

- **Total Keywords:** 445
- **PAA Questions:** 4
- **Subtopics:** 45
- **SERP Results:** 48

**Keywords by Type:**
- KW Related: 45
- KW Suggestion: 200
- KW Idea: 200
```

### For PAA Questions

**Rules:**
- Numbered list
- Exact question text from response
- NO rephrasing or editing
- NO answers or commentary

**Example:**
```markdown
## People Also Ask Questions

1. What is a Claude Code?
2. Can I use the Claude Code for free?
3. Is Claude Code better than ChatGPT?
4. How much does Claude Code cost?
```

### For Subtopics

**Rules:**
- Bullet list
- Exact text from response
- Preserve all subtopics (typically 20-50)
- NO categorization or grouping

**Example:**
```markdown
## Subtopics

- claude code vs code
- claude vs code
- vs code
- code with claude
- claude ai code
- what is claude code
- claude code mcp
```

### For SERP Results

**Rules:**
- Table format with 4 columns: URL, Domain, Title, Description
- Include ALL results (typically 40-50)
- URLs can be formatted as markdown links: `[domain](url)` (optional)
- Preserve full descriptions even if long
- NO analysis of competitors
- NO selection of "best" results

**Example:**
```markdown
| URL | Domain | Title | Description |
|-----|--------|-------|-------------|
| https://www.claude.com/product/claude-code | www.claude.com | Claude Code | Your code's new collaborator. Embed Claude directly in your terminal... |
```

---

## What NOT to Include

**Workflow 1 is data collection only. Do NOT include:**

❌ **No keyword categorization** (Quick Wins, Authority Builders, etc.)  
❌ **No opportunity analysis** ("This keyword has low difficulty!")  
❌ **No recommendations** ("Focus on these keywords first")  
❌ **No relevance filtering** (Keep ALL keywords, even "claude monet")  
❌ **No clustering or grouping** (That's Workflow 2's job)  
❌ **No strategic insights** ("Great opportunities in...")  
❌ **No commentary** on data quality  
❌ **No content ideas** (That's Workflow 2's job)  
❌ **No SERP analysis** (Just display the data)  
❌ **No competitor observations**

**Remember:** You're a data pipeline, not an analyst. Format the data cleanly and pass it forward. All analysis happens in Workflow 2.

---

## Integration with Workflow 2

### Data Handoff

**Old Approach (n8n + Airtable):**
- Wait for research to complete in Airtable
- Export "Master All KW Variations" as CSV
- Upload CSV to Claude for Workflow 2 analysis

**New Approach (Claude Desktop/Code + MCP):**
- Execute tool call
- Generate markdown artifact immediately
- Copy artifact to new session for Workflow 2 (or continue in same session)

**Why Markdown Artifact?**
- Zero external dependencies
- Immediate availability for downstream analysis
- Easy to copy/paste between sessions
- Human-readable format
- Ready for Workflow 2 strategic analysis

### Usage Pattern

**Option A: Same Session**
1. Run Workflow 1 tool call
2. Generate markdown artifact
3. Say "Execute Workflow 2 on this data"
4. Workflow 2 proceeds with strategic analysis

**Option B: New Session (Recommended)**
1. Run Workflow 1 tool call
2. Generate markdown artifact
3. Copy artifact content
4. Open new Claude session
5. Paste artifact + "Execute Workflow 2 on this data"
6. Fresh context for strategic analysis

---

## Performance Metrics

### Actual Execution Example

**Input:**
- Primary keyword: "claude code"
- Location: United States
- Language: en
- Standard limits (200/200/200, depth 2, SERP 50)

**Processing:**
- API Calls: 5 parallel requests (unified in single MCP tool call)
- Execution Time: **46 seconds**
- Data Optimization: 90%+ token reduction from raw API responses

**Output:**
- Keywords: **445 variations**
- PAA Questions: **4 questions**
- Subtopics: **45 topics**
- SERP Results: **48 organic results**
- Total Response: **~20K tokens** (vs. 130K+ raw)

**Keywords by Type:**
- KW Related: 45
- KW Suggestion: 200
- KW Idea: 200

### Resource Usage

**DataForSEO API Costs:**
- Per research run: ~$0.30-0.40
- 5 API calls × ~$0.06-0.08 each

**Token Efficiency:**
- Raw API responses: ~130K tokens
- MCP optimized output: ~11-20K tokens
- Reduction: 90%+ (fits comfortably in context window)

**Claude Desktop/Code:**
- Single tool call execution
- No external systems required
- Immediate artifact generation
- Ready for Workflow 2 in <60 seconds

---

## Example Execution Flow

### User Input
```
Run Workflow 1 keyword research for "Claude Code"
Location: United States
Language: en
```

### Claude Response
```
I'll run comprehensive keyword research for "Claude Code" using the DataForSEO MCP tool.

[Executes tool call with standard parameters]

[Receives 445 keywords + PAA + subtopics + SERP data in 46 seconds]

[Generates markdown artifact with formatted table]

**Research Complete!**

- 445 keywords discovered
- 4 People Also Ask questions
- 45 subtopics identified
- 48 SERP results captured
- Execution time: 46 seconds

The markdown artifact is ready. You can now:
1. Review the data
2. Copy to new session for Workflow 2 analysis
3. Or continue here with "Execute Workflow 2 on this data"
```

---

## Troubleshooting

### Common Issues

**Issue:** Tool call fails or times out  
**Solution:**
- Verify MCP server is running
- Check DataForSEO API credits
- Reduce limits if needed (100/100/100 instead of 200/200/200)

**Issue:** Response is empty or incomplete  
**Solution:**
- Check if keyword is too niche (try broader term)
- Verify location and language are valid
- Review DataForSEO API status

**Issue:** Token limit exceeded  
**Solution:**
- Should not happen with optimized MCP (90% reduction)
- If it does, reduce limits further
- Contact MCP server maintainer

---

## Summary

Workflow 1 is the **keyword discovery engine** of the SEO research system, completely rebuilt for Claude Desktop/Code with MCP. It:

✅ Takes a single primary keyword + location + language  
✅ Executes 5 DataForSEO APIs in parallel via single MCP tool call  
✅ Discovers 300-450+ keyword variations in <60 seconds  
✅ Returns optimized, filtered results with 90%+ token reduction  
✅ Outputs clean markdown artifact with formatted tables  
✅ Zero external dependencies (no n8n, no Airtable, no CSV exports)  
✅ Immediate availability for Workflow 2 strategic analysis  
✅ Execution time: 46 seconds average  
✅ Cost: ~$0.30-0.40 per research run (DataForSEO only)  
✅ Format: Publication-ready markdown table  

**Key Innovation:** Custom MCP server consolidates 5 API calls into one, filters unnecessary data, and returns compact format that fits in context window—enabling the entire SEO research workflow to run natively in Claude Desktop/Code.

**Status:** Production-ready, dramatically faster and simpler than n8n/Airtable approach.

**Next Step:** Copy markdown artifact → New Claude session → "Execute Workflow 2 on this data" for strategic analysis and content planning.

---

*End of Workflow 1 Documentation*