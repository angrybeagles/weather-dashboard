# SEO Boost Product Guide

Welcome to SEO Boost. You now have access to professional-grade keyword research data through your AI assistant, plus a comprehensive SEO knowledge system that transforms how you approach search optimization.

---

## Part 1: Getting Started

### What You Have Access To

**API Credits** - Each credit gives you one complete keyword research analysis returning hundreds of data points: search volumes, difficulty scores, intent classifications, related keywords, and competitor rankings.

**Instruction Files** - A complete skill system that teaches your AI how to execute professional SEO workflows. These files work with Claude, ChatGPT, Gemini, or any capable AI assistant.

Your credits never expire. Use them at your own pace.

---

### How SEO Boost Works

SEO Boost connects your AI to professional SEO data. Instead of guessing about keywords, your AI gets real metrics from the same data sources that power enterprise SEO tools.

You ask your AI to research keywords. It calls the SEO Boost API. You get actionable data in seconds.

---

### Step 1: Get Your API Key

Go to your [SEO Boost Dashboard](/account) and copy your API key. It looks like this:

```
sk_seoboost_abc123xyz...
```

Keep this key private. It's tied to your credit balance.

---

### Step 2: Make Your First API Call

Give your AI assistant this prompt:

```
I have an SEO Boost API key for keyword research. Here's how to use it:

POST https://seoboo.st/api/v1/seo

Headers:
  Authorization: Bearer [YOUR_API_KEY]
  Content-Type: application/json

Body:
{
  "keyword": "productivity apps",
  "serp_depth": 50,
  "ideas_limit": 200,
  "related_limit": 200,
  "suggestions_limit": 200,
  "location_name": "United States",
  "language_code": "en"
}

My API key is: [PASTE YOUR KEY HERE]

Research keywords for: [YOUR TOPIC HERE]
```

Replace the placeholders with your actual key and topic.

---

### What Each API Call Returns

One credit returns:

- **Your primary keyword** with complete metrics (volume, difficulty, intent, CPC)
- **Up to 200 keyword ideas** - variations and related terms
- **Up to 200 related keywords** - semantically connected terms
- **Up to 200 search suggestions** - autocomplete data
- **Top 50 SERP results** - who currently ranks and why
- **People Also Ask questions** - real questions searchers ask

That's potentially 650+ data points per credit.

---

### API Parameters Reference

**Required Parameters:**

| Parameter | Type   | Description           |
| --------- | ------ | --------------------- |
| `keyword` | string | The topic to research |

**Optional Parameters:**

| Parameter           | Default         | Description                                |
| ------------------- | --------------- | ------------------------------------------ |
| `serp_depth`        | 10              | Number of SERP results to analyze (10-100) |
| `ideas_limit`       | 100             | Keyword ideas to return (25-700)           |
| `related_limit`     | 100             | Related keywords to return (25-700)        |
| `suggestions_limit` | 100             | Search suggestions to return (25-700)      |
| `location_name`     | "United States" | Target geographic location                 |
| `language_code`     | "en"            | Target language                            |

**Recommended Settings:**

- Quick research: `serp_depth: 10`, limits at `25`
- Standard research: `serp_depth: 50`, limits at `200`
- Comprehensive: `serp_depth: 100`, limits at `500`

---

### Supported Languages

Use these exact codes for the `language_code` parameter.

#### Major Languages

| Language   | Code  |
| ---------- | ----- |
| English    | en    |
| Spanish    | es    |
| French     | fr    |
| German     | de    |
| Italian    | it    |
| Portuguese | pt    |
| Dutch      | nl    |
| Polish     | pl    |
| Russian    | ru    |
| Japanese   | ja    |
| Korean     | ko    |
| Chinese    | zh-CN |
| Arabic     | ar    |
| Hindi      | hi    |
| Turkish    | tr    |
| Vietnamese | vi    |
| Thai       | th    |
| Indonesian | id    |
| Malay      | ms    |
| Swedish    | sv    |
| Norwegian  | no    |
| Danish     | da    |
| Finnish    | fi    |
| Greek      | el    |
| Hebrew     | he    |
| Czech      | cs    |
| Hungarian  | hu    |
| Romanian   | ro    |
| Ukrainian  | uk    |

#### Complete Language List (A-Z)

| Language              | Code    |
| --------------------- | ------- |
| Afrikaans             | af      |
| Akan                  | ak      |
| Albanian              | sq      |
| Amharic               | am      |
| Arabic                | ar      |
| Armenian              | hy      |
| Azeri                 | az      |
| Balinese              | ban     |
| Basque                | eu      |
| Belarusian            | be      |
| Bengali               | bn      |
| Bosnian               | bs      |
| Bulgarian             | bg      |
| Burmese               | my      |
| Catalan               | ca      |
| Cebuano               | ceb     |
| Chichewa              | ny      |
| Chinese (Simplified)  | zh-CN   |
| Chinese (Traditional) | zh-TW   |
| Croatian              | hr      |
| Czech                 | cs      |
| Danish                | da      |
| Dutch                 | nl      |
| English               | en      |
| Estonian              | et      |
| Ewe                   | ee      |
| Faroese               | fo      |
| Farsi (Persian)       | fa      |
| Filipino              | fil     |
| Finnish               | fi      |
| French                | fr      |
| Frisian               | fy      |
| Ga                    | gaa     |
| Galician              | gl      |
| Ganda                 | lg      |
| Georgian              | ka      |
| German                | de      |
| Greek                 | el      |
| Gujarati              | gu      |
| Haitian               | ht      |
| Hausa                 | ha      |
| Hebrew                | he      |
| Hindi                 | hi      |
| Hungarian             | hu      |
| Icelandic             | is      |
| IciBemba              | bem     |
| Igbo                  | ig      |
| Indonesian            | id      |
| Irish                 | ga      |
| Italian               | it      |
| Japanese              | ja      |
| Kannada               | kn      |
| Kazakh                | kk      |
| Khmer                 | km      |
| Kinyarwanda           | rw      |
| Kirundi               | rn      |
| Kongo                 | kg      |
| Korean                | ko      |
| Kreol Morisien        | mfe     |
| Kreol Seselwa         | crs     |
| Krio                  | kri     |
| Kurdish               | ckb     |
| Kyrgyz                | ky      |
| Lao                   | lo      |
| Latvian               | lv      |
| Lingala               | ln      |
| Lithuanian            | lt      |
| Luo                   | ach     |
| Macedonian            | mk      |
| Malagasy              | mg      |
| Malay                 | ms      |
| Malayalam             | ml      |
| Maltese               | mt      |
| Maori                 | mi      |
| Marathi               | mr      |
| Mongolian             | mn      |
| Montenegrin           | sr-ME   |
| Nepali                | ne      |
| Northern Sotho        | nso     |
| Norwegian             | no      |
| Nyankole              | nyn     |
| Oromo                 | om      |
| Pashto                | ps      |
| Pidgin                | pcm     |
| Polish                | pl      |
| Portuguese            | pt      |
| Portuguese (Brazil)   | pt-BR   |
| Portuguese (Portugal) | pt-PT   |
| Punjabi               | pa      |
| Quechua               | qu      |
| Romanian              | ro      |
| Romansh               | rm      |
| Russian               | ru      |
| Serbian               | sr      |
| Serbian (Latin)       | sr-Latn |
| Sesotho               | st      |
| Shona                 | sn      |
| Silozi                | loz     |
| Sindhi                | sd      |
| Sinhalese             | si      |
| Slovak                | sk      |
| Slovenian             | sl      |
| Somali                | so      |
| Spanish               | es      |
| Spanish (Latin Am.)   | es-419  |
| Swahili               | sw      |
| Swedish               | sv      |
| Tajik                 | tg      |
| Tamil                 | ta      |
| Telugu                | te      |
| Thai                  | th      |
| Tigrinya              | ti      |
| Tonga                 | to      |
| Tshiluba              | lua     |
| Tswana                | tn      |
| Tumbuka               | tum     |
| Turkish               | tr      |
| Turkmen               | tk      |
| Ukrainian             | uk      |
| Urdu                  | ur      |
| Uzbek                 | uz      |
| Vietnamese            | vi      |
| Wolof                 | wo      |
| Xhosa                 | xh      |
| Yoruba                | yo      |
| Zulu                  | zu      |

---

### Supported Locations

Use these exact spellings for the `location_name` parameter.

#### Major Markets

| Location       |
| -------------- |
| United States  |
| United Kingdom |
| Canada         |
| Australia      |
| Germany        |
| France         |
| Spain          |
| Italy          |
| Netherlands    |
| Brazil         |
| Mexico         |
| Japan          |
| South Korea    |
| India          |
| Singapore      |

#### Complete Location List (A-Z)

| Location Name                                |
| -------------------------------------------- |
| Afghanistan                                  |
| Albania                                      |
| Algeria                                      |
| American Samoa                               |
| Andorra                                      |
| Angola                                       |
| Antarctica                                   |
| Antigua and Barbuda                          |
| Argentina                                    |
| Armenia                                      |
| Australia                                    |
| Austria                                      |
| Azerbaijan                                   |
| Bahrain                                      |
| Bangladesh                                   |
| Barbados                                     |
| Belgium                                      |
| Belize                                       |
| Benin                                        |
| Bhutan                                       |
| Bolivia                                      |
| Bosnia and Herzegovina                       |
| Botswana                                     |
| Brazil                                       |
| Brunei                                       |
| Bulgaria                                     |
| Burkina Faso                                 |
| Burundi                                      |
| Cambodia                                     |
| Cameroon                                     |
| Canada                                       |
| Cape Verde                                   |
| Caribbean Netherlands                        |
| Central African Republic                     |
| Chad                                         |
| Chile                                        |
| China                                        |
| Christmas Island                             |
| Cocos (Keeling) Islands                      |
| Colombia                                     |
| Comoros                                      |
| Cook Islands                                 |
| Costa Rica                                   |
| Cote d'Ivoire                                |
| Croatia                                      |
| Curacao                                      |
| Cyprus                                       |
| Czechia                                      |
| Democratic Republic of the Congo             |
| Denmark                                      |
| Djibouti                                     |
| Dominica                                     |
| Dominican Republic                           |
| Ecuador                                      |
| Egypt                                        |
| El Salvador                                  |
| Equatorial Guinea                            |
| Eritrea                                      |
| Estonia                                      |
| Eswatini                                     |
| Ethiopia                                     |
| Federated States of Micronesia               |
| Fiji                                         |
| Finland                                      |
| France                                       |
| French Polynesia                             |
| French Southern and Antarctic Lands          |
| Gabon                                        |
| Georgia                                      |
| Germany                                      |
| Ghana                                        |
| Greece                                       |
| Grenada                                      |
| Guam                                         |
| Guatemala                                    |
| Guernsey                                     |
| Guinea                                       |
| Guinea-Bissau                                |
| Guyana                                       |
| Haiti                                        |
| Heard Island and McDonald Islands            |
| Honduras                                     |
| Hungary                                      |
| Iceland                                      |
| India                                        |
| Indonesia                                    |
| Iraq                                         |
| Ireland                                      |
| Israel                                       |
| Italy                                        |
| Jamaica                                      |
| Japan                                        |
| Jersey                                       |
| Jordan                                       |
| Kazakhstan                                   |
| Kenya                                        |
| Kiribati                                     |
| Kuwait                                       |
| Kyrgyzstan                                   |
| Laos                                         |
| Latvia                                       |
| Lebanon                                      |
| Lesotho                                      |
| Liberia                                      |
| Libya                                        |
| Liechtenstein                                |
| Lithuania                                    |
| Luxembourg                                   |
| Madagascar                                   |
| Malawi                                       |
| Malaysia                                     |
| Maldives                                     |
| Mali                                         |
| Malta                                        |
| Marshall Islands                             |
| Mauritania                                   |
| Mauritius                                    |
| Mexico                                       |
| Moldova                                      |
| Monaco                                       |
| Mongolia                                     |
| Montenegro                                   |
| Morocco                                      |
| Mozambique                                   |
| Myanmar (Burma)                              |
| Namibia                                      |
| Nauru                                        |
| Nepal                                        |
| Netherlands                                  |
| New Caledonia                                |
| New Zealand                                  |
| Nicaragua                                    |
| Niger                                        |
| Nigeria                                      |
| Niue                                         |
| Norfolk Island                               |
| North Macedonia                              |
| Northern Mariana Islands                     |
| Norway                                       |
| Oman                                         |
| Pakistan                                     |
| Palau                                        |
| Panama                                       |
| Papua New Guinea                             |
| Paraguay                                     |
| Peru                                         |
| Philippines                                  |
| Pitcairn Islands                             |
| Poland                                       |
| Portugal                                     |
| Qatar                                        |
| Republic of the Congo                        |
| Romania                                      |
| Rwanda                                       |
| Saint Barthelemy                             |
| Saint Helena, Ascension and Tristan da Cunha |
| Saint Kitts and Nevis                        |
| Saint Lucia                                  |
| Saint Martin                                 |
| Saint Pierre and Miquelon                    |
| Saint Vincent and the Grenadines             |
| Samoa                                        |
| San Marino                                   |
| Sao Tome and Principe                        |
| Saudi Arabia                                 |
| Senegal                                      |
| Serbia                                       |
| Seychelles                                   |
| Sierra Leone                                 |
| Singapore                                    |
| Sint Maarten                                 |
| Slovakia                                     |
| Slovenia                                     |
| Solomon Islands                              |
| Somalia                                      |
| South Africa                                 |
| South Georgia and the South Sandwich Islands |
| South Korea                                  |
| South Sudan                                  |
| Spain                                        |
| Sri Lanka                                    |
| Sudan                                        |
| Suriname                                     |
| Sweden                                       |
| Switzerland                                  |
| Taiwan                                       |
| Tajikistan                                   |
| Tanzania                                     |
| Thailand                                     |
| The Bahamas                                  |
| The Gambia                                   |
| Timor-Leste                                  |
| Togo                                         |
| Tokelau                                      |
| Tonga                                        |
| Trinidad and Tobago                          |
| Tunisia                                      |
| Turkey                                       |
| Turkmenistan                                 |
| Tuvalu                                       |
| Uganda                                       |
| Ukraine                                      |
| United Arab Emirates                         |
| United Kingdom                               |
| United States                                |
| United States Minor Outlying Islands         |
| Uruguay                                      |
| Uzbekistan                                   |
| Vanuatu                                      |
| Vatican City                                 |
| Venezuela                                    |
| Vietnam                                      |
| Wallis and Futuna                            |
| Yemen                                        |
| Zambia                                       |
| Zimbabwe                                     |

#### Unsupported Locations

The following locations are not supported due to sanctions or restrictions:

- Belarus
- Russia
- Crimea
- Cuba
- Iran
- North Korea
- Syria

---

### Example Request

```json
{
  "keyword": "home office setup",
  "serp_depth": 50,
  "ideas_limit": 200,
  "related_limit": 200,
  "suggestions_limit": 200,
  "location_name": "United States",
  "language_code": "en"
}
```

Returns data like:

- Main keyword: "home office setup" (12,100 monthly searches, difficulty 45)
- Related: "best home office desk" (8,100 searches)
- Related: "small home office ideas" (6,600 searches)
- Plus 500+ more keywords with full metrics
- Top 50 ranking pages with domain authority and content analysis
- 8-12 People Also Ask questions

---

### Tips for Best Results

**Start Broad, Then Narrow**

Begin with general topics ("fitness apps") then drill into specifics ("workout tracking apps for beginners").

**Use Natural Follow-ups**

After getting data, ask your AI:

- "Which keywords would be easiest to rank for?"
- "Group these by search intent"
- "Create a content plan from the top 20"
- "Find the best long-tail opportunities"

**Conserve Credits Strategically**

Use lower limits (25-50) for quick exploratory research. Use higher limits (200-500) when you're ready to build a complete content strategy.

---

### Troubleshooting

**"Invalid API key"** - Double-check you copied the entire key including `sk_seoboost_`.

**"Insufficient credits"** - Check your [dashboard](/account) balance.

**"Request failed"** - Try again in a moment. Very obscure topics may have limited data.

**Questions?** Email support@seoboo.st

---

## URL Indexing API

After publishing new content, you can request Google indexing through the SEO Boost API. This accelerates how quickly Google discovers and indexes your pages.

### Why Use Indexing

When you publish a new page, Google may take days or weeks to discover it naturally. Submitting an indexing request tells Google "this page exists and is ready for search," significantly reducing wait time.

**Use indexing when:**

- Publishing new blog posts or pages
- Updating existing content with significant changes
- Launching new product or service pages
- Publishing time-sensitive content

### Indexing Endpoint

```
POST https://seoboo.st/api/v1/index

Headers:
  Authorization: Bearer [YOUR_API_KEY]
  Content-Type: application/json

Body:
{
  "urls": [
    "https://yourdomain.com/blog/new-post",
    "https://yourdomain.com/blog/another-post"
  ],
  "projectName": "January blog posts"
}
```

### Indexing Parameters

**Required:**

| Parameter | Type     | Description             |
| --------- | -------- | ----------------------- |
| `urls`    | string[] | Array of URLs to submit |

**Optional:**

| Parameter     | Type   | Description                       |
| ------------- | ------ | --------------------------------- |
| `projectName` | string | Label for organizing in dashboard |

### Limits and Pricing

| Limit            | Value                |
| ---------------- | -------------------- |
| URLs per request | 1-50                 |
| Credit cost      | 1 credit per URL     |
| Rate limit       | 1 request per second |

### Example Response

```json
{
  "success": true,
  "urls": 3,
  "credits": 3,
  "remaining": 97
}
```

### Indexing Best Practices

**Index strategically:** Submit new and significantly updated content only. Do not submit unchanged pages or pages you do not want indexed.

**Group similar content:** Use the `projectName` parameter to organize submissions (e.g., "Week 1 January posts").

**Allow time:** Indexing requests typically process within 24-72 hours. Check Google Search Console to verify indexing status.

### Quick Copy-Paste Prompt

```
I need to submit URLs for Google indexing using my SEO Boost API.

API Endpoint: POST https://seoboo.st/api/v1/index
My API Key: [PASTE YOUR KEY]

URLs to index:
1. [URL 1]
2. [URL 2]

Use project name: "[Describe this project]"
```

---

## Part 2: SEO Knowledge Base

This section contains synthesized SEO knowledge from thousands of practitioner insights. Use this reference to inform your SEO strategy and teach your AI the principles behind effective optimization.

---

### Content Strategy & Topical Authority

#### Hub & Spoke Architecture

Google rewards comprehensive topic coverage over superficial coverage of many topics.

**Structure:**

- **Hub (Pillar):** Comprehensive guide on main topic (3,000-5,000 words)
- **Spokes (Supporting):** Detailed subtopic content (1,500-2,500 words each)
- **Internal Links:** All spokes link to hub; hub links to all spokes

**Implementation:**

1. Choose one pillar topic within your niche
2. Identify 10-15 subtopics that support the pillar
3. Create hub page covering entire topic comprehensively
4. Write detailed spoke content for each subtopic
5. Implement strategic internal linking between all pieces
6. Update cluster quarterly

**Why It Works:**

- Establishes topical authority in Google's eyes
- Passes link equity through internal structure
- Covers user journey comprehensively
- Creates natural keyword targeting across cluster

**Optimal Cluster Size:**

- Minimum: 1 hub + 5 spokes
- Sweet spot: 1 hub + 8-12 spokes
- Maximum: 1 hub + 20 spokes (before creating second cluster)

---

#### E-E-A-T Signals

E-E-A-T = Experience, Expertise, Authoritativeness, Trust

**Experience Signals:**

- First-hand experience demonstrations in content
- Original photos/screenshots from actual usage
- Personal case studies with specific results
- Behind-the-scenes insights
- "What I learned" narratives with specifics

**Expertise Signals:**

- Author credentials displayed prominently
- Professional background relevant to topic
- Industry certifications mentioned
- Regular content in specific domain
- Technical depth appropriate to topic

**Authoritativeness Signals:**

- External validation (backlinks from reputable sites)
- Brand mentions across the web
- Citations in industry publications
- Awards or recognition
- Speaking engagements/thought leadership

**Trust Signals:**

- Transparent sourcing and citations
- Regular content updates (timestamps visible)
- Clear contact information and about page
- HTTPS and security indicators
- Privacy policy and terms of service
- Review platform presence with responses

---

#### Content Quality Framework

**Must Answer Completely:**

- Addresses user's full query, not just partially
- Provides actionable next steps
- Anticipates follow-up questions
- Leaves reader satisfied, not searching further

**Unique Value Requirements:**

- Original insights or perspectives
- First-hand experience or data
- Synthesis of multiple sources with analysis
- Goes deeper than existing content
- Presents information in new, useful format

**Depth Guidelines:**

- Match depth to search intent (not always longer)
- Informational: Comprehensive coverage required
- Commercial: Specific details (pricing, features, comparisons)
- Transactional: Clear path to action
- Navigational: Quick access to destination

**Content Freshness Strategy:**

- Update existing content > creating new thin content
- Add 500-1,000 words to proven performers
- Refresh statistics and examples annually
- Update timestamps where relevant
- Signal freshness in title when appropriate ("2025 Guide")

---

#### Update-Over-Create Philosophy

Updating existing content ranking positions 5-15 delivers faster results than creating new content.

**Why It Works:**

- Google already sees these pages as relevant (just need a push)
- Faster results (2-4 weeks vs. 3-6 months for new content)
- Less effort (1 hour update vs. 4+ hours new content)
- Preserves existing link equity and authority

**Update Process:**

**Step 1: Identify Opportunities (Google Search Console)**

- Filter last 3 months
- Export keyword data
- Focus on positions 5-15 (close to breakthrough)
- Prioritize high-impression, low-CTR pages

**Step 2: Content Enhancement**

- Trim unnecessary fluff (every sentence must serve purpose)
- Strengthen intro (make it compelling, under 30 words ideal)
- Add "Key Takeaways" section at top
- Update statistics and examples
- Add new sections addressing recent developments
- Improve readability and formatting

**Step 3: Internal Linking Boost**

- Find 3-5 relevant existing posts
- Add contextual links TO updated article
- Add links FROM updated article to related content
- Use natural, varied anchor text

**Step 4: Republish**

- Change publish date to current date
- Signal freshness to Google
- Monitor rankings over 2-4 weeks

**Expected Results:** 20-35% traffic increase on updated pages

---

### Keyword Research Tactics

#### Beyond Basic Tools

**SERP Mining Techniques:**

**People Also Ask (PAA) Extraction:**

- Mine PAA boxes for question-format keywords
- Each PAA = potential content piece or FAQ entry
- Expandable PAA boxes reveal deeper questions
- Format answers for featured snippet capture

**Autocomplete Strategy:**

- Google autocomplete for user language patterns
- Try variations: keyword + [letter], keyword + [preposition]
- Bing autocomplete often shows different suggestions
- YouTube autocomplete for video intent keywords

**Related Searches Analysis:**

- Bottom-of-SERP related searches reveal intent variations
- "People also search for" shows semantic relationships
- Track SERP features (calculators, maps, shopping) for intent

---

#### Search Intent Clustering

**Intent Categories:**

**Informational:**

- What/Why/How questions
- Definitions and explanations
- Guides and tutorials
- Research-oriented searches

**Commercial:**

- "Best" + keyword
- Comparisons and alternatives
- Reviews and ratings
- "Top" lists

**Transactional:**

- "Buy" + keyword
- Pricing queries
- Discount/coupon searches
- Specific product searches

**Navigational:**

- Brand name + keyword
- Login/dashboard searches
- Specific page lookups

**Clustering Strategy:**

- Group keywords by intent, not just topic
- Create content serving entire intent cluster
- One comprehensive piece > multiple thin pieces
- Internal link between related intent variations

---

#### Low-Volume Keyword Strategy

Start with 0-100 monthly search volume keywords rather than chasing high-volume terms.

**Why It Works:**

- Less competition (easier to rank)
- Faster results (weeks vs. months)
- Builds early authority signals
- Compound effect leads to bigger keywords
- More specific intent = better conversion

**Volume Progression:**

- Phase 1: 0-100 MSV keywords
- Phase 2: 100-500 MSV keywords
- Phase 3: 500-2,000 MSV keywords
- Phase 4: 2,000+ MSV keywords

---

### On-Page Optimization

#### Title Tag Formula

**Structure:** [Target Keyword] - [Benefit/Modifier] | [Brand]

**Character Limit:** 55-60 characters

**Best Practices:**

- Primary keyword toward the beginning
- Include power words (Guide, Complete, Ultimate, 2025)
- Numbers when relevant (7 Ways, 10 Best)
- Year for freshness signal when appropriate
- Brand at end (can be truncated if needed)

---

#### Meta Description Optimization

**Character Limit:** 155-160 characters

**Required Elements:**

- Target keyword (bolded in search results)
- Clear benefit or value proposition
- Call to action (subtle)
- Compelling reason to click

**Formula:** [Keyword] + [Benefit] + [Differentiator] + [CTA]

---

#### Header Hierarchy

**Structure Rules:**

- H1: One per page (main title, includes target keyword)
- H2: Major sections (5-8 per long-form content)
- H3: Subsections under H2s
- H4-H6: Further hierarchy as needed

**Keyword Strategy:**

- H1: Primary target keyword
- H2s: Semantic variations and subtopics
- H3s: Long-tail variations
- Natural language, not keyword stuffing

---

#### Content Length Guidelines

**Match Intent, Not Arbitrary Counts:**

**Informational Queries:**

- Comprehensive guides: 3,000-5,000 words
- How-to tutorials: 1,500-2,500 words
- Definition content: 500-1,000 words

**Commercial Queries:**

- Comparison posts: 2,000-3,000 words
- Product reviews: 1,500-2,500 words
- "Best of" lists: 2,000-4,000 words

**Transactional Queries:**

- Product pages: 500-1,500 words
- Service pages: 800-1,500 words

**Quality Over Length:**

- Every paragraph must serve a purpose
- Trim fluff and repetition
- Comprehensive does not equal bloated
- User satisfaction > word count

---

#### Image Optimization

**Alt Text Best Practices:**

- Describe image specifically and accurately
- Include target keyword naturally where relevant
- Be descriptive for accessibility
- 125 characters or fewer recommended

**Technical Requirements:**

- WebP format for compression
- Lazy loading for below-fold images
- Responsive images (srcset for different sizes)
- File size optimization
- Descriptive file names

---

### Technical SEO Essentials

#### Core Web Vitals

**Three Critical Metrics:**

**LCP (Largest Contentful Paint):**

- Target: <2.5 seconds
- Measures: Loading performance
- Common fixes: Image optimization, server response time, render-blocking resources

**FID (First Input Delay):**

- Target: <100ms
- Measures: Interactivity
- Common fixes: Minimize JavaScript, code splitting, efficient event handlers

**CLS (Cumulative Layout Shift):**

- Target: <0.1
- Measures: Visual stability
- Common fixes: Size attributes on images/videos, avoid inserting content above existing content

Core Web Vitals are confirmed ranking factors. Poor scores handicap rankings.

---

#### Schema Markup Implementation

**Priority Schema Types:**

**Organization Schema (Homepage):**

- Business name, logo, contact info
- Social media profiles
- Establishes entity in Knowledge Graph

**Article Schema (Blog Posts):**

- Headline, author, publish date, image
- Enables rich results in search
- Required for Google News consideration

**FAQ Schema:**

- Structured Q&A format
- Displays directly in search results
- Excellent for PAA box targeting

**LocalBusiness Schema:**

- Address, phone, hours
- Service area, price range
- Critical for local search visibility

**Implementation:** Use JSON-LD format. Test with Google's Rich Results Test.

---

#### Site Speed Optimization

**Target:** Page load time <3 seconds (ideally <2 seconds)

**High-Impact Fixes:**

**Image Optimization:**

- WebP format conversion
- Compression without quality loss
- Lazy loading
- Responsive images

**Code Optimization:**

- Minify CSS, JavaScript, HTML
- Remove unused code
- Defer non-critical JavaScript
- Inline critical CSS

**Server Optimization:**

- Use CDN for static assets
- Enable gzip/brotli compression
- Optimize server response time (TTFB <600ms)
- Use browser caching

---

### Link Building Strategies

#### What Still Works

**Strategic Guest Posting:**

- Target high-relevance sites (topical alignment)
- DR/DA 40+ domains
- Real traffic on target site
- Editorial standards and review process
- 1-2 contextual links
- Genuine value contribution

**Digital PR & Data Studies:**

- Original research and industry surveys
- Unique statistics and data analysis
- Interactive tools and calculators
- Comprehensive industry reports

**Resource Page Link Building:**

1. Search: `[topic] + "resources"` or `[topic] + "useful links"`
2. Identify relevant resource pages in your niche
3. Personalized outreach with your valuable content

---

#### Link Quality Signals

**Prioritize:**

- **Relevance** > Domain Authority (topical alignment most important)
- **Editorial placement** > Paid/sponsored links
- **Contextual links** > Sidebar/footer
- **Natural anchor text** > Exact match
- **Natural velocity** > Sudden spikes

**Link Diversity:**

- Mix of domain types (.com, .org, .edu, .gov where appropriate)
- Variety of anchor text (branded, naked URLs, generic, partial match)
- Different link types (content, directory, profile where relevant)

---

#### What to Avoid

**Never Do:**

- PBNs (Private Blog Networks)
- Mass directory submissions
- Exact match anchor text over-optimization
- Link exchanges at scale
- Buying links from marketplaces
- Generic blog comment links
- Low-quality guest posts for links

---

### GEO (Generative Engine Optimization)

Optimization for AI-powered search engines (ChatGPT, Perplexity, Claude, Google AI Overviews).

**Why It Matters:**

- Users shifting to AI search for answers
- AI gives 1-2 recommendations, not 10 results
- Early mover advantage

**What LLMs Look For:**

**1. Third-Party Validation:**

- Brand mentions on credible sites
- Editorial mentions (news, industry sites, reviews)
- Quality citations > quantity

**2. Snippable Content:**

- Clear, concise, fact-dense statements
- Can AI extract a sentence as "the answer"?
- Structured Q&A format
- Definitive statements vs. vague marketing language

**3. Schema Markup:**

- LocalBusiness, FAQ, Organization, Product schema
- Helps AI understand your entity

**4. Crawler Access:**

- Allow GPTBot (OpenAI crawler)
- Allow CCBot (Common Crawl)
- Ensure content is text-readable

**GEO vs Traditional SEO:**

- Entity + fact optimization (not just keywords)
- Brand mention quality (not backlink quantity)
- Being THE answer (not ranking 1-10)
- Snippable content blocks (not meta descriptions)

Traditional SEO remains essential. GEO is additive.

---

### Featured Snippets & Rich Results

#### Featured Snippet Optimization

Featured snippets average 35% click share. Can win from positions 5-9.

**Format for Snippet Win:**

**Paragraph Snippets:**

- Answer question in <50 words (300 characters max)
- Position answer after H2 formatted as question
- Start answer by repeating question
- Use concise, direct language

**List Snippets:**

- Format as numbered or bullet list
- 5-10 items ideal
- Each item 1-2 sentences
- Clear, action-oriented items

**Table Snippets:**

- Create comparison table
- 3-5 columns maximum
- 5-10 rows ideal
- Clear, descriptive headers

---

#### People Also Ask Optimization

Create content specifically answering PAA questions.

**Implementation:**

- Extract PAA questions from SERPs
- Create dedicated FAQ page OR answer within relevant content
- Use H2 format for question
- Provide concise answer (50-100 words)
- Link to more comprehensive content

---

### Content Maintenance

#### Content Refresh Strategy

**When to Update:**

- Content 12+ months old
- Rankings declining
- Statistics or data outdated
- New developments in topic
- Competitor content surpassed yours

**What to Update:**

- Current statistics and data
- Examples and case studies
- Screenshots (if outdated)
- Broken links
- Dates in title and content

**Content Improvements:**

- Add 500-1,000 words of new sections
- Improve introduction
- Add "Key Takeaways" section
- Improve formatting and scannability
- Add visuals

**Expected Results:** 20-40% traffic increase within 4-6 weeks

---

#### Content Pruning

**When to Prune:**

- Pages with 0 impressions in last 6+ months
- Thin content that never gained traction
- Outdated content not worth updating
- Duplicate or cannibalizing content

**Process:**

1. Export all URLs from Google Search Console
2. Filter for 0 impressions in last 6 months
3. Evaluate each page:
   - Worth updating? Update it
   - Has backlinks? Keep or redirect
   - No value and no links? Delete or noindex

**Expected Results:** 15-40% traffic increase by improving site quality signal

---

### Internal Linking Systems

#### Why Internal Linking Matters

**Impact:**

- Passes link equity between pages
- Helps Google understand site structure
- Distributes authority to important pages
- Improves crawlability and indexing
- Reduces orphan pages

---

#### Internal Linking Best Practices

**Architecture:**

- Homepage links to main category pages
- Category pages link to subcategories and key content
- All content links to related pieces
- Create hub pages that link to content clusters

**Best Practices:**

- Use descriptive anchor text (not "click here")
- Vary anchor text naturally
- Link early in content (first 100-300 words ideal)
- 3-5 internal links per 1,000 words
- Link to both newer and older content
- Prioritize relevance over link count

---

### Measurement Framework

#### Primary KPIs

**Organic Traffic Metrics:**

- Total organic sessions (monthly trend)
- Landing pages receiving organic traffic
- Month-over-month growth rate
- Traffic by keyword category

**Ranking Metrics:**

- Average position for target keywords
- Keywords in top 3 positions
- Keywords in top 10 positions
- Featured snippet wins

**Engagement Metrics:**

- Bounce rate from organic traffic
- Average time on page
- Pages per session
- Conversion rate from organic traffic

---

### Quick Reference Checklists

#### New Content Publication

- [ ] Keyword research completed
- [ ] Search intent validated
- [ ] Content comprehensive and unique
- [ ] Title tag optimized (<60 chars)
- [ ] Meta description compelling (<160 chars)
- [ ] H1 includes target keyword
- [ ] Header hierarchy logical
- [ ] Images optimized with alt text
- [ ] Internal links added (3-5)
- [ ] Schema markup implemented
- [ ] Mobile preview checked
- [ ] Page speed tested

---

#### Monthly SEO Maintenance

- [ ] Review Google Search Console performance
- [ ] Check Core Web Vitals health
- [ ] Fix technical errors or warnings
- [ ] Update 1-2 pieces of content (12+ months old)
- [ ] Add internal links to recent content
- [ ] Monitor backlink profile
- [ ] Check for broken links
- [ ] Analyze top-performing and declining pages
- [ ] Identify quick win opportunities (positions 5-15)

---

## Part 3: From API Data to Content Strategy

The SEO Boost API gives you comprehensive keyword data. But what do you do with hundreds of keywords once you have them?

### The Strategic Analysis Challenge

Raw keyword data is just the beginning. Professional SEO requires:

- **Categorization** - Sorting keywords by opportunity type (Quick Wins, Authority Builders, Emerging Topics)
- **Clustering** - Grouping semantically related keywords into content themes
- **Prioritization** - Determining which keywords to target first for maximum ROI
- **Content Planning** - Transforming keyword clusters into hub/spoke content architectures

### What Pro Users Get

The **Pro Pack** includes Workflow S: Content Strategy, which transforms your API data into actionable content plans:

**Strategic Categorization**

- Automatically categorizes every keyword by difficulty, volume, and intent
- Identifies Quick Wins (low difficulty + good volume)
- Flags Authority Builders (high-value competitive targets)
- Surfaces Intent Signals (question keywords perfect for featured snippets)

**Semantic Clustering**

- Groups 300-450 keywords into 4-6 thematic clusters
- Each cluster becomes a content pillar opportunity
- Identifies which keywords belong together topically

**Hub/Spoke Content Architecture**

- Creates pillar content strategy for each cluster
- 1 hub article + 3-5 spoke articles per cluster
- Internal linking blueprint for topical authority

**Prioritized Action Plan**

- Week 1-2: Quick Wins priority list
- Week 3-4: Intent Signals for SERP features
- Month 2+: Authority building campaign
- Clear sequence from highest to lowest ROI

### The Complete Pro Workflow

```
API Call (Basic + Pro)
    |
    v
JSON Response with 300-450 keywords
    |
    v
Workflow S: Content Strategy (Pro)
    |
    v
Categorized, clustered, prioritized content plan
    |
    v
Workflow C: Content Generation (Pro)
    |
    v
Publication-ready SEO articles
```

### Upgrade to Pro

Ready to transform raw keyword data into strategic content plans?

The Pro Pack ($39) includes:

- **500 API credits** (15+ complete content strategies)
- **Workflow S** - Content strategy automation
- **Workflow C** - SEO article generation
- **Workflow A** - SEO audit framework
- **Workflow R** - Rank improvement for existing content
- **Workflow T** - Technical SEO implementation
- **Plus** prioritization frameworks, experiment tracking, and content brief templates

[Upgrade to Pro](/pricing) to unlock the complete SEO workflow system.
