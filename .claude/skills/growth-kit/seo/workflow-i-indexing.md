# Workflow I: Indexing

- **Purpose:** Get new or updated content discovered by Google faster
- **Platform:** Claude Desktop (or Claude Code) with API access
- **Input:** URLs of published pages you want indexed
- **Output:** Confirmation that URLs were submitted for indexing
- **When to Use:** After publishing new content (Workflow C) or updating existing pages

---

## What is Google Indexing?

When you publish a page, Google does not see it immediately. Google's bots must first discover and crawl your page before it appears in search results. This process can take days or even weeks.

**The Problem:**

- New blog posts sit undiscovered for 1-4 weeks
- Updated content does not reflect changes in search for days
- Time-sensitive content misses its window
- You wait and wonder if Google found your page

**The Solution:**

Indexing requests tell Google "this page exists and is ready for search." Instead of waiting for Google to find your content, you proactively notify them.

---

## When to Use Indexing

### Perfect Timing

**Use indexing after:**

- Publishing a new blog post (Workflow C output)
- Updating an existing page with significant changes (Workflow R improvements)
- Launching a new product or service page
- Publishing time-sensitive content (event announcements, seasonal guides)
- Republishing refreshed content with a new date

**Do not use indexing for:**

- Pages already ranking well (Google knows about them)
- Minor typo fixes or formatting changes
- Pages you do not want in search results
- Duplicate content or thin pages

### The Workflow Position

Indexing sits between content creation and rank optimization:

```
Workflow C (Create Content)
    |
    v
Publish to Your Website
    |
    v
Workflow I (Request Indexing)  <-- You are here
    |
    v
Google Indexes Your Page (hours to days)
    |
    v
Workflow R (Rank Improvement if needed)
```

---

## How to Use the Indexing API

### Step 1: Gather Your URLs

Collect the full URLs of pages you want indexed. These must be:

- Live and publicly accessible
- Fully published (not drafts)
- HTTP or HTTPS protocol
- Your own domain (not third-party sites)

**Example URLs:**

```
https://yourdomain.com/blog/new-seo-guide
https://yourdomain.com/products/summer-collection
https://yourdomain.com/about
```

### Step 2: Make the API Call

Give your AI assistant this prompt:

```
I want to submit URLs for Google indexing using my SEO Boost API key.

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

My API key is: [PASTE YOUR KEY HERE]

Submit these URLs for indexing:
[LIST YOUR URLS HERE]
```

Replace the placeholders with your actual key and URLs.

### Step 3: Understand the Response

A successful response looks like:

```json
{
  "status": "ok",
  "project_id": "idx_abc123xyz",
  "urls_submitted": 3,
  "credits_used": 3,
  "credits_remaining": 97
}
```

This confirms:

- Your URLs were submitted successfully
- Credits were deducted from your balance
- You can track this project in your dashboard

---

## API Parameters Reference

### Required Parameters

| Parameter | Type     | Description             |
| --------- | -------- | ----------------------- |
| `urls`    | string[] | Array of URLs to submit |

### Optional Parameters

| Parameter     | Type   | Description                       |
| ------------- | ------ | --------------------------------- |
| `projectName` | string | Label for organizing in dashboard |

### Limits

| Limit            | Value                |
| ---------------- | -------------------- |
| URLs per request | 1-50                 |
| Credit cost      | 1 credit per URL     |
| Rate limit       | 1 request per second |

---

## Credit Cost

Indexing uses a simple credit model:

**1 credit = 1 URL submitted**

| Batch Size | Credits Used |
| ---------- | ------------ |
| 1 URL      | 1 credit     |
| 5 URLs     | 5 credits    |
| 10 URLs    | 10 credits   |
| 50 URLs    | 50 credits   |

**Budget Planning:**

- Typical blog post batch (5 URLs): 5 credits
- Weekly publishing (10 URLs): 10 credits
- Large content launch (50 URLs): 50 credits

---

## Best Practices

### 1. **IMPORTANT: Always Batch URLs Together**

**Why Batching Matters:**

The indexing API processes requests through a queue system that handles rate limiting automatically. You should always group related URLs into a single API call rather than making individual requests.

**DO THIS (Good):**

```json
{
  "urls": [
    "https://yourdomain.com/blog/post-1",
    "https://yourdomain.com/blog/post-2",
    "https://yourdomain.com/blog/post-3",
    "https://yourdomain.com/blog/post-4",
    "https://yourdomain.com/blog/post-5"
  ],
  "projectName": "Week 1 blog batch"
}
```

**DON'T DO THIS (Bad):**

```json
// Making 5 separate API calls
{ "urls": ["https://yourdomain.com/blog/post-1"] }
{ "urls": ["https://yourdomain.com/blog/post-2"] }
{ "urls": ["https://yourdomain.com/blog/post-3"] }
{ "urls": ["https://yourdomain.com/blog/post-4"] }
{ "urls": ["https://yourdomain.com/blog/post-5"] }
```

**Batching Benefits:**

- Faster processing (all URLs submitted at once)
- Better organization in your dashboard
- Single project tracking instead of multiple entries
- More efficient credit usage tracking

**When to Batch:**

- Weekly blog posts → Batch all URLs from that week
- Product launch → Batch all product-related pages
- Content refresh → Batch all updated pages
- Site section → Batch all pages from one category

**Maximum Batch Size:** Up to 50 URLs per request

### 2. Index Strategically

Do not index every page you have. Focus on:

- New content you want ranking quickly
- Important updates to high-value pages
- Pages critical to your business goals

### 3. Use Descriptive Project Names

Use the `projectName` parameter to organize:

```json
{
  "urls": ["url1", "url2", "url3"],
  "projectName": "Q1 product launch"
}
```

Good project names:

- "January 2026 blog posts"
- "Product pages refresh"
- "Holiday landing pages"

This helps you track what was submitted when.

### 4. Time Your Submissions

**Best timing:**

- Right after publishing (within minutes to hours)
- After significant content updates
- During business hours for tracking

**Avoid:**

- Submitting drafts or incomplete pages
- Re-indexing pages that have not changed
- Bulk submissions of old, unchanged content

### 5. Track Results

After submitting:

1. Check your SEO Boost dashboard to see project history
2. Wait 24-72 hours for initial indexing
3. Use Google Search Console to verify indexed status
4. Note which content types index fastest for future planning

---

## Example Workflows

### After Publishing a New Blog Post

```
User: I just published a new blog post at https://myblog.com/seo-tips-2026

AI Action:
POST /api/v1/index
{
  "urls": ["https://myblog.com/seo-tips-2026"],
  "projectName": "Blog post - SEO tips"
}

Result: URL submitted, 1 credit used
```

### After a Weekly Content Batch

```
User: Submit these 4 new posts for indexing:
- https://myblog.com/post-1
- https://myblog.com/post-2
- https://myblog.com/post-3
- https://myblog.com/post-4

AI Action:
POST /api/v1/index
{
  "urls": [
    "https://myblog.com/post-1",
    "https://myblog.com/post-2",
    "https://myblog.com/post-3",
    "https://myblog.com/post-4"
  ],
  "projectName": "Week 2 January posts"
}

Result: 4 URLs submitted, 4 credits used
```

### After Updating Important Pages

```
User: I updated my pricing and about pages with new information.
Submit them for re-indexing.

AI Action:
POST /api/v1/index
{
  "urls": [
    "https://mybusiness.com/pricing",
    "https://mybusiness.com/about"
  ],
  "projectName": "Core page updates Jan 2026"
}

Result: 2 URLs submitted, 2 credits used
```

---

## Copy-Paste Prompts

### Prompt 1: Index a Single Page

```
I just published a new page. Please submit it for Google indexing using my SEO Boost API.

API Endpoint: POST https://seoboo.st/api/v1/index
My API Key: [PASTE YOUR KEY]

URL to index: [PASTE YOUR URL]

Use batch name: "[Describe what this page is]"
```

### Prompt 2: Index Multiple Pages

```
I have several new pages to submit for indexing. Please use my SEO Boost API.

API Endpoint: POST https://seoboo.st/api/v1/index
My API Key: [PASTE YOUR KEY]

URLs to index:
1. [URL 1]
2. [URL 2]
3. [URL 3]

Use batch name: "[Describe this batch]"
```

### Prompt 3: Post-Workflow C Indexing

```
I just created a new article using Workflow C and published it.
Now I need to request indexing so Google finds it faster.

API Endpoint: POST https://seoboo.st/api/v1/index
My API Key: [PASTE YOUR KEY]

The published URL is: [YOUR URL]

Batch name: "Workflow C output - [article topic]"
```

---

## Integration with Other Workflows

### From Workflow C (Content Creation)

After Workflow C generates your article and you publish it:

1. Copy the published URL
2. Use Prompt 3 above to request indexing
3. Track in your dashboard

**Tip:** Make indexing part of your publishing checklist:

- [ ] Article reviewed and edited
- [ ] Published to website
- [ ] Indexed via SEO Boost API

### Before Workflow R (Rank Improvement)

If you are about to optimize existing content:

1. Check if the page is already indexed (Google Search Console)
2. If not indexed, submit via Workflow I first
3. Wait for indexing confirmation
4. Then proceed with Workflow R optimizations

**Why this order matters:** Optimizing a page Google has not indexed is wasted effort. Ensure discoverability first.

---

## Troubleshooting

### "Invalid URL" Error

**Cause:** URL format is incorrect or page is not accessible.

**Fix:**

- Ensure URL starts with `https://` or `http://`
- Test URL in a browser (must load successfully)
- Check for typos in the domain or path
- Ensure page is not behind a login or paywall

### "Insufficient Credits" Error

**Cause:** Not enough credits for the batch size.

**Fix:**

- Check your credit balance in the dashboard
- Reduce batch size to match available credits
- Purchase additional credits if needed

### Page Still Not Appearing in Google

**Timeline:** Indexing requests typically take 24-72 hours to process.

**If not indexed after 1 week:**

- Verify the URL is accessible to search engines
- Check robots.txt is not blocking the page
- Ensure no "noindex" meta tag on the page
- Try submitting again

---

## Dashboard Tracking

Your SEO Boost dashboard shows:

**Batch History:**

- Batch name and date
- Number of URLs submitted
- Credits used
- Status (completed, partial, failed)

**URL Details:**

- Click any batch to see individual URLs
- Copy URLs for reference
- Track submission timestamps

Use the dashboard to:

- Verify submissions went through
- Review your indexing history
- Track credit usage over time

---

## Summary

Workflow I accelerates Google discovery of your content:

- **What:** Submit URLs for faster Google indexing
- **When:** After publishing new content or significant updates
- **Cost:** 1 credit per URL submitted
- **Timeline:** 24-72 hours for typical indexing

**The Complete Flow:**

```
Create Content (Workflow C)
    |
    v
Publish to Website
    |
    v
Submit for Indexing (Workflow I)
    |
    v
Google Indexes Page (24-72 hours)
    |
    v
Monitor & Optimize (Workflow R if needed)
```

Stop waiting weeks for Google to find your content. Request indexing and get discovered faster.
