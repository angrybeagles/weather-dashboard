# SuperCode Node - Essential Knowledge for n8n

## What is SuperCode?
**SuperCode** is a community node (`@kenkaiii/n8n-nodes-supercode.superCodeNodeVmSafe`) that provides **47+ production-ready JavaScript libraries** in n8n workflows. It solves the fundamental limitation of n8n's built-in Code node: zero external libraries.

**Install**: Settings → Community Nodes → `@kenkaiii/n8n-nodes-supercode`

## Complete Library Arsenal (47 Libraries)

### Data Processing
`lodash` (_), `dayjs`, `moment-timezone`, `date-fns`, `bytes`, `ms`, `uuid`, `nanoid`

### Validation & Parsing
`joi`, `validator`, `Ajv`, `yup`, `zod` (z), `qs`

### Files & Documents
`XLSX` (xlsx), `pdf-lib`, `csv-parse`, `papaparse` (Papa), `archiver`, `ini`, `toml`

### Web & HTTP
`axios`, `cheerio`, `FormData`

### Text & Content
`handlebars`, `marked`, `html-to-text`, `xml2js`, `XMLParser`, `YAML`, `pluralize`, `slug`, `string-similarity`, `fuse.js`

### Security & Crypto
`crypto-js` (CryptoJS), `jsonwebtoken` (jwt), `bcryptjs`, `node-forge`

### Specialized
`QRCode`, `jimp`, `mathjs`, `iban`, `libphonenumber-js`, `currency.js`

### Natural Language
`franc-min`, `compromise`

### Async Control
`p-retry`, `p-limit`

### Blockchain
`ethers`, `web3`

### Media Processing
`@distube/ytdl-core` (ytdl), `fluent-ffmpeg`, `ffmpeg-static`

**Critical**: All libraries are **pre-loaded as globals** - no `require()` needed!

### ✅ When to Use SuperCode
Use SuperCode when you need:
1. **Multiple sequential API calls** - Loops with HTTP requests (axios)
2. **Complex HTTP operations** - Better error handling, interceptors
3. **Advanced date manipulation** - Beyond basic Date objects
4. **Data transformations** - lodash utilities for complex operations
5. **Standard JS patterns** - When $http.request() is too limiting
6. **Excel/CSV processing** - Read/write/manipulate spreadsheets (XLSX, papaparse)
7. **Email/phone validation** - Professional validation libraries (joi, validator)
8. **JWT/Security** - Token generation, password hashing (jsonwebtoken, bcryptjs)
9. **Web scraping** - Parse HTML properly (cheerio)
10. **PDF manipulation** - Read/create PDFs (pdf-lib)
11. **QR codes** - Generate QR codes (QRCode)
12. **Image processing** - Manipulate images (jimp)
13. **Phone formatting** - International phone numbers (libphonenumber-js)
14. **Workflow consolidation** - Replace 10-15 nodes with one SuperCode node

### 🎯 Production Impact
Real metrics from actual workflows:
- **Node reduction**: 80-92% fewer nodes (15 nodes → 1 node)
- **Execution speed**: 74% faster (4.2s → 1.1s)
- **Memory usage**: 73% less (248MB → 67MB)

### ⚠️ When Built-in Code is Sufficient
Stick with built-in Code node for:
- Simple data transformations
- Expression evaluations
- Basic array/object manipulation
- Single $http.request() calls

## Configuration Pattern

```json
{
  "type": "@kenkaiii/n8n-nodes-supercode.superCodeNodeVmSafe",
  "parameters": {
    "code": "// Your JavaScript code here\n// Can use: axios, dayjs, lodash\nreturn results;"
  },
  "typeVersion": 1
}
```

## Critical Usage Notes

### Global Library Access
**No require() needed** - All libraries are pre-loaded as globals:
```javascript
// ✅ Correct - just use them directly
const grouped = _.groupBy(data, 'category');
const token = jwt.sign(payload, secret);
const $ = cheerio.load(html);

// ❌ Wrong - don't use require()
const _ = require('lodash');  // Not needed!
```

### Accessing Previous Node Data
```javascript
// Get data from any node in workflow
const webhookData = $('Webhook').first().json;
const apiResponse = $('HTTP Request').all();
const previousNode = $('Previous Node Name').first().json;
```

### HTTP Requests with Axios
```javascript
// Use axios instead of $http
const response = await axios.get('https://api.example.com/endpoint', {
  params: { key: 'value' },
  headers: { 'Authorization': 'Bearer token' }
});

const data = response.data;
```

### Output Structure Gotcha
**IMPORTANT**: SuperCode may output data with nested structure:
```javascript
// SuperCode output structure
{
  json: {
    json: {
      actualData: "here"
    }
  }
}
```

**Solution**: Add transform node after SuperCode to unwrap:
```javascript
// In next Code node
return $input.all().map(item => ({
  json: item.json.json,  // Unwrap the nested structure
  pairedItem: item.pairedItem
}));
```

### Loop Patterns with API Calls
SuperCode excels at sequential API operations:
```javascript
const results = [];
let currentId = startId;

while (currentId && results.length < maxItems) {
  const response = await axios.get(`/api/resource/${currentId}`);
  results.push(response.data);
  
  // Find next ID in chain
  currentId = findNext(response.data);
  
  // Rate limiting
  await new Promise(resolve => setTimeout(resolve, 200));
}

return results.map(item => ({ json: item }));
```

## Common Patterns

### 1. Chain Following API Calls
```javascript
// Fetch linked resources sequentially
let current = startResource;
const chain = [];

while (current) {
  const response = await axios.get(`/api/${current.id}`);
  chain.push(response.data);
  current = response.data.next;  // Follow the chain
}

return chain.map(item => ({ json: item }));
```

### 2. Date Formatting with dayjs
```javascript
// Better date handling
const formatted = dayjs(item.timestamp)
  .format('YYYY-MM-DD HH:mm:ss');
```

### 3. Data Transformation with lodash
```javascript
// Complex grouping and manipulation
const grouped = _.groupBy(items, 'category');
const sorted = _.orderBy(items, ['priority', 'date'], ['desc', 'asc']);
```

## Troubleshooting

### Issue: "Library not defined"
**Cause**: Using wrong library name or not using SuperCode node
**Solution**: 
- Verify exact library name from list above (case-sensitive)
- Ensure you're using SuperCode node, not regular Code node
- Libraries are globals, no `require()` needed

### Issue: Data not reaching next node
**Cause**: Output structure mismatch (nested json.json)
**Solution**: Add transform node to unwrap structure or output directly as `{ json: data }`

### Issue: "fetch is not defined"
**Cause**: Using fetch() instead of axios in SuperCode
**Solution**: Replace `fetch()` with `axios.get()` or `axios.post()`

### Issue: "Cannot read property of undefined"
**Cause**: Data doesn't exist or wrong path
**Solution**: Use optional chaining `data?.property?.value` and check data exists first

### Issue: Slow first execution
**Cause**: SuperCode loads all libraries on first run (1-2 seconds)
**Solution**: Normal behavior - subsequent runs are fast. Don't create multiple SuperCode nodes if one can do it all.

### Issue: Performance degradation
**Cause**: Creating multiple SuperCode nodes unnecessarily
**Solution**: Consolidate logic into fewer SuperCode nodes - one node can often handle everything

## Performance Considerations

### Rate Limiting
Always add delays in loops:
```javascript
await new Promise(resolve => setTimeout(resolve, 200)); // 200ms delay
```

### Error Handling
Wrap API calls in try-catch:
```javascript
try {
  const response = await axios.get(url);
  results.push(response.data);
} catch (error) {
  console.log(`Error: ${error.message}`);
  // Continue or break based on needs
}
```

### Max Iterations
Set safety limits:
```javascript
const maxDepth = 50;  // Prevent infinite loops
let iteration = 0;

while (condition && iteration < maxDepth) {
  // ... processing
  iteration++;
}
```

## When to Recommend SuperCode

### ✅ Recommend SuperCode For:
- "I need to process Excel/CSV files"
- "I need to validate emails/phones properly"
- "I need to generate JWTs or hash passwords"
- "I need to follow a chain of API calls"
- "The Code node gives 'fetch/axios is not defined' error"
- "I need to scrape/parse HTML or XML"
- "I need to generate QR codes or PDFs"
- "I need to loop through API responses"
- "I have 10+ nodes that should be simpler"
- "I need date formatting beyond Date object"
- "I need to make multiple sequential HTTP requests"
- "I need lodash/underscore utilities"

### ❌ Don't Recommend For:
- Simple transformations (use Set or Edit Fields)
- Single HTTP calls (use HTTP Request node)
- Basic array operations (use built-in Code)
- When user hasn't hit any limitations yet
- n8n Cloud users (community nodes not supported)
- Complete beginners who've never seen JavaScript

## Tips for Getting Started

1. **Start Simple**: Try one library at a time. `lodash` for data manipulation is easiest.

2. **No require()**: All libraries are pre-loaded globals:
   ```javascript
   // ✅ Just use them
   const result = _.groupBy(data, 'category');
   const validated = joi.string().email().validate(email);
   ```

3. **Access Any Node**: Get data from any previous node:
   ```javascript
   const webhookData = $('Webhook').first().json;
   const apiResponse = $('HTTP Request').all();
   ```

4. **Use AI for Code**: ChatGPT/Claude can write SuperCode:
   - Tell it you're using SuperCode in n8n
   - Show it the library list
   - Specify libraries are pre-loaded as globals

5. **Test in Preview**: Use n8n's preview feature to test before full runs

6. **Check Examples**: Look at official use cases for patterns

## Integration with Workflow

### Typical Flow
```
Trigger → SuperCode (API chain) → Transform (unwrap) → Action Node
```

### Post-SuperCode Transform
Often needed to prepare data for next node:
```javascript
// Transform SuperCode output for Google Sheets/Airtable/etc.
const items = $input.first().json.results || [];
return items.map(item => ({
  json: {
    field1: item.data.field1,
    field2: item.data.field2
  }
}));
```

## Summary

**SuperCode = Built-in Code + 47 JavaScript Libraries**

### Key Capabilities
- **Data**: lodash, dayjs, moment, date-fns, uuid
- **Validation**: joi, validator, yup, zod
- **Files**: XLSX (Excel), papaparse (CSV), pdf-lib (PDF)
- **Web**: axios (HTTP), cheerio (HTML scraping)
- **Security**: jsonwebtoken (JWT), bcrypt (passwords), crypto-js
- **Specialized**: QRCode, jimp (images), libphonenumber-js
- **Media**: fluent-ffmpeg, ytdl-core
- **Blockchain**: ethers, web3

### When to Use
- Excel/CSV/PDF processing
- Email/phone/data validation
- JWT/security operations
- Web scraping/parsing
- Complex sequential API calls
- Workflow consolidation (10-15 nodes → 1 node)
- Any "impossible in Code node" scenario

### Key Differences from Built-in Code
| Feature | Built-in Code | SuperCode |
|---------|--------------|-----------|
| External libraries | ❌ None | ✅ 47+ libraries |
| HTTP requests | $http.request() | axios (full-featured) |
| Excel files | ❌ Impossible | ✅ XLSX read/write |
| JWT/crypto | ❌ Impossible | ✅ Full crypto suite |
| Validation | Manual regex | joi/validator/yup |
| Date handling | Basic Date | dayjs/moment |
| Performance | Baseline | 70-90% faster for complex tasks |

### Installation
Settings → Community Nodes → `@kenkaiii/n8n-nodes-supercode`

### Remember
- ✅ All libraries are pre-loaded globals (no require())
- ✅ Access any previous node: `$('Node Name').first().json`
- ✅ First run loads libraries (1-2s), then fast
- ✅ Watch for nested output structures (may need transform)
- ✅ Can replace 10-15 nodes with one SuperCode node
- ✅ Production-ready for complex automations
- ⚠️ Not available on n8n Cloud (self-hosted only)

**Built by Ken Kai for workflows that need real JavaScript power.**