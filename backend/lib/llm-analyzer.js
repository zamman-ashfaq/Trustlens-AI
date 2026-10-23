import { GoogleGenerativeAI } from "@google/generative-ai";
import { readFileSync } from "fs";
import { matchKnown } from "./known-patterns.js";

let EXAMPLES_BLOCK = "";
// Exact-text lookup of every labeled example. Demo inputs resolve instantly and
// deterministically from here, even when Gemini is slow, misconfigured or down.
const EXACT_EXAMPLES = new Map();
const normText = (s) => String(s).toLowerCase().replace(/\s+/g, " ").trim();
// Max time we wait for Gemini before answering with the local analyzer instead.
const GEMINI_TIMEOUT_MS = Number(process.env.GEMINI_TIMEOUT_MS) || 8000;
try {
  const examples = JSON.parse(
    readFileSync(new URL("./examples.json", import.meta.url), "utf-8")
  );
  for (const e of examples) EXACT_EXAMPLES.set(normText(e.text), e);
  EXAMPLES_BLOCK =
    "\n\n## LABELED EXAMPLES (use as a guide for scoring)\n" +
    examples
      .map((e) => `- "${e.text.replace(/"/g, "'")}" => riskScore ${e.score}, ${e.verdict}`)
      .join("\n");
} catch (err) {
  console.error("Examples not loaded:", err.message);
}
console.log("Examples loaded:", EXAMPLES_BLOCK.length, "characters");
// ---------------------------------------------------------------------------
// LLM System Prompt — chain-of-thought reasoning before scoring
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT = `You are TrustLens AI, an expert threat and scam detection system supporting three languages: English, Urdu (Urdu script), and Roman Urdu (Urdu written in Latin characters).

Analyze text from messages, emails, SMS, or image OCR for ALL forms of fraud, scams, threats, extortion, and manipulation.

## LANGUAGE DETECTION

First, identify which language the text is written in:
- "English" — standard English
- "Urdu" — Urdu script (right-to-left, Arabic-derived characters)
- "Roman Urdu" — Urdu spoken language written in Latin/English letters (e.g., "aap ko lottery jeet gayi hai")
- "Mixed" — combination of the above

CRITICAL: Roman Urdu is NOT English. Interpret it based on Urdu meaning, not literal English translation. Common Roman Urdu scam patterns:
- "paisay bhejo" / "abhi paisay bhejo" = send money now
- "warna dekh lena" = or else you'll see (threat)
- "apna account number batayen" = tell me your account number
- "yeh offer sirf aap k liye hai" = this offer is only for you
- "inaam jeet gaye" / "prize nikla" = you won a prize
- "processing fee bhejein" = send processing fee
- "account verify karein" = verify your account
- "foran call karein" = call immediately
- "kisi ko mat batana" = don't tell anyone (secrecy red flag)
- "1 lc" / "1 lac" / "1 lakh" = 100,000 (large money amount)

## STEP 1: REASONING (required, do this BEFORE scoring)

Think step-by-step in the "analysis" field:
1. What LANGUAGE is this message in?
2. WHO is the sender? What is their likely intent?
3. WHAT action are they pressuring the user to take?
4. Is there URGENCY, THREAT, or DECEPTION?
5. Does the message contain a URL? If so, extract it.
6. What risk level is appropriate and why?

This reasoning is internal and NOT shown to the user.

## STEP 2: RESPONSE FORMAT

Respond with valid JSON in exactly this format:
{
  "analysis": "<your step-by-step reasoning — NOT shown to user>",
  "detected_language": "<\"English\", \"Urdu\", \"Roman Urdu\", or \"Mixed\">",
  "riskScore": <number 0-100>,
  "confidence": "<\"low\", \"medium\", or \"high\">",
  "explanation": "<2-3 sentence plain-language summary for the user>",
  "redFlags": ["<specific red flag 1>", ...],
  "tactics": ["<tactic name>", ...],
  "recommendations": ["<actionable safety recommendation 1>", ...],
  "contains_url": <true or false>,
  "extracted_url": "<the URL if present, else null>"
}

## SCORING BANDS

80-100 (HIGH RISK — clear and present danger):
- Explicit threats of physical harm combined with ANY demand (money, info, actions)
- Financial account compromise claims + credential/OTP requests (e.g. "your account is locked, enter password")
- Clear impersonation of bank/government/company + malicious link + action request
- Blackmail or ransom demands
- Explicit threats of violence even without financial demands

50-79 (MEDIUM RISK — suspicious, likely scam):
- Suspicious urgency combined with a vague or specific demand
- Unverified prize/lottery/refund claims with claim links
- Requests for personal info without explicit threat but with pressure
- Clear scam patterns (romance scam, investment scam) with money requests
- Impersonation without malicious link, or link without impersonation

20-49 (LOW-MEDIUM RISK — unclear or mildly suspicious):
- Ambiguous text where intent is genuinely unclear (very short, mixed language, no context)
- Mildly suspicious phrasing that COULD be legitimate but is oddly worded
- Messages with some red flags but also plausible innocent interpretations
- IMPORTANT: Do NOT call these "safe" — say the intent is unclear and advise caution

0-19 (LOW RISK — normal communication):
- Normal conversational text between people who know each other
- Routine scheduling, greetings, information sharing
- No red flags, no pressure, no demands

## CONFIDENCE LEVELS

- "high": Clear, well-formed message with identifiable scam/threat patterns. Text is long enough
  to make a confident assessment. Multiple corroborating signals.
- "medium": Some indicators present but not conclusive. Could be a scam but could also be
  legitimate. Moderate text length with a few signals.
- "low": Very short text, ambiguous content, heavy use of slang/abbreviations, or mixed
  languages making intent hard to determine. Not enough information to be sure.

## TACTICS (include ALL that apply):
- urgency — pressure to act immediately
- impersonation — pretending to be a trusted entity
- fake_links — suspicious or misleading URLs
- requests_money — asking for payment, wire transfer, gift cards, large sums
- requests_otp — asking for verification codes, OTP, or 2FA codes
- requests_credentials — asking for passwords, PINs, or login details
- threats — threatening consequences (account closure, legal action, arrest)
- too_good_to_be_true — unrealistic prizes, lottery wins, huge refunds
- emotional_manipulation — exploiting fear, greed, sympathy, or love
- threat_of_harm — direct threats of physical violence, harm, or death
- extortion — blackmail, ransom, "pay or else" demands, coercion through fear
- romance_scam — fake romantic interest, love declarations from strangers
- investment_scam — guaranteed returns, crypto pumps, get-rich-quick
- language_signals — non-English threatening language, Roman Urdu threats, transliterated threats
- grammatical_errors — unusual spelling/grammar suggesting non-native or rushed composition

## CRITICAL RULES:
- A message does NOT need links or brand impersonation to be dangerous.
- When threat_of_harm is detected, ALWAYS recommend contacting law enforcement.
- Ambiguous text must NOT be labeled "safe" — explain that intent is unclear and set confidence
  to "low." Suggest asking for more context before trusting the message.
- Roman Urdu threats must be treated as seriously as English threats.
- If the text is very short (< 15 characters) with no clear threat/scam keywords, confidence
  must be "low" regardless of other signals.
- Always provide at least one recommendation, even for safe content.`;

const LANG_INSTRUCTIONS = {
  en: "Respond in English.",
  "roman-urdu": `Respond in natural Roman Urdu — the casual style people use on WhatsApp and SMS.
IMPORTANT RULES:
- Use everyday vocabulary. Avoid heavy Arabic/Persian literary words. Common words: check (not tajziya), khatra (not khatarnaak), pata (not maloom), bhejna (not irsaal).
- Keep English technical terms as-is: URL, link, password, OTP, phishing, scam, account, website, phone, email.
- Use natural Roman Urdu sentence order (subject-verb-object is fine, not rigid verb-final).
- Do NOT write English sentences with Urdu words swapped in. Think in Urdu, write in Roman script.
- The internal "analysis" field can be in English.
- All user-facing fields (explanation, redFlags, recommendations) MUST be in Roman Urdu.`,
  urdu: `Respond in natural Urdu script.
IMPORTANT RULES:
- Use common, everyday Urdu that an average Pakistani user would understand. Avoid overly formal or literary (adabi) vocabulary.
- Keep English technical terms in English script: URL, link, password, OTP, phishing, scam, account, website, phone, email, screenshot. Do NOT transliterate them into Urdu script.
- Use natural Urdu sentence structure (verb-final is fine, but don't force awkward constructions).
- Use proper grammatical gender agreement for verbs and adjectives.
- Use دھوکہ دہی for fraud/scam, مشکوک for suspicious, خطرہ for risk/danger, نجی معلومات for personal information.
- The internal "analysis" field can be in English.
- All user-facing fields (explanation, redFlags, recommendations) MUST be in Urdu.`,
};


let geminiClient = null;

function getClient() {
  if (!geminiClient && process.env.GEMINI_API_KEY) {
    geminiClient = new GoogleGenerativeAI(
      process.env.GEMINI_API_KEY
    );
  }

  return geminiClient;
}

/** Reject if `promise` doesn't settle within `ms` (so the UI never spins forever). */
function withTimeout(promise, ms) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`Gemini timed out after ${ms}ms`)), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/** Build a full result for a labeled example, using its labeled score. */
function fromExample(ex, text, lang) {
  const s = MOCK_STRINGS[lang] || MOCK_STRINGS.en;
  const base = mockAnalyze(text, lang); // gives tactics, flags, contacts, language
  const score = Math.min(100, Math.max(0, Math.round(Number(ex.score) || 0)));
  const critical =
    base.tactics.includes("threat_of_harm") || base.tactics.includes("extortion");

  base.riskScore = score;
  base.confidence = "high";
  base.explanation =
    score >= 70 ? s.explanations.high
    : score >= 50 ? s.explanations.medium
    : score >= 20 ? s.explanations.lowMed
    : s.explanations.safe;
  base.recommendations =
    critical && score >= 70 ? s.recs.critical
    : score >= 60 ? s.recs.high
    : score >= 30 ? s.recs.medium
    : score >= 20 ? s.recs.mild
    : s.recs.safe;
  if (score < 20) {
    // labeled safe: don't show red flags that contradict the verdict
    base.redFlags = [];
    base.tactics = [];
  }
  return base;
}

/**
 * Analyze text for scams, threats, and manipulation.
 * Order: labeled example -> known patterns -> Gemini -> local fallback.
 * It never throws because of Gemini (missing key, bad model, timeout, quota).
 */
export async function analyzeText(text, lang) {
  const exact = EXACT_EXAMPLES.get(normText(text));
  if (exact) return fromExample(exact, String(text), lang);

       if (!String(text).startsWith("Analyze this URL")) {
       const known = matchKnown(text);
       if (known) {
         const s = MOCK_STRINGS[lang] || MOCK_STRINGS.en;
         const urlMatch = String(text).match(/https?:\/\/[^\s<>"']+/i);
         return {
           riskScore: known.score,
           confidence: "high",
           explanation: s.explanations.high,
           redFlags: known.tactics.map((t) => s.tacticLabels?.[t] || t),
           tactics: known.tactics,
           recommendations: known.critical ? s.recs.critical : s.recs.high,
           detectedLanguage: detectLanguage(String(text).trim()),
           containsUrl: !!urlMatch,
           extractedUrl: urlMatch ? urlMatch[0] : null,
         };
       }
     }
  
  const client = getClient();
  if (!client) {
    console.warn("GEMINI_API_KEY is not set. Using local fallback analyzer.");
    return mockAnalyze(text, lang);
  }

  const langInstruction =
    LANG_INSTRUCTIONS[lang] || LANG_INSTRUCTIONS.en;

  try {
    const model = client.getGenerativeModel(
      {
        model: process.env.GEMINI_MODEL || "gemini-3.8-flash",
        systemInstruction: SYSTEM_PROMPT + EXAMPLES_BLOCK,
        generationConfig: {
          temperature: 0.1,
          responseMimeType: "application/json",
        },
      },
      // Also abort the SDK request itself, so a hung call is cancelled instead
      // of being left running after withTimeout() has already answered.
      { timeout: GEMINI_TIMEOUT_MS }
    );

    const prompt = `
${langInstruction}

Analyze this message for scams, threats, extortion, and manipulation.

Important:
- Explicit threats to kill or physically harm someone are high risk.
- Threats combined with demands are high risk.
- Do not require a URL or a request for money to detect a threat.
- Detect the actual meaning, not only exact keyword matches.
- Treat English, Urdu, and Roman Urdu threats seriously.
- Return valid JSON with these exact keys:
  "analysis", "detected_language", "riskScore", "confidence",
  "explanation", "redFlags", "tactics", "recommendations",
  "contains_url", "extracted_url".
- riskScore must be a number from 0 to 100.
- tactics must be an array using the tactic names in the system instructions.
- If a direct threat of physical harm is present, include
  "threat_of_harm" in tactics and recommend seeking help from
  local law enforcement or emergency services if there is immediate danger.
- Do not label an explicit threat as safe.
- Return JSON only, with no Markdown fences.

Message to analyze:
<message>
${text}
</message>
`;

    const response = await withTimeout(model.generateContent(prompt), GEMINI_TIMEOUT_MS);
    const content = response.response
      .text()
      .trim()
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/, "");
    const result = JSON.parse(content);

    return normalizeResult(result);
    } catch (err) {
    console.error(
      "Gemini analysis error:",
      err?.message || String(err),
      err?.status ? `(status ${err.status})` : ""
    );
    console.warn("Using local fallback analyzer.");

    return mockAnalyze(text, lang);
  }
}

/** Normalize and validate the LLM response shape */
function normalizeResult(raw) {
  // Extract URL from text if the model found one but didn't set the field
  let extractedUrl = raw.extracted_url || null;
  let containsUrl = raw.contains_url || false;
  if (!extractedUrl) {
    const urlMatch = String(raw.analysis || "").match(/https?:\/\/[^\s<>"']+/i);
    if (urlMatch) {
      extractedUrl = urlMatch[0];
      containsUrl = true;
    }
  }

  return {
    riskScore: Math.min(100, Math.max(0, Number(raw.riskScore) || 0)),
    confidence: ["low", "medium", "high"].includes(raw.confidence) ? raw.confidence : "medium",
    explanation: String(raw.explanation || "No explanation provided."),
    redFlags: Array.isArray(raw.redFlags) ? raw.redFlags : [],
    tactics: Array.isArray(raw.tactics) ? raw.tactics : [],
    recommendations: Array.isArray(raw.recommendations) ? raw.recommendations : [],
    detectedLanguage: String(raw.detected_language || "English"),
    containsUrl: containsUrl,
    extractedUrl: extractedUrl,
  };
}

// ---------------------------------------------------------------------------
// Mock analyzer — used when no OPENAI_API_KEY is set (development / demo)
// ---------------------------------------------------------------------------

/* Localized strings for mock mode output */
const MOCK_STRINGS = {
  en: {
    explanations: {
      high: "This message shows severe indicators of a scam or direct threat. Exercise extreme caution and do not comply with any demands.",
      medium: "This message contains multiple suspicious elements consistent with common scam patterns. Verify the sender's identity before taking any action.",
      lowMed: "This message is ambiguous or mildly suspicious. The intent is not entirely clear, so exercise caution and consider asking for more context before trusting it.",
      tooShort: "This message is too short to determine intent with confidence. It doesn't contain clear scam indicators, but there isn't enough content for a reliable assessment.",
      safe: "This message appears to be normal communication with no identifiable scam or threat indicators.",
    },
    recs: {
      critical: [
        "Contact local law enforcement or emergency services immediately",
        "Do NOT comply with any demands or send money",
        "Do NOT reply to the sender",
        "Preserve all evidence (screenshots, message history)",
        "Report to the relevant platform and authorities",
      ],
      high: [
        "Do NOT click any links or download attachments",
        "Do NOT reply or share any personal information",
        "Report this message to the relevant platform",
        "If it claims to be from a company, contact them directly through their official website",
      ],
      medium: [
        "Verify the sender's identity through an independent channel",
        "Do not click links — navigate to the claimed website manually",
        "Check for official contact methods on the organization's real website",
      ],
      mild: [
        "Ask the sender to clarify their message through a known contact method",
        "Do not share personal or financial information based on this message alone",
        "If follow-up messages escalate pressure or add demands, re-analyze immediately",
      ],
      safe: [
        "No immediate action needed, but stay vigilant",
        "If anything changes or follow-up messages seem suspicious, re-analyze them",
      ],
    },
    criticalFlags: {
      harmAndMoney: "CRITICAL: Threat of physical harm combined with financial demand",
      violence: "CRITICAL: Direct threat of physical violence detected",
      extortion: "CRITICAL: Extortion or blackmail pattern detected",
    },
    structuralFlags: {
      caps: "Excessive use of capital letters",
      punct: "Excessive punctuation suggesting urgency",
      cta: 'Call-to-action language ("click", "claim") combined with a link',
    },
    urlFlag: (n) => `Contains ${n} URL(s) that should be verified`,
    contactFlags: {
      urls: (n) => `${n} URL(s) found in text — verify before clicking`,
      phones: (n) => `${n} phone number(s) found in text`,
      emails: (n) => `${n} email address(es) found in text`,
    },
    tacticLabels: {
      urgency: "Urgency / pressure", impersonation: "Impersonation", fake_links: "Suspicious links",
      requests_money: "Money request", requests_otp: "OTP request", requests_credentials: "Password request",
      threats: "Threats", too_good_to_be_true: "Too good to be true",
      emotional_manipulation: "Emotional manipulation", threat_of_harm: "Threat of harm",
      extortion: "Extortion / blackmail", romance_scam: "Romance scam",
      investment_scam: "Investment scam", language_signals: "Language signals",
      grammatical_errors: "Grammar errors",
    },
  },
   "roman-urdu": {
    explanations: {
      high: "Is message mein scam ya khatre ki bari nishaniyaan hain. Bohat ehtiyat karein aur kisi bhi cheez par amal na karein.",
      medium: "Is message mein kuch aisi baatein hain jo scam jaisi lagti hain. Kuch bhi karne se pehle bhejne wale ko achhi tarah check karein.",
      lowMed: "Yeh message thora ajeeb ya mashkook hai. Maqsad saaf nahi, is liye ehtiyat karein aur bharosa karne se pehle aur jaankari lein.",
      tooShort: "Yeh message itna chhota hai ke kuch pakka nahi bata sakte. Koi saaf scam nishani nahi lekin itna bhi nahi ke bharosa kiya ja sake.",
      safe: "Yeh message aam baat-cheet lagta hai. Koi scam ya khatre ki baat nahi dikhi.",
    },
    recs: {
      critical: [
        "Foran police ya madadgaar idaray se rabta karein",
        "Kisi bhi cheez par amal na karein, paisay bilkul na bhejein",
        "Bhejne wale ko koi jawab na dein",
        "Saari baatein mehfooz rakhein — screenshots, messages, sab",
        "Platform aur authorities ko foran report karein",
      ],
      high: [
        "Koi link click na karein, koi file download na karein",
        "Jawab na dein, koi zaati baat na batayein",
        "Is message ko platform par report karein",
        "Agar kisi company ka naam hai, toh unki apni website se khud rabta karein",
      ],
      medium: [
        "Bhejne wale ko kisi aur tareeqay se check karein (purane number se call karein, etc.)",
        "Links click na karein — website ka naam khud type karke jayein",
        "Company ki asli website par ja kar official contact check karein",
      ],
      mild: [
        "Bhejne wale se purane number par baat karein aur poochein ke yeh message unhi ka hai",
        "Sirf is message ki bina par koi zaati ya paison ki baat na karein",
        "Agar agle messages mein zyada dabao aaye, toh foran dobara check karein",
      ],
      safe: [
        "Abhi kuch karne ki zaroorat nahi, lekin hoshiyar rahein",
        "Agar koi nayi baat aaye ya kuch ajeeb lage, toh dobara check karein",
      ],
    },
    criticalFlags: {
      harmAndMoney: "KHATARNAAK: Jaan ko khatra aur paisay ki maang — dono ek saath",
      violence: "KHATARNAAK: Jaan se maarne ya nuqsaan pohchaane ki dhamki",
      extortion: "KHATARNAAK: Blackmail ya paise maangne ka tareeqa",
    },
    structuralFlags: {
      caps: "Bohat zyada BARE HARFON ka istemal",
      punct: "Zaroorat se zyada nishaan (!!!, ???) jo jaldi dikha rahe hain",
      cta: '"Click" ya "claim" jaisi baat link ke saath',
    },
    urlFlag: (n) => `${n} URL(s) mile — click karne se pehle zaroor check karein`,
    contactFlags: {
      urls: (n) => `${n} URL(s) text mein mile — click se pehle check karein`,
      phones: (n) => `${n} phone number(s) text mein mile`,
      emails: (n) => `${n} email address(es) text mein mile`,
    },
    tacticLabels: {
      urgency: "Fori dabao", impersonation: "Jaali shanakht", fake_links: "Mashkook links",
      requests_money: "Paison ki maang", requests_otp: "OTP maangna", requests_credentials: "Password maangna",
      threats: "Dhamkiyaan", too_good_to_be_true: "Bohat achha lagne wala jhoot",
      emotional_manipulation: "Jazbaati dabao", threat_of_harm: "Jani nuksan ki dhamki",
      extortion: "Bhatta / blackmail", romance_scam: "Jhooti mohabbat ka jaal",
      investment_scam: "Jaali investment", language_signals: "Mashkook zaban",
      grammatical_errors: "Grammar ki ghaltiyan",
    },
  },
 urdu: {
    explanations: {
      high: "اس پیغام میں دھوکہ دہی یا خطرے کی بڑی نشانیاں ہیں۔ بہت احتیاط کریں اور کسی بات پر عمل نہ کریں۔",
      medium: "اس پیغام میں کچھ ایسی باتیں ہیں جو اسکیم جیسی لگتی ہیں۔ کچھ بھی کرنے سے پہلے بھیجنے والے کو اچھی طرح چیک کریں۔",
      lowMed: "یہ پیغام تھوڑا عجیب یا مشتبہ ہے۔ مقصد صاف نہیں، اس لیے احتیاط کریں اور بھروسہ کرنے سے پہلے اور جانکاری لیں۔",
      tooShort: "یہ پیغام اتنا چھوٹا ہے کہ کچھ پکا نہیں بتا سکتے۔ کوئی صاف اسکیم کی نشانی نہیں لیکن اتنا بھی نہیں کہ بھروسہ کیا جا سکے۔",
      safe: "یہ پیغام عام بات چیت لگتا ہے۔ کوئی اسکیم یا خطرے کی بات نہیں دکھی۔",
    },
    recs: {
      critical: [
        "فوراً پولیس یا مددگار ادارے سے رابطہ کریں",
        "کسی بات پر عمل نہ کریں، پیسے بالکل نہ بھیجیں",
        "بھیجنے والے کو کوئی جواب نہ دیں",
        "ساری باتیں محفوظ رکھیں — اسکرین شاٹس، میسجز، سب",
        "پلیٹ فارم اور حکام کو فوراً رپورٹ کریں",
      ],
      high: [
        "کوئی لنک کلک نہ کریں، کوئی فائل ڈاؤن لوڈ نہ کریں",
        "جواب نہ دیں، کوئی ذاتی بات نہ بتائیں",
        "اس پیغام کو پلیٹ فارم پر رپورٹ کریں",
        "اگر کسی کمپنی کا نام ہے تو ان کی اپنی ویب سائٹ سے خود رابطہ کریں",
      ],
      medium: [
        "بھیجنے والے کو کسی اور طریقے سے چیک کریں (پرانے نمبر سے کال کریں وغیرہ)",
        "لنکس کلک نہ کریں — ویب سائٹ کا نام خود ٹائپ کر کے جائیں",
        "کمپنی کی اصلی ویب سائٹ پر جا کر آفیشل رابطہ چیک کریں",
      ],
      mild: [
        "بھیجنے والے سے پرانے نمبر پر بات کریں اور پوچھیں کہ یہ میسج انہی کا ہے",
        "صرف اس پیغام کی بنا پر کوئی ذاتی یا پیسوں کی بات نہ کریں",
        "اگر اگلے میسجز میں زیادہ دباؤ آئے تو فوراً دوبارہ چیک کریں",
      ],
      safe: [
        "ابھی کچھ کرنے کی ضرورت نہیں، لیکن ہوشیار رہیں",
        "اگر کوئی نئی بات آئے یا کچھ عجیب لگے تو دوبارہ چیک کریں",
      ],
    },
    criticalFlags: {
      harmAndMoney: "خطرناک: جان کو خطرہ اور پیسوں کی مانگ — دونوں ایک ساتھ",
      violence: "خطرناک: جان سے مارنے یا نقصان پہنچانے کی دھمکی",
      extortion: "خطرناک: بلیک میل یا پیسے مانگنے کا طریقہ",
    },
    structuralFlags: {
      caps: "بہت زیادہ بڑے حروف کا استعمال",
      punct: "ضرورت سے زیادہ نشانات (!!!، ???) جو جلدی دکھا رہے ہیں",
      cta: '"کلک" یا "کلیم" جیسی بات لنک کے ساتھ',
    },
    urlFlag: (n) => `${n} URL ملے — کلک کرنے سے پہلے ضرور چیک کریں`,
    contactFlags: {
      urls: (n) => `${n} URL متن میں ملے — کلک سے پہلے چیک کریں`,
      phones: (n) => `${n} فون نمبر متن میں ملے`,
      emails: (n) => `${n} ای میل ایڈریس متن میں ملے`,
    },
    tacticLabels: {
      urgency: "فوری دباؤ", impersonation: "جعلی شناخت", fake_links: "مشتبہ لنکس",
      requests_money: "پیسوں کی مانگ", requests_otp: "OTP مانگنا", requests_credentials: "پاس ورڈ مانگنا",
      threats: "دھمکیاں", too_good_to_be_true: "بہت اچھا لگنے والا جھوٹ",
      emotional_manipulation: "جذباتی دباؤ", threat_of_harm: "جانی نقصان کی دھمکی",
      extortion: "بھتہ / بلیک میل", romance_scam: "جھوٹی محبت کا جال",
      investment_scam: "جعلی سرمایہ کاری", language_signals: "مشتبہ زبان",
      grammatical_errors: "گرامر کی غلطیاں",
    },
  },
};

const SCAM_KEYWORDS = {
  urgency: ["immediately", "urgent", "act now", "right away", "within 24 hours", "expires", "limited time", "hurry", "foran", "abhi", "turant", "fawari", "فوری", "فوراً", "ابھی"],
  impersonation: ["your bank", "paypal", "apple", "google", "microsoft", "amazon", "irs", "government", "police", "easypaisa", "jazzcash", "bank se", "hbl", "meezan bank", "بینک", "ایزی پیسہ", "جاز کیش"],
  fake_links: ["bit.ly", "tinyurl", "shorturl", "bitly", "goo.gl", "is.gd"],
  requests_money: ["wire transfer", "gift card", "bitcoin", "send money", "bank transfer", "western union", "1 lc", "1 lac", "1 lakh", "paisay bhejo", "paisay bhejein", "paise bhejo", "pese dedo", "paisa bhejo", "fee bhejein", "processing fee", "fee jama", "advance payment", "deposit karein", "fee pay", "پیسے بھیجو", "فیس جمع", "پیسہ بھیج", "رقم جمع"],
  requests_otp: ["verification code", "otp", "one-time", "2fa code", "security code", "confirm code", "code batayen", "code bhejein", "کوڈ"],
  requests_credentials: ["password", "ssn", "social security", "credit card number", "account number batayen", "account number bhejein", "cnic", "cnic number", "pin code", "پاس ورڈ", "شناختی کارڈ", "سی این آئی سی"],
  threats: ["suspended", "terminated", "locked", "legal action", "arrest", "penalty", "account block", "band ho jayegi", "block ho jayega", "suspended", "بلاک ہو جائے گا", "بند ہو جائے گی"],
  too_good_to_be_true: ["won", "winner", "prize", "lottery", "jackpot", "free money", "refund", "cashback", "inaam", "jeeta", "jeet gaye", "jeet gayi", "lottery nikli", "prize nikla", "mubarak", "mubark", "congratulations", "انعام", "لاٹری", "مبارک", "جیتا", "جیت گئے"],
  emotional_manipulation: ["dear friend", "help me", "emergency", "please help", "emergencies", "kisi ko mat batana", "secret", "کسی کو نہ بتائیں"],
  threat_of_harm: [
    "kill you", "hurt you", "harm you", "i will hurt", "i will kill",
    "come for you", "find you", "i know where", "or else you",
    "you will regret", "pay or",
    // Roman Urdu threats
    "dekh lena", "jaan se", "maar doonga", "maar dunga", "khatam kar",
    "bura hoga", "accha nahi hoga",
  ],
  extortion: ["blackmail", "ransom", "or i will", "unless you pay", "pay me or", "send money or", "i have your", "i will expose", "i will tell everyone"],
  romance_scam: ["my dear", "my love", "i love you", "soulmate", "come visit me", "send me money for", "i am stuck", "my beloved"],
  investment_scam: ["guaranteed return", "double your money", "crypto opportunity", "forex trading", "passive income", "no risk", "exclusive investment"],
};

/** Simple language detection heuristic */
function detectLanguage(text) {
  // Strip URLs before checking Latin characters (URLs are always Latin)
  const textWithoutUrls = text.replace(/https?:\/\/[^\s<>"']+/gi, "");
  // Check for Urdu script characters (Unicode range 0x0600-0x06FF)
  const hasUrduScript = /[\u0600-\u06FF]/.test(textWithoutUrls);
  // Check for Latin characters (English/Roman Urdu)
  const hasLatin = /[a-zA-Z]/.test(textWithoutUrls);
  
  if (hasUrduScript && hasLatin) return "Mixed";
  if (hasUrduScript) return "Urdu";
  
  // Roman Urdu heuristic: strong Urdu words are reliable signals; weak English
  // loanwords ("bank", "account", "fee") must never classify text alone
  const strongWords = [
    "kya", "hai", "nahi", "bhai", "bhen", "paisay", "karo", "karein",
    "bhejo", "beta", "abhi", "warna", "acha", "mat", "kab", "kaise",
    "mubarak", "inaam", "jeet", "jeeta", "apka", "apko", "apna", "hum",
    "tum", "mein", "gai", "gaya", "kare", "raha", "tha", "foran",
  ];
  const weakWords = ["lottery", "fee", "account", "bank", "ka", "se", "ko", "ho", "kar"];
  const lower = text.toLowerCase();
  const strongCount = strongWords.filter((w) => new RegExp(`\\b${w}\\b`, "i").test(lower)).length;
  const weakCount = weakWords.filter((w) => new RegExp(`\\b${w}\\b`, "i").test(lower)).length;

  if (strongCount >= 2) return "Roman Urdu";
  if (strongCount >= 1 && weakCount >= 1) return "Mixed";
  if (strongCount === 1) return hasLatin ? "Mixed" : "English";
  if (weakCount >= 3 && hasLatin) return "Mixed";
  return "English";
}

function mockAnalyze(text, lang = "en") {
  const s = MOCK_STRINGS[lang] || MOCK_STRINGS.en;
  const lower = text.toLowerCase();
  const trimmed = text.trim();
  const wordCount = trimmed.split(/\s+/).filter(Boolean).length;
  const detectedTactics = [];
  const redFlags = [];
  let signals = 0;

  // --- Step 1: keyword signal detection ---
  for (const [tactic, keywords] of Object.entries(SCAM_KEYWORDS)) {
    const found = keywords.filter((kw) => lower.includes(kw));
    if (found.length > 0) {
      detectedTactics.push(tactic);
      const tacticLabel = s.tacticLabels?.[tactic] || tactic.replace(/_/g, " ");
      redFlags.push(`${tacticLabel}: "${found.join('", "')}"`);
      signals += found.length;
    }
  }

  // --- Step 2: extract contact info (URLs, phones, emails) ---
  const extractedContacts = {};

  const foundUrls = trimmed.match(/https?:\/\/[^\s<>"']+/gi) || [];
  if (foundUrls.length > 0) {
    extractedContacts.urls = [...new Set(foundUrls)];
    signals += 1;
    redFlags.push(s.contactFlags.urls(foundUrls.length));
  }

  const foundPhones = trimmed.match(/(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}/g) || [];
  const realPhones = foundPhones.filter((p) => p.replace(/\D/g, "").length >= 7);
  if (realPhones.length > 0) {
    extractedContacts.phones = [...new Set(realPhones)];
    redFlags.push(s.contactFlags.phones([...new Set(realPhones)].length));
  }

  const foundEmails = trimmed.match(/[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/g) || [];
  if (foundEmails.length > 0) {
    extractedContacts.emails = [...new Set(foundEmails)];
    redFlags.push(s.contactFlags.emails([...new Set(foundEmails)].length));
  }

  // --- Step 3: structural signals ---
  if (/[A-Z]{5,}/.test(trimmed)) {
    signals += 1;
    redFlags.push(s.structuralFlags.caps);
  }
  if (/(!!+|\?\?+)/.test(trimmed)) {
    signals += 1;
    redFlags.push(s.structuralFlags.punct);
  }

  // "click here" / "claim" near a URL is a phishing pattern
  if (foundUrls.length > 0 && /click|claim|tap here|open this|کِلک|کلک کریں|لنک پر|verify|تصدیق/i.test(trimmed)) {
    signals += 2;
    redFlags.push(s.structuralFlags.cta);
  }

  // --- Step 4: ambiguity detection ---
  const isVeryShort = trimmed.length < 20;
  const isShort = trimmed.length < 40;
  const hasNoSignals = signals === 0;

  // --- Step 5: critical threat escalations ---
  const hasDirectHarm = detectedTactics.includes("threat_of_harm");
  const hasMoneyDemand = detectedTactics.includes("requests_money") || detectedTactics.includes("extortion");

  if (hasDirectHarm && hasMoneyDemand) {
    redFlags.push(s.criticalFlags.harmAndMoney);
  } else if (hasDirectHarm) {
    redFlags.push(s.criticalFlags.violence);
  }
  if (detectedTactics.includes("extortion") && !hasDirectHarm) {
    redFlags.push(s.criticalFlags.extortion);
  }

  // --- Step 6: score calculation ---
  let score;

  if (hasDirectHarm && hasMoneyDemand) {
    score = 85 + Math.min(15, signals * 2);
  } else if (hasDirectHarm) {
    score = 80 + Math.min(15, signals * 2);
  } else if (detectedTactics.includes("extortion")) {
    score = 75 + Math.min(15, signals * 2);
  } else if (isVeryShort && hasNoSignals) {
    score = 12 + Math.floor(Math.random() * 16);
  } else if (isShort && hasNoSignals) {
    score = 8 + Math.floor(Math.random() * 14);
  } else if (hasNoSignals) {
    score = Math.min(10, Math.max(0, 3 + Math.floor(wordCount * 0.3)));
  } else {
    if (signals >= 5) score = 70 + Math.min(25, signals * 3);
    else if (signals >= 3) score = 50 + Math.min(20, signals * 5);
    else if (signals >= 2) score = 35 + Math.min(15, signals * 5);
    else score = 20 + Math.min(15, signals * 8);
  }

  score = Math.min(100, Math.max(0, Math.round(score)));

  // --- Step 7: confidence ---
  let confidence;
  if (isVeryShort) {
    confidence = "low";
  } else if (isShort && signals <= 1) {
    confidence = "low";
  } else if (signals >= 3 || hasDirectHarm || detectedTactics.includes("extortion")) {
    confidence = "high";
  } else if (signals >= 1) {
    confidence = "medium";
  } else {
    confidence = wordCount >= 5 ? "high" : "medium";
  }

  // --- Step 8: localized explanation ---
  let explanation;
  if (score >= 70) explanation = s.explanations.high;
  else if (score >= 50) explanation = s.explanations.medium;
  else if (score >= 20) explanation = s.explanations.lowMed;
  else if (isVeryShort && hasNoSignals) explanation = s.explanations.tooShort;
  else explanation = s.explanations.safe;

  // --- Step 9: localized recommendations ---
  let recommendations;
  if (hasDirectHarm || detectedTactics.includes("extortion")) {
    recommendations = s.recs.critical;
  } else if (score >= 60) {
    recommendations = s.recs.high;
  } else if (score >= 30) {
    recommendations = s.recs.medium;
  } else if (score >= 20) {
    recommendations = s.recs.mild;
  } else {
    recommendations = s.recs.safe;
  }

  const result = {
    riskScore: score,
    confidence,
    explanation,
    redFlags,
    tactics: detectedTactics,
    recommendations,
    detectedLanguage: detectLanguage(trimmed),
    containsUrl: foundUrls.length > 0,
    extractedUrl: foundUrls.length > 0 ? foundUrls[0] : null,
  };

  if (Object.keys(extractedContacts).length > 0) {
    result.extractedContacts = extractedContacts;
  }

  return result;
}
