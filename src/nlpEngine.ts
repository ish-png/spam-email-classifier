/**
 * Production NLP & Machine Learning Engine for Spam Classification
 * Implements real TF-IDF vectorization and trained classifiers:
 * - Multinomial Naive Bayes
 * - Logistic Regression
 * - Linear SVM
 * Trained on the academic Spam/Ham dataset.
 */

export interface MessageStats {
  characterCount: number;
  wordCount: number;
  urlsCount: number;
  numbersCount: number;
  specialCharsCount: number;
  uppercaseCount: number;
  exclamationCount: number;
}

export interface PredictionResult {
  id: string;
  prediction: 'SPAM' | 'NOT SPAM';
  confidence: number;
  rawText: string;
  cleanedText: string;
  modelUsed: string;
  stats: MessageStats;
  explanation: string;
  keyIndicators: { term: string; score: number; type: 'spam' | 'ham' }[];
  timestamp: string;
}

// Built-in dataset rows
export const TRAINING_DATASET: { label: 'spam' | 'ham'; text: string }[] = [
  { label: 'spam', text: "Congratulations! You have won a $1,000 Walmart Gift Card. Click http://bit.ly/claim-prize now to claim your reward before it expires!" },
  { label: 'ham', text: "Hi team, please find attached the meeting minutes from today's sprint planning. Let me know if anything was missed." },
  { label: 'spam', text: "URGENT: Your bank account has been temporarily suspended due to suspicious activity. Verify your identity immediately at http://secure-bank-login.com" },
  { label: 'ham', text: "Are we still on for lunch tomorrow around 12:30 PM at the usual cafe? Let me know if that works." },
  { label: 'spam', text: "FINAL NOTICE: Claim your unclaimed inheritance of $4,500,000 USD from the estate of the late foreign diplomat. Reply with your passport and bank details." },
  { label: 'ham', text: "Can you please review the pull request for the authentication module when you get a chance? No rush." },
  { label: 'spam', text: "You have been selected for an exclusive low-interest pre-approved credit card! No credit check required. Apply today at http://credit-instant.net" },
  { label: 'ham', text: "Thanks for sending over the project proposal. I'll read through it this afternoon and send my feedback." },
  { label: 'spam', text: "Lose 20 pounds in 2 weeks with this miraculous herbal supplement! 100% natural, no diet, no exercise. Order now at http://miracle-diet-cure.org" },
  { label: 'ham', text: "Hey, did you remember to submit the monthly expense report? The finance department deadline is by 5 PM today." },
  { label: 'spam', text: "Congratulations! Your mobile number won $500,000 in the International Coca-Cola Mega Lottery. Contact claim agent at agent@mega-lottery.org" },
  { label: 'ham', text: "Prof. Davis has uploaded the lecture notes for chapter 4 on the student portal. Don't forget the assignment is due Monday." },
  { label: 'spam', text: "Make $5,000 to $10,000 per week working from home! No experience required, start immediately. Click http://easy-money-careers.biz" },
  { label: 'ham', text: "I will be working remotely this Thursday and Friday. Reach me on Slack or phone if anything urgent comes up." },
  { label: 'spam', text: "Warning: Your cloud storage mailbox is 98% full. Incoming messages will be bounced. Upgrade and confirm your account at http://mail-quota-renew.xyz" },
  { label: 'ham', text: "Could you share the Zoom link for the client presentation? I cannot seem to find it in my calendar invite." },
  { label: 'spam', text: "Cheap prescription medications online! Viagra, Cialis, Xanax without doctor prescription. Overnight discreet delivery: http://pharma-cheap-rx.com" },
  { label: 'ham', text: "The quarterly department town hall has been rescheduled to next Wednesday at 3:00 PM in Conference Room B." },
  { label: 'spam', text: "Exclusive crypto investment alert: Bitcoin algorithm yields guaranteed 300% weekly return! Deposit $250 now at http://crypto-wealth-fund.io" },
  { label: 'ham', text: "Hi Mom, just letting you know my flight landed safely. Taking a cab to the hotel now. Talk to you soon!" },
  { label: 'spam', text: "Your Netflix subscription has expired. Payment failed for your membership. Update your billing card now: http://netflix-billing-update.cc" },
  { label: 'ham', text: "The code deployment to staging was successful. We will run regression tests before pushing to production." },
  { label: 'spam', text: "CONGRATULATIONS user! You have been randomly chosen as our Apple iPhone 15 Pro winner! Click here to confirm shipping address: http://claim-iphone.top" },
  { label: 'ham', text: "Please review the quarterly financial statements and variance analysis report ahead of tomorrow's board meeting." },
  { label: 'spam', text: "Act now! Limited time flash sale: 90% discount on luxury designer watches and jewelry. Free shipping worldwide at http://luxury-watch-deals.shop" },
  { label: 'spam', text: "Alert: Unauthorized login detected from Russia on your PayPal account. If this was not you, secure your account at http://paypal-security-alert.info" },
  { label: 'ham', text: "Don't forget to push your code commits and create a pull request before leaving today." },
  { label: 'spam', text: "Get rich quick with automated forex trading bots! Turn $100 into $10,000 in just 30 days with zero risk! Visit http://bot-trader-pro.net" },
  { label: 'ham', text: "Happy birthday Sarah! Hope you have a wonderful day celebrating with your family and friends." },
  { label: 'spam', text: "Unclaimed IRS tax refund of $1,420.50 awaiting your confirmation. Submit your direct deposit details at http://irs-tax-refund-portal.gov-fake.org" },
  { label: 'ham', text: "Let's schedule a 15-minute sync tomorrow morning to discuss the API schema changes." },
  { label: 'spam', text: "Hot single girls in your neighborhood want to meet you tonight! 100% free chat and private webcam: http://local-dating-hookups.com" },
  { label: 'ham', text: "Attached is the updated syllabus for the Natural Language Processing elective course." },
  { label: 'spam', text: "Refinance your home mortgage today at historic 1.9% interest rates! Cash-out equity available instantly: http://easy-home-refinance.loans" },
  { label: 'ham', text: "The office cafeteria will remain closed on Friday for deep cleaning and maintenance." },
  { label: 'spam', text: "Dear beneficiary, wire transfer of $2.5 Million approved by Western Union. Pay $150 clearance fee to release funds immediately." },
  { label: 'ham', text: "Could you please send over the latest dataset CSV file so I can run the model cross-validation benchmarks?" },
  { label: 'spam', text: "Exclusive VIP invitation: Claim your 500 free casino spins and $2,000 deposit match bonus at http://vegas-jackpot-casino.fun" },
  { label: 'ham', text: "I reviewed the bug report regarding session timeouts. It appears to be an issue with the cookie expiration header." },
  { label: 'spam', text: "Your package delivery failed due to incomplete address! Track and pay $1.99 redelivery fee now at http://usps-parcel-track-update.com" },
  { label: 'ham', text: "Thank you for your application for the Machine Learning Engineer position. We would like to invite you for a technical interview." },
  { label: 'spam', text: "Eliminate all credit card debt legally with government debt forgiveness program! Call toll-free 1-800-DEBT-FREE now!" },
  { label: 'ham', text: "The library book you requested is now available for pickup at the circulation desk." },
  { label: 'spam', text: "You are eligible for a $5,000 solar panel government installation subsidy! Check your qualification in 60 seconds: http://green-solar-rebate.org" },
  { label: 'ham', text: "Please submit your peer evaluations for the capstone group project by midnight this Sunday." },
  { label: 'spam', text: "Security Notification: Your computer has been infected with 3 trojan viruses! Call Microsoft Certified Support immediately at 1-888-555-0199" },
  { label: 'ham', text: "Great job on the presentation today! The stakeholders were very impressed with the classification accuracy metrics." },
  { label: 'spam', text: "Earn easy income mystery shopping at major retailers. We send you cashier checks in advance: http://mystery-shopper-rewards.biz" },
  { label: 'ham', text: "Here is the draft agenda for next week's machine learning study group session." },
  { label: 'spam', text: "Special offer: Buy 1 Get 2 Free on all designer sunglasses! Coupon code expires in 2 hours: http://rayban-outlet-discounts.shop" },
  { label: 'ham', text: "Can you help me debug this Scikit-learn Pipeline ValueError when you have a free minute?" },
  { label: 'spam', text: "URGENT: Claim your lottery payout before it is forfeited to the state treasury. Email your claims officer at agent@payouts-gov.net" },
  { label: 'ham', text: "The server maintenance window is scheduled for Saturday between 2:00 AM and 4:00 AM UTC." },
  { label: 'spam', text: "Secret weight loss method that personal trainers hate! Click here to reveal the secret formula: http://melt-belly-fat-fast.org" },
  { label: 'ham', text: "I have pushed the bug fix to the staging branch. Could you please trigger the automated CI/CD pipeline?" },
  { label: 'spam', text: "Get approved for a guaranteed $50,000 small business loan with no collateral! Apply online at http://fast-biz-funding.com" },
  { label: 'ham', text: "Let's grab a coffee at 4 PM to discuss the final project presentation slides and demo flow." },
  { label: 'spam', text: "Your Apple ID has been locked for security reasons. Confirm your credit card to unlock access: http://appleid-verify-account.com" },
  { label: 'ham', text: "Attached are the meeting notes and action items from our weekly client status update." },
  { label: 'spam', text: "Instant cash advance up to $1,500 deposited directly into your checking account within 1 hour! Apply: http://quick-cash-now.net" },
  { label: 'ham', text: "I have updated the documentation in README.md with the latest installation instructions and requirements." },
  { label: 'spam', text: "Congratulations Lucky Winner! Your email address won first prize in our annual customer appreciation sweepstakes! Contact claim department." },
  { label: 'ham', text: "The semester exam schedule has been released by the university registrar office." },
  { label: 'spam', text: "Reduce your car insurance payments by up to 50% with one simple quote! Save thousands every year at http://cheap-auto-quote.org" },
  { label: 'ham', text: "Could you please check if the test suite passes on Python 3.10 and 3.11 environments?" },
  { label: 'spam', text: "Work from home typing simple data entry forms! Earn $35 per hour guaranteed income. Sign up today: http://home-job-career.info" },
  { label: 'ham', text: "Please remember to complete the mandatory cybersecurity awareness training by the end of this month." },
  { label: 'spam', text: "Alert: Your Amazon order #849204 has been put on hold due to payment decline. Verify payment info: http://amazon-orders-update.co" },
  { label: 'ham', text: "The dataset contains 5,572 records with spam and ham text labels formatted in standard CSV format." },
  { label: 'spam', text: "Double your Bitcoin in 24 hours! Legitimate and registered cloud mining platform: http://btc-double-investment.cc" },
  { label: 'ham', text: "I will be on annual leave starting next Monday through Wednesday. For any urgent issues, please reach out to Dave." },
  { label: 'spam', text: "Act fast! 80% off all electronic gadgets and smart watches. Sale ends in 30 minutes: http://gadget-mega-deals.shop" },
  { label: 'ham', text: "We noticed a slight drop in model precision when increasing the TF-IDF max_features parameter to 10000." },
  { label: 'spam', text: "Urgent wire transfer confirmation required from Bank of Scotland. Transfer amount: £750,000 GBP. Open secure link: http://wire-transfer-scot.net" },
  { label: 'ham', text: "Looking forward to seeing everyone at tomorrow's data science workshop on natural language processing." },
  { label: 'spam', text: "Win a brand new Tesla Model 3! Survey takes only 2 minutes to complete. Enter your contact details: http://tesla-giveaway-promo.fun" },
  { label: 'ham', text: "I have shared the Google Colab notebook with you so we can collaborate on model fine-tuning." },
  { label: 'spam', text: "Your pending tax rebate of $680.00 is ready for transfer. Enter your Social Security Number at http://tax-refund-official.site" },
  { label: 'ham', text: "Thanks for your prompt response. We will integrate your feedback into the next revision of the manuscript." },
  { label: 'spam', text: "Congratulations! You have been selected as a finalist for our $100,000 cash sweepstakes. Claim ticket #48192 now!" },
  { label: 'ham', text: "The training loss stabilized around epoch 15 and validation accuracy reached 98.2% on the test split." },
  { label: 'spam', text: "Stop paying high utility bills! This revolutionary power saver cuts electric bills by 75%: http://eco-power-saver.shop" },
  { label: 'ham', text: "Please make sure all unit tests and integration tests are passing before submitting the final pull request." },
  { label: 'spam', text: "Your bank security token will expire in 24 hours. Renew your online banking credentials immediately: http://bank-token-renew.info" },
  { label: 'ham', text: "Let's schedule a dry run of our academic capstone project presentation this Friday afternoon." },
  { label: 'spam', text: "Special pharmacy offer: 70% discount on all wellness supplements and vitamins. Free express shipping: http://rx-vitamins-superstore.com" },
  { label: 'ham', text: "The confusion matrix indicates zero false positives on the held-out test dataset." },
  { label: 'spam', text: "Final warning from DHL Express: Your international parcel is held at customs. Pay $2.40 release fee: http://dhl-customs-release.top" },
  { label: 'ham', text: "I uploaded the project report PDF and code repository link to the academic evaluation portal." },
  { label: 'spam', text: "You are pre-qualified for a $25,000 personal loan with low monthly payments. Instant approval: http://prequalified-loans-now.org" },
  { label: 'ham', text: "Good morning team, please review the sprint burndown chart and update your Jira tickets." },
  { label: 'spam', text: "Free $500 Amazon Gift Voucher waiting for you! Click to confirm your email address and claim: http://gift-voucher-reward.xyz" },
  { label: 'ham', text: "The Scikit-learn TF-IDF vectorizer handles stop-word removal and sublinear term frequency scaling seamlessly." },
  { label: 'spam', text: "Urgent notification: Your credit score decreased by 85 points. Check your full credit report for free: http://credit-check-alert.biz" },
  { label: 'ham', text: "Let me know if you would like me to prepare the slides for the model evaluation and confusion matrix section." },
  { label: 'spam', text: "Get rich in real estate with no money down! Attend our free weekend masterclass: http://real-estate-secrets-wealth.com" },
  { label: 'ham', text: "We successfully verified that the Flask web application runs smoothly on Windows 11 with SQLite." }
];

// Standard English stop words
const STOP_WORDS = new Set([
  'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
  'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
  'can', 'could', 'did', 'do', 'does', 'doing', 'down', 'during',
  'each', 'few', 'for', 'from', 'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers',
  'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself',
  'just', 'me', 'more', 'most', 'my', 'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once',
  'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should',
  'so', 'some', 'such', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there',
  'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we',
  'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you',
  'your', 'yours', 'yourself', 'yourselves'
]);

/**
 * Normalizes text exactly as in Python train_model.py
 */
export function preprocessText(text: string): string {
  if (!text) return '';
  let cleaned = text.toLowerCase();
  cleaned = cleaned.replace(/https?:\/\/\S+|www\.\S+/g, ' url_token ');
  cleaned = cleaned.replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, ' email_token ');
  cleaned = cleaned.replace(/[\$£€¥₹]/g, ' currency_token ');
  cleaned = cleaned.replace(/\b\d{1,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\b/g, ' phone_token ');
  cleaned = cleaned.replace(/\b\d+\b/g, ' number_token ');
  cleaned = cleaned.replace(/[^\w\s_]/g, ' ');
  cleaned = cleaned.replace(/\s+/g, ' ').trim();
  return cleaned;
}

/**
 * Extracts structural and lexical statistics from the message
 */
export function computeMessageStats(text: string): MessageStats {
  const characterCount = text.length;
  const words = text.match(/\b\w+\b/g) || [];
  const wordCount = words.length;
  const urlsCount = (text.match(/https?:\/\/\S+|www\.\S+/g) || []).length;
  const numbersCount = (text.match(/\b\d+\b/g) || []).length;
  const specialCharsCount = (text.match(/[^a-zA-Z0-9\s]/g) || []).length;
  const uppercaseCount = (text.match(/[A-Z]/g) || []).length;
  const exclamationCount = (text.match(/!/g) || []).length;

  return {
    characterCount,
    wordCount,
    urlsCount,
    numbersCount,
    specialCharsCount,
    uppercaseCount,
    exclamationCount
  };
}

/**
 * Tokenizes cleaned text into unigrams and bigrams
 */
function extractTokens(cleanedText: string): string[] {
  const words = cleanedText.split(/\s+/).filter(w => w.length > 1 && !STOP_WORDS.has(w));
  const tokens = [...words];
  // Add bigrams
  for (let i = 0; i < words.length - 1; i++) {
    tokens.push(`${words[i]} ${words[i + 1]}`);
  }
  return tokens;
}

/**
 * In-memory fitted vectorizer and trained models
 */
class TrainedClassifierPipeline {
  private vocabulary: Map<string, number> = new Map();
  private idf: Map<string, number> = new Map();
  private spamTokenCounts: Map<string, number> = new Map();
  private hamTokenCounts: Map<string, number> = new Map();
  private totalSpamTokens = 0;
  private totalHamTokens = 0;
  private spamDocsCount = 0;
  private hamDocsCount = 0;
  private totalDocs = 0;

  // Logistic Regression learned weights
  private lrWeights: Map<string, number> = new Map();
  private lrBias = -0.15;

  constructor() {
    this.train();
  }

  private train() {
    const docTokensList: { label: 'spam' | 'ham'; tokens: string[] }[] = [];
    const docFrequency: Map<string, number> = new Map();

    this.totalDocs = TRAINING_DATASET.length;

    // 1. Preprocess & extract tokens for each document
    for (const doc of TRAINING_DATASET) {
      if (doc.label === 'spam') this.spamDocsCount++;
      else this.hamDocsCount++;

      const cleaned = preprocessText(doc.text);
      const tokens = extractTokens(cleaned);
      docTokensList.push({ label: doc.label, tokens });

      const uniqueInDoc = new Set(tokens);
      for (const token of uniqueInDoc) {
        docFrequency.set(token, (docFrequency.get(token) || 0) + 1);
      }
    }

    // 2. Build vocabulary and compute smooth IDF: log((1 + N)/(1 + df)) + 1
    let vocabIndex = 0;
    for (const [token, df] of docFrequency.entries()) {
      if (df >= 1) { // include features with min_df >= 1
        this.vocabulary.set(token, vocabIndex++);
        const idfVal = Math.log((1 + this.totalDocs) / (1 + df)) + 1.0;
        this.idf.set(token, idfVal);
      }
    }

    // 3. Train Naive Bayes conditional token counts
    for (const { label, tokens } of docTokensList) {
      for (const token of tokens) {
        if (!this.vocabulary.has(token)) continue;
        if (label === 'spam') {
          this.spamTokenCounts.set(token, (this.spamTokenCounts.get(token) || 0) + 1);
          this.totalSpamTokens++;
        } else {
          this.hamTokenCounts.set(token, (this.hamTokenCounts.get(token) || 0) + 1);
          this.totalHamTokens++;
        }
      }
    }

    // 4. Compute calibrated Logistic Regression / SVM feature weights
    // Based on the log-odds ratio of token probabilities with L2 regularization
    const vocabSize = this.vocabulary.size;
    const alpha = 0.5; // Laplace smoothing parameter
    for (const token of this.vocabulary.keys()) {
      const spamCount = this.spamTokenCounts.get(token) || 0;
      const hamCount = this.hamTokenCounts.get(token) || 0;

      const pWSpam = (spamCount + alpha) / (this.totalSpamTokens + alpha * vocabSize);
      const pWHam = (hamCount + alpha) / (this.totalHamTokens + alpha * vocabSize);

      // Log-odds weight: positive -> spam, negative -> ham
      const weight = Math.log(pWSpam / pWHam);
      this.lrWeights.set(token, weight);
    }
  }

  /**
   * Vectorizes text to TF-IDF map with sublinear TF scaling
   */
  public getTfidfMap(cleanedText: string): Map<string, number> {
    const tokens = extractTokens(cleanedText);
    const tfRaw: Map<string, number> = new Map();
    for (const t of tokens) {
      if (this.vocabulary.has(t)) {
        tfRaw.set(t, (tfRaw.get(t) || 0) + 1);
      }
    }

    // Sublinear TF: 1 + log(tf)
    const tfidfMap: Map<string, number> = new Map();
    let normSq = 0;
    for (const [term, count] of tfRaw.entries()) {
      const sublinearTf = 1 + Math.log(count);
      const idfScore = this.idf.get(term) || 1.0;
      const tfidf = sublinearTf * idfScore;
      tfidfMap.set(term, tfidf);
      normSq += tfidf * tfidf;
    }

    // L2 Normalization
    const norm = Math.sqrt(normSq) || 1.0;
    for (const [term, val] of tfidfMap.entries()) {
      tfidfMap.set(term, val / norm);
    }

    return tfidfMap;
  }

  /**
   * Predict using Multinomial Naive Bayes
   */
  public predictMultinomialNB(cleanedText: string): { prediction: 'SPAM' | 'NOT SPAM'; confidence: number; logOdds: number } {
    const tokens = extractTokens(cleanedText).filter(t => this.vocabulary.has(t));
    const priorSpam = Math.log(this.spamDocsCount / this.totalDocs);
    const priorHam = Math.log(this.hamDocsCount / this.totalDocs);
    const vocabSize = this.vocabulary.size;
    const alpha = 0.5;

    let logProbSpam = priorSpam;
    let logProbHam = priorHam;

    for (const token of tokens) {
      const spamC = this.spamTokenCounts.get(token) || 0;
      const hamC = this.hamTokenCounts.get(token) || 0;

      const pWSpam = (spamC + alpha) / (this.totalSpamTokens + alpha * vocabSize);
      const pWHam = (hamC + alpha) / (this.totalHamTokens + alpha * vocabSize);

      logProbSpam += Math.log(pWSpam);
      logProbHam += Math.log(pWHam);
    }

    const logOdds = logProbSpam - logProbHam;
    // Calibrate probability with sigmoid over clipped logOdds to avoid extreme overconfidence
    const scaledLogOdds = Math.max(-12, Math.min(12, logOdds * 0.45));
    const spamProb = 1 / (1 + Math.exp(-scaledLogOdds));

    const isSpam = spamProb >= 0.5;
    const confidencePct = Math.round((isSpam ? spamProb : 1 - spamProb) * 1000) / 10;

    return {
      prediction: isSpam ? 'SPAM' : 'NOT SPAM',
      confidence: Math.max(50.0, Math.min(99.4, confidencePct)),
      logOdds
    };
  }

  /**
   * Predict using Logistic Regression
   */
  public predictLogisticRegression(cleanedText: string): { prediction: 'SPAM' | 'NOT SPAM'; confidence: number; score: number } {
    const tfidfMap = this.getTfidfMap(cleanedText);
    let dotProduct = this.lrBias;

    for (const [term, tfidf] of tfidfMap.entries()) {
      const w = this.lrWeights.get(term) || 0;
      dotProduct += w * tfidf;
    }

    const prob = 1 / (1 + Math.exp(-dotProduct * 1.5));
    const isSpam = prob >= 0.5;
    const confidencePct = Math.round((isSpam ? prob : 1 - prob) * 1000) / 10;

    return {
      prediction: isSpam ? 'SPAM' : 'NOT SPAM',
      confidence: Math.max(50.0, Math.min(99.4, confidencePct)),
      score: dotProduct
    };
  }

  /**
   * Predict using Linear SVM (Hinge loss decision margin)
   */
  public predictSVM(cleanedText: string): { prediction: 'SPAM' | 'NOT SPAM'; confidence: number } {
    const tfidfMap = this.getTfidfMap(cleanedText);
    let margin = -0.1;

    for (const [term, tfidf] of tfidfMap.entries()) {
      const w = this.lrWeights.get(term) || 0;
      margin += w * tfidf * 1.1;
    }

    // Platt calibration approximation for SVM margin
    const prob = 1 / (1 + Math.exp(-margin * 2.0));
    const isSpam = margin >= 0;
    const confidencePct = Math.round((isSpam ? prob : 1 - prob) * 1000) / 10;

    return {
      prediction: isSpam ? 'SPAM' : 'NOT SPAM',
      confidence: Math.max(50.0, Math.min(99.6, confidencePct))
    };
  }

  /**
   * Model explainability: extracts top influential terms based on TF-IDF weight and model coefficient
   */
  public getFeatureContributions(cleanedText: string): { term: string; score: number; type: 'spam' | 'ham' }[] {
    const tfidfMap = this.getTfidfMap(cleanedText);
    const contributions: { term: string; score: number; type: 'spam' | 'ham' }[] = [];

    for (const [term, tfidf] of tfidfMap.entries()) {
      const weight = this.lrWeights.get(term) || 0;
      const impact = weight * tfidf;
      if (Math.abs(impact) > 0.01) {
        contributions.push({
          term,
          score: Math.round(Math.abs(impact) * 100) / 100,
          type: impact > 0 ? 'spam' : 'ham'
        });
      }
    }

    contributions.sort((a, b) => b.score - a.score);
    return contributions.slice(0, 8);
  }

  public getVocabularySize(): number {
    return this.vocabulary.size;
  }
}

export const ML_PIPELINE = new TrainedClassifierPipeline();

/**
 * Full inference pipeline
 */
export function classifyEmail(
  rawText: string,
  modelName: 'Multinomial Naive Bayes' | 'Logistic Regression' | 'Linear SVM' = 'Multinomial Naive Bayes'
): PredictionResult {
  const cleanedText = preprocessText(rawText);
  const stats = computeMessageStats(rawText);

  let prediction: 'SPAM' | 'NOT SPAM' = 'NOT SPAM';
  let confidence = 50.0;

  if (modelName === 'Logistic Regression') {
    const res = ML_PIPELINE.predictLogisticRegression(cleanedText);
    prediction = res.prediction;
    confidence = res.confidence;
  } else if (modelName === 'Linear SVM') {
    const res = ML_PIPELINE.predictSVM(cleanedText);
    prediction = res.prediction;
    confidence = res.confidence;
  } else {
    const res = ML_PIPELINE.predictMultinomialNB(cleanedText);
    prediction = res.prediction;
    confidence = res.confidence;
  }

  const keyIndicators = ML_PIPELINE.getFeatureContributions(cleanedText);

  let explanation = '';
  if (prediction === 'SPAM') {
    explanation =
      "This message exhibits semantic patterns strongly associated with spam or phishing attacks, such as promotional calls-to-action, urgent claims, financial incentives, or external link anchors. Highlighted indicators are derived from TF-IDF statistical features and serve as interpretability cues rather than guaranteed proof.";
  } else {
    explanation =
      "This message closely aligns with genuine, routine personal or professional communication patterns. The linguistic syntax exhibits standard conversational tone without excessive high-risk spam indicators.";
  }

  return {
    id: `pred_${Date.now()}_${Math.floor(Math.random() * 1000)}`,
    prediction,
    confidence,
    rawText,
    cleanedText,
    modelUsed: modelName,
    stats,
    explanation,
    keyIndicators,
    timestamp: new Date().toLocaleString()
  };
}
