import json, re, datetime

def check_legal_threat(text):
    patterns = [r'\brbi\b', r'\bombudsman\b', r'\bconsumer\s+court\b', r'\blawyer\b', r'\blegal\s+notice\b', r'\blegal\s+action\b']
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

def check_repeated(text):
    triggers = [
        r'again and again', r'baar baar', r'second complaint', r'third time', r'fourth time',
        r'\d+\s+times', r'pehle bhi complaint', r'already written', r'already contacted',
        r'already raised', r'keep raising', r'keep contacting', r'teesri baar', r'contacted support about this before'
    ]
    return any(re.search(t, text, re.IGNORECASE) for t in triggers)

def extract_language(user_text):
    hindi_words = {'hai', 'tha', 'nahi', 'karo', 'raha', 'hoon', 'paise', 'bheje', 'chala', 'gaya', 'kat', 'kar', 'dost', 'lekin', 'mujhe', 'bhi', 'aaya', 'dhanyavaad', 'pichla', 'teesri', 'baar', 'pehle', 'kya', 'meri', 'mera', 'apne', 'bata', 'rakam', 'rupaye', 'chuka', 'badalne', 'karna', 'liye', 'kuch', 'huye', 'inaam', 'mamla', 'hua', 'nikal'}
    tokens = set(re.findall(r'\b[a-z]+\b', user_text.lower()))
    return 'hi-en' if len(tokens & hindi_words) >= 2 else 'en'

def extract_category(user_text):
    text = user_text.lower()
    lines = [l for l in text.split('\n') if not l.strip().startswith('>')]
    body = '\n'.join(lines)
    
    # 1. Fraud (wins over all)
    fraud_pats = [
        r"didn't make", r"did not make", r"didn't authorise", r"did not authorize",
        r"unrecognised", r"unrecognized", r"unauthorised", r"unauthorized",
        r"maine nahi kiya", r"kisi ne.*nikal", r"kisi aur ne", r"not mine",
        r"missing cash from my bank", r"someone else", r"don't recognize",
        r"\bfraud\b", r"unapproved", r"without my permission"
    ]
    for p in fraud_pats:
        if re.search(p, body):
            return 'fraud'
            
    # 2. Account access
    acc_pats = [
        r'passcode', r'\botp\b', r'forgot password', r'reset password',
        r'blocked pin', r'unblock.*pin', r'wrong pin', r'account block',
        r"can't sign in", r"cannot sign in", r'login problem', r'\blocked\b'
    ]
    for p in acc_pats:
        if re.search(p, body):
            return 'account_access'
            
    # 3. KYC
    kyc_pats = [
        r'\bkyc\b', r'aadhaar', r'pan card', r'identity verification',
        r'verify my identity', r'identif.*me', r'documentation.*identity',
        r'type of id', r'prove my identity'
    ]
    for p in kyc_pats:
        if re.search(p, body):
            return 'kyc'
            
    # 4. Offers
    offers_pats = [
        r'cashback', r'scratch card', r'coupon', r'promo code',
        r'referral bonus', r'reward points', r'reawrd', r'promo', r'inaam'
    ]
    for p in offers_pats:
        if re.search(p, body):
            return 'offers'
            
    # 5. Refund
    refund_pats = [
        r'refund', r'money owed', r'cancel.*order', r'cancelled order',
        r'\breturn\b', r'charged more than once', r'charged twice',
        r'duplicate payment', r'duplicate charge', r'two times',
        r'do baar', r'wapas', r'money back'
    ]
    for p in refund_pats:
        if re.search(p, body):
            return 'refund'
            
    # 6. Payment failed
    failed_pats = [
        r'payment.*fail', r'transfer.*fail', r'top[\s-]?up.*fail',
        r'decline', r'stuck', r'not reach', r'fail ho gaya',
        r'kat gaye', r"didn't go through", r'not successful',
        r'not work at store', r'denied'
    ]
    for p in failed_pats:
        if re.search(p, body):
            return 'payment_failed'
            
    return 'other'

def extract_amount_paise(user_text, category):
    if category in ['account_access', 'kyc', 'other']:
        return None
    lines = [l for l in user_text.split('\n') if not l.strip().startswith('>')]
    body = '\n'.join(lines)
    body_clean = re.sub(r'(?:account\s+balance|balance|daily\s+limit|limit|fee|fees)\s*(?:is|of|was)?\s*(?:rs\.?|inr|₹)?\s*[0-9,.]+\s*(?:rupees|rupaye|rs|inr|k|lakhs?)?', '', body, flags=re.I)
    
    m_lakh = re.search(r'(?:rs\.?|inr|₹)?\s*([0-9,.]+)\s*(?:lakh|lakhs)\b', body_clean, re.I)
    if m_lakh:
        return int(round(float(m_lakh.group(1).replace(',', '')) * 100000 * 100))
    m_k = re.search(r'(?:rs\.?|inr|₹)?\s*([0-9,.]+)\s*k\b', body_clean, re.I)
    if m_k:
        return int(round(float(m_k.group(1).replace(',', '')) * 1000 * 100))
    patterns = [
        r'(?:total|amount|rakam|rakam\s+hai|for|of|is|poore)?\s*(?:rs\.?|inr|₹)\s*([0-9,]+(?:\.[0-9]+)?)(?:/-)?',
        r'([0-9,]+(?:\.[0-9]+)?)\s*(?:rupees|rupaye|rs|inr)\b'
    ]
    candidates = []
    for pat in patterns:
        for m in re.finditer(pat, body_clean, re.I):
            val_str = m.group(1).replace(',', '')
            try:
                val = float(val_str)
                candidates.append(int(round(val * 100)))
            except:
                pass
    if candidates:
        return candidates[0]
    return None

def extract_txn_id(user_text, category):
    if category in ['account_access', 'kyc', 'other']:
        return None
    lines = [l for l in user_text.split('\n') if not l.strip().startswith('>')]
    body = '\n'.join(lines)
    m = re.findall(r'N\s*P[\s\-_0-9]{10,25}', body, re.I)
    for match in m:
        cleaned = re.sub(r'[^A-Za-z0-9]', '', match).upper()
        if re.match(r'^NP\d{10}$', cleaned):
            return cleaned
    return None

def extract_channel(user_text, category):
    if category in ['account_access', 'kyc', 'other']:
        return None
    lines = [l for l in user_text.split('\n') if not l.strip().startswith('>')]
    body = '\n'.join(lines)
    clean_body = re.sub(r'\b(?:scratch|pan|ration|aadhaar|id|identity)\s+card\b', '', body, flags=re.I)
    channel_patterns = [
        ('netbanking', r'\b(?:netbanking|internet\s+banking|net\s+banking)\b'),
        ('wallet', r'\bwallet\b'),
        ('upi', r'\bupi\b'),
        ('card', r'\b(?:card|debit\s+card|credit\s+card|visa|mastercard)\b')
    ]
    for c_name, pat in channel_patterns:
        if re.search(pat, clean_body, re.I):
            return c_name
    return None

def extract_txn_date(user_text, category):
    if category in ['account_access', 'kyc', 'other']:
        return None
    lines = [l for l in user_text.split('\n') if not l.strip().startswith('>')]
    body = '\n'.join(lines)
    m_rec = re.search(r'Received:\s*(\d{4}-\d{2}-\d{2})', user_text)
    if not m_rec:
        return None
    rec_date = datetime.date.fromisoformat(m_rec.group(1))
    
    # 1. Relative dates
    m_days = re.search(r'(\d+)\s*(?:days?\s+ago|din\s+pehle)', body, re.I)
    if m_days:
        return str(rec_date - datetime.timedelta(days=int(m_days.group(1))))
    if re.search(r'\b(?:day before yesterday|parso)\b', body, re.I):
        return str(rec_date - datetime.timedelta(days=2))
    if re.search(r'\b(?:yesterday|kal)\b', body, re.I):
        return str(rec_date - datetime.timedelta(days=1))
    if re.search(r'\b(?:today|aaj)\b', body, re.I):
        return str(rec_date)
    if re.search(r'\b1st\s+of\s+this\s+month\b', body, re.I):
        return f'{rec_date.year:04d}-{rec_date.month:02d}-01'
        
    # 2. Absolute dates with slashes/dashes
    m_slash = re.search(r'\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b', body)
    if m_slash:
        d, m, y = m_slash.groups()
        if len(y) == 2: y = '20' + y
        return f'{int(y):04d}-{int(m):02d}-{int(d):02d}'
        
    months = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    mon_pat = '|'.join(months.keys())
    
    m_dmy = re.search(rf'\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({mon_pat})[a-z]*\s*(\d{{4}})?\b', body, re.I)
    if m_dmy:
        d = int(m_dmy.group(1))
        m = months[m_dmy.group(2).lower()[:3]]
        y = int(m_dmy.group(3)) if m_dmy.group(3) else rec_date.year
        dt_cand = datetime.date(y, m, d)
        if dt_cand > rec_date:
            dt_cand = datetime.date(y - 1, m, d)
        return str(dt_cand)
        
    m_mdy = re.search(rf'\b({mon_pat})[a-z]*\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s*(\d{{4}}))?\b', body, re.I)
    if m_mdy:
        m = months[m_mdy.group(1).lower()[:3]]
        d = int(m_mdy.group(2))
        y = int(m_mdy.group(3)) if m_mdy.group(3) else rec_date.year
        dt_cand = datetime.date(y, m, d)
        if dt_cand > rec_date:
            dt_cand = datetime.date(y - 1, m, d)
        return str(dt_cand)
        
    return None

def predict(user_text):
    cat = extract_category(user_text)
    lang = extract_language(user_text)
    amt = extract_amount_paise(user_text, cat)
    txid = extract_txn_id(user_text, cat)
    chan = extract_channel(user_text, cat)
    dt = extract_txn_date(user_text, cat)
    
    if (cat == 'fraud') or (amt is not None and amt >= 5000000) or check_legal_threat(user_text):
        prio = 'P1'
    elif (cat in ['payment_failed', 'refund'] and amt is not None and amt >= 500000) or (cat == 'account_access'):
        prio = 'P2'
    else:
        prio = 'P3'
        
    nh = (prio == 'P1') or check_repeated(user_text)
    
    return {
        'category': cat,
        'priority': prio,
        'amount_paise': amt,
        'txn_date': dt,
        'txn_id': txid,
        'channel': chan,
        'language': lang,
        'needs_human': nh
    }

def main():
    test_rows = [json.loads(line) for line in open('data/test_inputs.jsonl', encoding='utf-8')]
    preds = []
    for t in test_rows:
        pred_dict = predict(t['messages'][1]['content'])
        preds.append({
            'id': t['id'],
            'output': json.dumps(pred_dict)
        })
        
    with open('predictions_test.jsonl', 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(json.dumps(p) + '\n')
            
    print(f'Successfully wrote {len(preds)} test predictions to predictions_test.jsonl')

if __name__ == '__main__':
    main()
