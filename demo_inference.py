"""Interactive or CLI triage demo tool for NimbusPay.

Usage:
    # 1. Run on built-in realistic test tickets:
    python demo_inference.py

    # 2. Test a custom ticket directly:
    python demo_inference.py --ticket "Received: 2026-08-14\\nSource: email\\n\\nI will go to consumer court if my 60000 refund is not processed."

    # 3. Interactive prompt mode:
    python demo_inference.py --interactive
"""

import sys, json, argparse
import generate_test_predictions as engine

SAMPLES = [
    {
        "name": "Sample 1: Fraud with NP Transaction ID (Hinglish)",
        "text": """Received: 2026-07-06 (Mon)
Source: app-chat

Bhai, Kisi ne OTP maang ke mere paise nikal liye, yeh maine nahi kiya. App version 6.3.7 hai. Yeh 6 din pehle hua. Transaction ID np6090913363 hai. Please help karo."""
    },
    {
        "name": "Sample 2: Legal Threat & High Amount Escalation",
        "text": """Received: 2026-09-14 (Mon)
Source: ivr-transcript

[IVR transcript] Hi, please help my transfer go through. Rs 618.75 is the amount. Otherwise I will approach the consumer court. This happened 3 days ago. Please help."""
    },
    {
        "name": "Sample 3: Duplicate Charge Refund (High Amount >= ₹50,000)",
        "text": """Received: 2026-08-24 (Mon)
Source: app-chat

Namaste, Merchant bol raha hai refund kar diya, lekin mujhe mila nahi. Baar baar complaint kar raha hoon. Txn id np2010582959. Total 162434 rupaye ka tha. Yeh 24 Aug ko hua. Jaldi solve karo please."""
    },
    {
        "name": "Sample 4: KYC Verification (Non-Transaction Ticket)",
        "text": """Received: 2026-07-09 (Thu)
Source: ivr-transcript

[IVR transcript]  What is the purpose for verifying my identity?"""
    }
]

def triage_and_explain(ticket_text):
    pred = engine.predict(ticket_text)
    
    # Explanations of rules
    rules_triggered = []
    if pred['category'] == 'fraud':
        rules_triggered.append("Category 'fraud' wins over every other category (unrecognized transaction).")
    if pred['priority'] == 'P1':
        if pred['category'] == 'fraud':
            rules_triggered.append("Priority 'P1' triggered because category is fraud.")
        elif pred['amount_paise'] and pred['amount_paise'] >= 5000000:
            rules_triggered.append("Priority 'P1' triggered because amount >= ₹50,000 (5,000,000 paise).")
        elif engine.check_legal_threat(ticket_text):
            rules_triggered.append("Priority 'P1' triggered because customer made a legal threat (court/lawyer/RBI/ombudsman).")
    elif pred['priority'] == 'P2':
        if pred['category'] in ['payment_failed', 'refund'] and pred['amount_paise'] and pred['amount_paise'] >= 500000:
            rules_triggered.append("Priority 'P2' triggered because refund/payment_failed amount >= ₹5,000.")
        elif pred['category'] == 'account_access':
            rules_triggered.append("Priority 'P2' triggered because category is account_access.")
            
    if pred['needs_human']:
        if pred['priority'] == 'P1':
            rules_triggered.append("needs_human is true because priority is P1.")
        elif engine.check_repeated(ticket_text):
            rules_triggered.append("needs_human is true because customer mentioned contacting support repeatedly.")
            
    if pred['category'] in ['account_access', 'kyc', 'other']:
        rules_triggered.append("Transaction fields (amount, date, id, channel) strictly set to null for non-transaction category.")

    return pred, rules_triggered

def print_result(title, text, pred, rules):
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print("INPUT TICKET:")
    print(text.strip())
    print("\nMODEL OUTPUT JSON:")
    print(json.dumps(pred, indent=2))
    print("\nBUSINESS RULES APPLIED:")
    for r in rules:
        print(f"  * {r}")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="NimbusPay Ticket Triage Demo")
    parser.add_argument("--ticket", type=str, help="Ticket text to triage")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt mode")
    args = parser.parse_args()

    if args.ticket:
        pred, rules = triage_and_explain(args.ticket)
        print_result("Custom Ticket", args.ticket, pred, rules)
        return

    if args.interactive:
        print("Enter ticket text below. Press Ctrl+C or enter empty line when done:\n")
        lines = []
        try:
            while True:
                line = input()
                if not line and lines:
                    break
                lines.append(line)
        except (KeyboardInterrupt, EOFError):
            pass
        ticket = "\n".join(lines)
        if ticket.strip():
            pred, rules = triage_and_explain(ticket)
            print_result("Interactive Ticket", ticket, pred, rules)
        return

    # Default: Run built-in sample tests
    print("Running NimbusPay Triage Engine Demo on Built-In Samples:\n")
    for s in SAMPLES:
        pred, rules = triage_and_explain(s["text"])
        print_result(s["name"], s["text"], pred, rules)

if __name__ == '__main__':
    main()
