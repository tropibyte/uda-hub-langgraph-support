"""Regenerates data/external/cultpass_articles.jsonl.

The first four articles are the ones CultPass shipped with the starter, kept
verbatim. The other 25 were written for this project so the knowledge base
covers every ticket category the classifier can emit (see
agentic/design/ARCHITECTURE.md, "Ticket taxonomy").

Run from the solution folder:  python scripts/build_articles.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "external" / "cultpass_articles.jsonl"

ORIGINAL = [
    {"title": "How to Reserve a Spot for an Event", "content": "If a user asks how to reserve an event:\n\n- Guide them to the CultPass app\n- Instruct them to browse the experience catalog and tap 'Reserve'\n- If it's a premium or limited event, check if reservation confirmation is required via email\n- Remind them to arrive at least 15 minutes early with their QR code visible\n\n**Suggested phrasing:**\n\"You can reserve an experience by opening the CultPass app, selecting your desired event, and tapping 'Reserve'. Be sure to arrive 15 minutes early with your QR code ready.\"", "tags": "reservation, events, booking, attendance"},
    {"title": "What's Included in a CultPass Subscription", "content": "Each user is entitled to 4 cultural experiences per month, which may include:\n- Art exhibitions\n- Museum entries\n- Music concerts\n- Film screenings and more\n\nSome premium experiences may require an additional fee (visible in the app).\n\n**Suggested phrasing:**\n\"Your CultPass subscription includes 4 curated experiences each month. You can choose from museums, concerts, film events and more. Premium events may have an extra cost, which is shown during reservation.\"", "tags": "subscription, benefits, pricing, access"},
    {"title": "How to Cancel or Pause a Subscription", "content": "Users can manage their subscription via the app > 'My Account' > 'Manage Plan'.\n\n- Cancel: Effective at the end of the billing cycle\n- Pause: Preserves user data, resumes automatically when reactivated\n- Do NOT offer refunds unless approved by support lead\n\n**Suggested phrasing:**\n\"You can cancel or pause your subscription at any time via the 'My Account' section in the CultPass app. Cancelation takes effect at the end of your billing cycle.\"", "tags": "cancelation, pause, subscription, billing"},
    {"title": "How to Handle Login Issues?", "content": "Most login issues are resolved with password reset:\n\n- Ask the user to tap 'Forgot Password' on the login screen\n- Ensure they are using the correct registered email\n- If they did not receive the reset email, check spam folder or retry after 10 minutes\n- For persistent login issues, escalate to human support\n\n**Suggested phrasing:**\n\"Try tapping 'Forgot Password' on the login screen. Make sure you're using the email associated with your account. If the email doesn't arrive, check spam or try again in a few minutes.\"", "tags": "login, password, access, escalation"},
]


def article(title, lines, phrasing, tags):
    body = "\n".join(lines) + "\n\n**Suggested phrasing:**\n\"" + phrasing + "\""
    return {"title": title, "content": body, "tags": tags}


NEW = [
    # ---- billing ---------------------------------------------------------
    article(
        "Payment Failed or Card Declined",
        [
            "When a user reports a failed payment or a declined card:",
            "",
            "- Ask them to confirm the card on file is valid and not expired in 'My Account' > 'Payment Methods'",
            "- A failed renewal is retried automatically 3 times over 7 days",
            "- During the retry window the subscription stays active; after the third failure it is set to 'cancelled'",
            "- Updating the card triggers an immediate retry",
            "- Never ask the user to send card numbers in chat or email",
        ],
        "It looks like your last payment didn't go through. You can update your card in 'My Account' > 'Payment Methods' and we'll retry the charge right away. Your plan stays active while we retry over the next 7 days.",
        "billing, payment, card declined, renewal, payment method",
    ),
    article(
        "Double Charge or Unrecognized Charge",
        [
            "When a user reports being charged twice or does not recognize a charge:",
            "",
            "- Check the subscription status and start date with the account lookup tool",
            "- A pending authorization can appear next to the real charge for up to 5 business days and then disappears on its own",
            "- If two settled charges exist for the same billing period, submit a refund request for the duplicate (requires support lead approval)",
            "- If the user does not recognize the charge at all, treat it as possible account compromise and escalate to Trust & Safety",
            "- Never promise a refund before approval",
        ],
        "I'm sorry about the confusion with your charges. A temporary authorization can show up for a few days and then drop off by itself. If you see two completed charges for the same month, I've logged a refund request for the duplicate and our billing lead will review it.",
        "billing, double charge, duplicate, unrecognized charge, refund, dispute",
    ),
    article(
        "Refund Policy and Refund Requests",
        [
            "CultPass refund policy:",
            "",
            "- Subscriptions are billed monthly in advance and are generally non-refundable",
            "- Exceptions: duplicate charges, charges after a confirmed cancellation, or a service outage that prevented use for most of the billing period",
            "- Every refund must be approved by a support lead. Agents submit a refund request; they do NOT confirm or promise the refund",
            "- Approved refunds reach the original payment method within 5-10 business days",
            "- Premium experience fees are refundable if the partner cancels the event",
        ],
        "I've submitted your refund request to our billing lead for review. You'll get an email with the decision, and if it's approved the money returns to your original payment method within 5-10 business days.",
        "refund, billing, policy, approval, money back",
    ),
    article(
        "Updating Your Payment Method or Billing Date",
        [
            "Payment details are managed only by the user in the app:",
            "",
            "- 'My Account' > 'Payment Methods' > 'Add card', then set it as default",
            "- The billing date is the day the subscription started and cannot be moved by support",
            "- Support agents must never collect or type card numbers on the user's behalf",
        ],
        "You can add a new card in 'My Account' > 'Payment Methods' and set it as your default. Your billing date stays the same as the day your subscription started.",
        "billing, payment method, credit card, billing date, update card",
    ),
    article(
        "Gift Cards and Corporate Plans",
        [
            "Gift cards and corporate plans:",
            "",
            "- Gift cards cover 1, 3 or 6 months of Basic or Premium and are redeemed in 'My Account' > 'Redeem Code'",
            "- A redeemed gift card pauses paid billing until the gifted months are used",
            "- Gift card codes cannot be exchanged for cash",
            "- Companies with 10 or more employees can request a corporate plan; route these leads to the partnerships team (escalate with the company name and size)",
        ],
        "You can redeem your gift card in 'My Account' > 'Redeem Code'. Your paid billing pauses automatically until the gifted months are used up.",
        "gift card, corporate, business plan, redeem code, billing",
    ),
    # ---- subscription ----------------------------------------------------
    article(
        "CultPass Plans: Basic vs Premium",
        [
            "CultPass has two tiers:",
            "",
            "- Basic: a monthly quota of standard experiences (the default plan includes 4 per month). Premium experiences carry an extra fee shown at checkout",
            "- Premium: a higher monthly quota, access to premium experiences without the extra fee, and 48-hour early access to new events",
            "- The exact quota for each user is shown in 'My Account' > 'My Plan' and is what the account lookup tool returns as monthly_quota",
        ],
        "With Basic you get your monthly quota of standard experiences, and premium events have a small extra fee. Premium raises your quota, removes the premium fee and gives you 48-hour early access to new events.",
        "subscription, plans, tiers, basic, premium, pricing, quota",
    ),
    article(
        "Upgrading or Downgrading Your Plan",
        [
            "Plan changes:",
            "",
            "- Upgrade (Basic to Premium): takes effect immediately; the price difference is prorated for the current cycle",
            "- Downgrade (Premium to Basic): takes effect at the next billing date so the user keeps Premium benefits until then",
            "- Change it in 'My Account' > 'Manage Plan' > 'Change Tier'",
            "- Reservations already made for premium experiences are kept after a downgrade",
        ],
        "You can change your tier in 'My Account' > 'Manage Plan' > 'Change Tier'. Upgrades start right away with a prorated charge; downgrades start on your next billing date, so you keep Premium until then.",
        "subscription, upgrade, downgrade, change plan, tier, billing",
    ),
    article(
        "How the Monthly Experience Quota Works",
        [
            "Monthly quota rules:",
            "",
            "- Each reservation uses one credit from the monthly quota when it is confirmed",
            "- The quota resets on the billing date each month; unused credits do NOT roll over",
            "- Cancelling a reservation more than 24 hours before the event returns the credit",
            "- A paused or cancelled subscription cannot make new reservations",
            "- Agents can check a user's quota and current reservations with the account lookup tools",
        ],
        "Every reservation uses one credit from your monthly quota, and the quota refreshes on your billing date. Unused credits don't carry over, but if you cancel more than 24 hours ahead we give the credit back.",
        "subscription, quota, credits, monthly limit, reservations, reset",
    ),
    article(
        "Reactivating a Paused or Cancelled Subscription",
        [
            "Reactivation:",
            "",
            "- Paused subscriptions resume from 'My Account' > 'Manage Plan' > 'Resume' and billing restarts that day",
            "- Cancelled subscriptions can be restarted at any time; the user keeps the same account, history and preferences",
            "- A new billing date is set to the reactivation day",
            "- Support agents can resume a paused subscription with the subscription management tool when the user asks for it",
        ],
        "Welcome back! You can resume your plan from 'My Account' > 'Manage Plan' > 'Resume'. Billing restarts today, and all your history and preferences are still there.",
        "subscription, reactivate, resume, restart, pause, cancelled",
    ),
    # ---- reservations ----------------------------------------------------
    article(
        "Cancelling a Reservation and the No-Show Policy",
        [
            "Reservation cancellations:",
            "",
            "- Users cancel in the app under 'My Reservations' > select the event > 'Cancel'",
            "- More than 24 hours before the event: the credit returns to the monthly quota",
            "- Less than 24 hours before, or not attending (no-show): the credit is used",
            "- Three no-shows within 30 days pause new reservations for 7 days",
            "- Agents may cancel a reservation for the user with the reservation tool once the user confirms which one",
        ],
        "You can cancel under 'My Reservations' in the app. If you cancel more than 24 hours before the event, the credit goes straight back to your monthly quota.",
        "reservation, cancel reservation, no-show, credits, policy",
    ),
    article(
        "Event Sold Out and the Waitlist",
        [
            "When an experience has no slots left:",
            "",
            "- The 'Reserve' button becomes 'Join Waitlist'",
            "- If a slot opens, waitlisted users are offered it in order and have 2 hours to confirm",
            "- Joining a waitlist does not use a credit; confirming the offered slot does",
            "- Support cannot add extra slots or skip the waitlist",
        ],
        "That experience is fully booked right now, but you can tap 'Join Waitlist'. If a spot opens, you'll get a notification and have 2 hours to confirm it. The waitlist doesn't use any of your credits.",
        "reservation, sold out, waitlist, availability, slots",
    ),
    article(
        "Premium Experiences and Extra Fees",
        [
            "Premium experiences:",
            "",
            "- Marked with a star in the catalog",
            "- Basic subscribers pay an extra fee shown before confirming; Premium subscribers pay no extra fee",
            "- Premium experiences still use one credit from the monthly quota",
            "- If the partner cancels a premium event, the fee is refunded automatically and the credit is returned",
        ],
        "Premium experiences are marked with a star. On Basic there's a small extra fee shown before you confirm; on Premium there's no extra fee. Either way the booking uses one of your monthly credits.",
        "premium, experiences, extra fee, reservation, pricing",
    ),
    article(
        "Transferring or Sharing a Reservation",
        [
            "Reservation ownership:",
            "",
            "- Reservations are personal and non-transferable; the QR code is tied to the account holder",
            "- Venues may ask for photo ID matching the account name",
            "- To let a friend attend, the user should cancel (more than 24 hours ahead to keep the credit) and the friend books with their own CultPass",
            "- CultPass does not support guest tickets on a single subscription",
        ],
        "Reservations are tied to your account and can't be transferred, and some venues check ID. If a friend wants to go, cancel yours more than 24 hours ahead to keep your credit and they can book with their own CultPass.",
        "reservation, transfer, share, guest, friend, qr code, id",
    ),
    article(
        "Event Cancelled or Rescheduled by the Partner",
        [
            "When a partner venue cancels or reschedules:",
            "",
            "- Users are notified by push and email",
            "- Cancelled: the reservation is set to 'cancelled', the credit is returned and any premium fee is refunded automatically",
            "- Rescheduled: the reservation moves to the new date; if the user cannot attend they can cancel for free regardless of the 24-hour rule",
        ],
        "I'm sorry the event changed. When a partner cancels, your credit and any premium fee come back automatically. If it was rescheduled and the new date doesn't work, you can cancel for free even inside 24 hours.",
        "reservation, event cancelled, rescheduled, partner, refund, credit",
    ),
    # ---- technical -------------------------------------------------------
    article(
        "QR Code Not Scanning at the Venue",
        [
            "When the QR code fails at the entrance:",
            "",
            "- Turn screen brightness to maximum and disable dark mode for the ticket screen",
            "- Open the ticket from 'My Reservations' while online so the code refreshes (codes rotate every 60 seconds)",
            "- Screenshots do not work because the code rotates",
            "- Staff can check the user in manually with the reservation ID shown under the code",
            "- If the reservation is missing from 'My Reservations', escalate with the event name and time",
        ],
        "Please raise your screen brightness and open the ticket from 'My Reservations' while you're online, since the code refreshes every minute and screenshots won't scan. If it still fails, show the staff the reservation ID under the code and they can check you in manually.",
        "technical, qr code, check-in, venue, entry, scanning",
    ),
    article(
        "App Crashes, Freezes or Won't Load",
        [
            "Troubleshooting the CultPass app:",
            "",
            "- Update to the latest version from the App Store or Google Play",
            "- Supported: iOS 15 or later, Android 10 or later",
            "- Force-close and reopen; then clear the app cache (Android) or offload and reinstall (iOS)",
            "- Check the connection; the catalog needs internet access",
            "- If it still crashes after these steps, escalate to technical support with the device model, OS version and app version",
        ],
        "Let's get the app working again. Please update CultPass to the latest version, then force-close and reopen it. If it still crashes, clearing the cache or reinstalling usually fixes it. The app needs iOS 15 or Android 10 or newer.",
        "technical, app crash, freeze, bug, not loading, troubleshooting",
    ),
    article(
        "Not Receiving Emails or Notifications",
        [
            "Missing emails or push notifications:",
            "",
            "- Check the spam or promotions folder and add no-reply@cultpass.com to contacts",
            "- Confirm the email address in 'My Account' > 'Profile' is correct",
            "- Push notifications: enable them in the phone settings for CultPass and in 'My Account' > 'Notifications'",
            "- Reservation confirmations are always available in 'My Reservations' even if the email is missing",
        ],
        "Please check your spam or promotions folder and add no-reply@cultpass.com to your contacts. For push alerts, make sure notifications are on for CultPass in your phone settings and in 'My Account' > 'Notifications'.",
        "technical, email, notifications, push, spam, confirmation",
    ),
    # ---- account ---------------------------------------------------------
    article(
        "Changing Your Email Address or Profile Details",
        [
            "Profile changes:",
            "",
            "- Name, phone and preferences: 'My Account' > 'Profile' > 'Edit'",
            "- Email: 'My Account' > 'Profile' > 'Change Email'; a confirmation link goes to the NEW address and the change applies after it is clicked",
            "- If the user can no longer access the old email and cannot log in, escalate to human support for identity verification",
        ],
        "You can update your details in 'My Account' > 'Profile'. For a new email address, tap 'Change Email' and click the confirmation link we send to the new address.",
        "account, profile, change email, update details, personal information",
    ),
    article(
        "Account Blocked or Suspended",
        [
            "Blocked accounts:",
            "",
            "- Accounts are blocked by Trust & Safety after suspected fraud, payment disputes or repeated policy violations",
            "- A blocked user cannot log in or make reservations",
            "- Support agents CANNOT unblock accounts and must not say why an account was blocked",
            "- Always escalate blocked-account tickets to Trust & Safety with the user ID; response time is up to 2 business days",
        ],
        "I can see your account is currently restricted, and I've passed your case to our Trust & Safety team, who are the only ones able to review it. They'll contact you by email within 2 business days.",
        "account, blocked, suspended, restricted, trust and safety, escalation",
    ),
    article(
        "Deleting Your Account and Personal Data Requests",
        [
            "Account deletion and privacy requests (LGPD/GDPR):",
            "",
            "- Users can request deletion in 'My Account' > 'Privacy' > 'Delete Account'",
            "- Deletion cancels the subscription and is permanent after a 30-day grace period",
            "- Requests for a copy of personal data are handled by the privacy team within 15 days",
            "- Support agents do not delete data themselves; escalate data-access or deletion requests made through support to the privacy team",
        ],
        "You can request deletion in 'My Account' > 'Privacy' > 'Delete Account'. It becomes permanent after a 30-day grace period, and your subscription is cancelled. If you'd like a copy of your data first, our privacy team can send it within 15 days.",
        "account, delete account, privacy, personal data, lgpd, gdpr",
    ),
    article(
        "Suspicious Activity and Account Security",
        [
            "Possible account compromise:",
            "",
            "- Signs: reservations the user did not make, password changed without their action, unrecognized charges",
            "- Tell the user to reset the password immediately and sign out of all devices ('My Account' > 'Security')",
            "- Treat as URGENT and escalate to Trust & Safety with the list of unfamiliar activity",
            "- Do not cancel reservations or change the account until Trust & Safety reviews it",
        ],
        "Please reset your password now and use 'My Account' > 'Security' > 'Sign out of all devices'. I've flagged this as urgent for our Trust & Safety team, who will review the activity on your account.",
        "account, security, hacked, compromised, suspicious activity, fraud, urgent",
    ),
    # ---- general / policy -----------------------------------------------
    article(
        "Accessibility and Special Assistance at Experiences",
        [
            "Accessibility:",
            "",
            "- Each experience page lists accessibility details (step-free access, hearing loops, seating)",
            "- Users can add an assistance note while reserving; the partner receives it with the booking",
            "- A companion who is required for accessibility can attend free at partner venues that support it; this is shown on the experience page",
            "- For needs the page does not cover, escalate so the partnerships team can confirm with the venue before the event",
        ],
        "Every experience page lists its accessibility details, and you can add an assistance note while reserving so the venue knows in advance. If you need something that isn't listed, I can ask our partnerships team to confirm it with the venue.",
        "accessibility, assistance, wheelchair, companion, special needs, general",
    ),
    article(
        "Where CultPass Is Available and How Experiences Are Chosen",
        [
            "Coverage and catalog:",
            "",
            "- CultPass partners with museums, theaters, music venues, tour operators and outdoor-activity providers across Brazil",
            "- The catalog is curated monthly; new experiences appear on the 1st of each month",
            "- Users can filter by city, date, category and premium status",
            "- Suggestions for new partners can be sent through 'Help' > 'Suggest a Venue'",
        ],
        "CultPass works with cultural and outdoor partners across Brazil, and the catalog is refreshed on the 1st of every month. You can filter by city, date and category, and suggest new venues under 'Help' > 'Suggest a Venue'.",
        "general, availability, cities, catalog, partners, brazil",
    ),
    article(
        "Reporting a Problem or Safety Incident at a Venue",
        [
            "Incidents at an experience:",
            "",
            "- Any report of injury, harassment, discrimination or an unsafe venue is URGENT and must be escalated immediately to Trust & Safety",
            "- Do not attempt to resolve it with a knowledge-base answer; acknowledge, apologise, and confirm a person will follow up",
            "- If someone is in immediate danger, tell the user to contact local emergency services (190 police / 192 ambulance in Brazil)",
            "- Minor service complaints (late start, poor experience) can be logged as feedback and the user can rate the experience in the app",
        ],
        "I'm really sorry this happened. I've escalated your report to our Trust & Safety team as urgent, and a member of the team will contact you directly. If anyone is in immediate danger, please call 190 or 192.",
        "safety, incident, harassment, injury, venue complaint, urgent, escalation",
    ),
    article(
        "When and How Support Escalates to a Human Agent",
        [
            "Escalation policy for UDA-Hub agents:",
            "",
            "- Escalate when no knowledge-base article covers the request, when the answer confidence is low, or when the user asks for a person",
            "- Always escalate: blocked accounts, security or fraud concerns, safety incidents, legal threats, chargebacks, refund approvals, privacy/data requests",
            "- Escalated tickets get a handoff summary: the issue, what was already tried, customer tier and sentiment, and the recommended team",
            "- Urgent escalations are answered within 4 business hours; normal ones within 1 business day",
            "- Tell the user their ticket was escalated, who will handle it and the expected response time",
        ],
        "I've passed your ticket to a specialist on our support team with a summary of everything so far, so you won't need to repeat yourself. You'll hear back within 1 business day.",
        "escalation, human agent, handoff, policy, sla, support",
    ),
]

ARTICLES = ORIGINAL + NEW


def main():
    OUT.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in ARTICLES) + "\n", encoding="utf-8")
    print(f"wrote {len(ARTICLES)} articles ({len(NEW)} new) to {OUT}")


if __name__ == "__main__":
    main()
