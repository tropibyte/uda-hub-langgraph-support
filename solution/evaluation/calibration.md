# Retrieval calibration

44 labelled queries (`evaluation/retrieval_queries.jsonl`): in-domain questions with the article that should answer them, plus off-topic questions that must not match anything.
Dense anchors: cosine 0.22 -> confidence 0, cosine 0.58 -> confidence 1.

| metric | hybrid + category (production) | hybrid | bm25_only |
|---|---|---|---|
| method | hybrid(dense+bm25, weighted RRF) + category boost | hybrid(dense+bm25, weighted RRF) | bm25 |
| in_domain | 37 | 37 | 37 |
| off_topic | 7 | 7 | 7 |
| hit@1 | 36/37 | 33/37 | 24/37 |
| hit@3 | 37/37 | 37/37 | 31/37 |
| min_in_domain_confidence | 0.201 | 0.201 | 0.33 |
| max_off_topic_confidence | 0.332 | 0.225 | 0.418 |
| in_domain_above_gate(0.35) | 36/37 | 36/37 | 35/37 |
| off_topic_below_gate(0.35) | 7/7 | 7/7 | 6/7 |

## hybrid + category (production)

| query | expected | top-1 | hit@3 | confidence | cosine |
|---|---|---|---|---|---|
| I can't log in, the password reset email never arrives | How to Handle Login Issues? | How to Handle Login Issues? | ✓ | 0.918 | 0.4784 |
| my card was declined when renewing | Payment Failed or Card Declined | Payment Failed or Card Declined | ✓ | 1.0 | 0.6022 |
| I was charged twice this month | Double Charge or Unrecognized Charge | Double Charge or Unrecognized Charge | ✓ | 1.0 | 0.6288 |
| can I get my money back for last month? | Refund Policy and Refund Requests | Refund Policy and Refund Requests | ✓ | 0.759 | 0.4214 |
| how do I change the card I pay with | Updating Your Payment Method or Billing Date | Updating Your Payment Method or Billing Date | ✓ | 1.0 | 0.5511 |
| I got a gift card code, where do I put it | Gift Cards and Corporate Plans | Gift Cards and Corporate Plans | ✓ | 0.928 | 0.4821 |
| what's the difference between basic and premium | CultPass Plans: Basic vs Premium | CultPass Plans: Basic vs Premium | ✓ | 1.0 | 0.5384 |
| I want to upgrade my plan | Upgrading or Downgrading Your Plan | Upgrading or Downgrading Your Plan | ✓ | 1.0 | 0.5668 |
| do unused experiences roll over to next month? | How the Monthly Experience Quota Works | How the Monthly Experience Quota Works | ✓ | 0.998 | 0.5074 |
| how do I restart my paused membership | Reactivating a Paused or Cancelled Subscription | Reactivating a Paused or Cancelled Subscription | ✓ | 1.0 | 0.5978 |
| I need to cancel my booking for Saturday | Cancelling a Reservation and the No-Show Policy | Cancelling a Reservation and the No-Show Policy | ✓ | 0.75 | 0.4468 |
| the event I want is full | Event Sold Out and the Waitlist | Event Cancelled or Rescheduled by the Partner | ✓ | 0.725 | 0.3565 |
| why is there an extra fee on this event | Premium Experiences and Extra Fees | Premium Experiences and Extra Fees | ✓ | 0.867 | 0.4601 |
| can my friend go in my place? | Transferring or Sharing a Reservation | Transferring or Sharing a Reservation | ✓ | 0.685 | 0.3948 |
| the concert was cancelled by the venue, what happens now | Event Cancelled or Rescheduled by the Partner | Event Cancelled or Rescheduled by the Partner | ✓ | 0.904 | 0.4734 |
| the entrance scanner won't read my ticket | QR Code Not Scanning at the Venue | QR Code Not Scanning at the Venue | ✓ | 1.0 | 0.5972 |
| the app keeps crashing on my android | App Crashes, Freezes or Won't Load | App Crashes, Freezes or Won't Load | ✓ | 0.889 | 0.4681 |
| I'm not getting any notifications | Not Receiving Emails or Notifications | Not Receiving Emails or Notifications | ✓ | 0.949 | 0.4896 |
| how do I update my email address | Changing Your Email Address or Profile Details | Changing Your Email Address or Profile Details | ✓ | 1.0 | 0.5967 |
| my account says it is suspended | Account Blocked or Suspended | Account Blocked or Suspended | ✓ | 1.0 | 0.5267 |
| please delete all my data | Deleting Your Account and Personal Data Requests | Deleting Your Account and Personal Data Requests | ✓ | 0.837 | 0.4494 |
| someone else is using my account | Suspicious Activity and Account Security | Suspicious Activity and Account Security | ✓ | 0.833 | 0.4768 |
| is the museum wheelchair accessible? | Accessibility and Special Assistance at Experiences | Accessibility and Special Assistance at Experiences | ✓ | 0.69 | 0.3964 |
| is CultPass available in Curitiba? | Where CultPass Is Available and How Experiences Are Chosen | Where CultPass Is Available and How Experiences Are Chosen | ✓ | 1.0 | 0.6962 |
| a staff member harassed me at the show | Reporting a Problem or Safety Incident at a Venue | Reporting a Problem or Safety Incident at a Venue | ✓ | 0.582 | 0.3577 |
| I want to talk to a real person | When and How Support Escalates to a Human Agent | When and How Support Escalates to a Human Agent | ✓ | 0.201 | 0.2834 |
| how do I book an event | How to Reserve a Spot for an Event | How to Reserve a Spot for an Event | ✓ | 1.0 | 0.5354 |
| how many experiences do I get per month | What's Included in a CultPass Subscription | What's Included in a CultPass Subscription | ✓ | 0.765 | 0.4234 |
| how can I cancel my subscription | How to Cancel or Pause a Subscription | How to Cancel or Pause a Subscription | ✓ | 1.0 | 0.5845 |
| how much does premium cost compared to basic | CultPass Plans: Basic vs Premium | CultPass Plans: Basic vs Premium | ✓ | 0.731 | 0.4111 |
| I never got the confirmation email for my booking | Not Receiving Emails or Notifications | Not Receiving Emails or Notifications | ✓ | 0.81 | 0.4395 |
| the venue moved my concert to another date and I can't go | Event Cancelled or Rescheduled by the Partner | Event Cancelled or Rescheduled by the Partner | ✓ | 0.543 | 0.3724 |
| can I still get my credit back if I cancel tomorrow's show? | Cancelling a Reservation and the No-Show Policy | Cancelling a Reservation and the No-Show Policy | ✓ | 0.877 | 0.4637 |
| there are bookings on my account that I never made | Suspicious Activity and Account Security | Suspicious Activity and Account Security | ✓ | 0.812 | 0.4693 |
| I want my money back for this month | Refund Policy and Refund Requests | Refund Policy and Refund Requests | ✓ | 0.712 | 0.4044 |
| I can't sign in anymore | How to Handle Login Issues? | How to Handle Login Issues? | ✓ | 0.643 | 0.3794 |
| I can't log in to my Cultpass account. | How to Handle Login Issues? | How to Handle Login Issues? | ✓ | 0.863 | 0.3669 |
| Can you recommend a good pizza recipe? | (off-topic) | Transferring or Sharing a Reservation | - | 0.0 | 0.1081 |
| What's the weather forecast in Rio tomorrow? | (off-topic) | Where CultPass Is Available and How Experiences Are Chosen | - | 0.061 | 0.2419 |
| Do you sell airline tickets to Lisbon? | (off-topic) | Where CultPass Is Available and How Experiences Are Chosen | - | 0.332 | 0.2965 |
| Can you help me with my tax return? | (off-topic) | Refund Policy and Refund Requests | - | 0.08 | 0.2007 |
| Who won the football match last night? | (off-topic) | QR Code Not Scanning at the Venue | - | 0.08 | 0.1237 |
| Is my car insurance still valid? | (off-topic) | Payment Failed or Card Declined | - | 0.225 | 0.2721 |
| Write me a poem about the ocean | (off-topic) | Double Charge or Unrecognized Charge | - | 0.08 | 0.022 |

## hybrid

| query | expected | top-1 | hit@3 | confidence | cosine |
|---|---|---|---|---|---|
| I can't log in, the password reset email never arrives | How to Handle Login Issues? | How to Handle Login Issues? | ✓ | 0.798 | 0.4785 |
| my card was declined when renewing | Payment Failed or Card Declined | Payment Failed or Card Declined | ✓ | 1.0 | 0.6022 |
| I was charged twice this month | Double Charge or Unrecognized Charge | Double Charge or Unrecognized Charge | ✓ | 1.0 | 0.6288 |
| can I get my money back for last month? | Refund Policy and Refund Requests | Refund Policy and Refund Requests | ✓ | 0.639 | 0.4214 |
| how do I change the card I pay with | Updating Your Payment Method or Billing Date | Updating Your Payment Method or Billing Date | ✓ | 1.0 | 0.5511 |
| I got a gift card code, where do I put it | Gift Cards and Corporate Plans | Gift Cards and Corporate Plans | ✓ | 0.808 | 0.4821 |
| what's the difference between basic and premium | CultPass Plans: Basic vs Premium | CultPass Plans: Basic vs Premium | ✓ | 0.964 | 0.5384 |
| I want to upgrade my plan | Upgrading or Downgrading Your Plan | Upgrading or Downgrading Your Plan | ✓ | 1.0 | 0.5668 |
| do unused experiences roll over to next month? | How the Monthly Experience Quota Works | How the Monthly Experience Quota Works | ✓ | 0.878 | 0.5074 |
| how do I restart my paused membership | Reactivating a Paused or Cancelled Subscription | Reactivating a Paused or Cancelled Subscription | ✓ | 1.0 | 0.5978 |
| I need to cancel my booking for Saturday | Cancelling a Reservation and the No-Show Policy | Cancelling a Reservation and the No-Show Policy | ✓ | 0.63 | 0.4468 |
| the event I want is full | Event Sold Out and the Waitlist | Event Cancelled or Rescheduled by the Partner | ✓ | 0.605 | 0.3565 |
| why is there an extra fee on this event | Premium Experiences and Extra Fees | Premium Experiences and Extra Fees | ✓ | 0.747 | 0.4601 |
| can my friend go in my place? | Transferring or Sharing a Reservation | Transferring or Sharing a Reservation | ✓ | 0.565 | 0.3948 |
| the concert was cancelled by the venue, what happens now | Event Cancelled or Rescheduled by the Partner | Event Cancelled or Rescheduled by the Partner | ✓ | 0.784 | 0.4734 |
| the entrance scanner won't read my ticket | QR Code Not Scanning at the Venue | QR Code Not Scanning at the Venue | ✓ | 1.0 | 0.5973 |
| the app keeps crashing on my android | App Crashes, Freezes or Won't Load | App Crashes, Freezes or Won't Load | ✓ | 0.769 | 0.4681 |
| I'm not getting any notifications | Not Receiving Emails or Notifications | Not Receiving Emails or Notifications | ✓ | 0.829 | 0.4896 |
| how do I update my email address | Changing Your Email Address or Profile Details | Changing Your Email Address or Profile Details | ✓ | 1.0 | 0.5967 |
| my account says it is suspended | Account Blocked or Suspended | Account Blocked or Suspended | ✓ | 0.932 | 0.5267 |
| please delete all my data | Deleting Your Account and Personal Data Requests | Deleting Your Account and Personal Data Requests | ✓ | 0.717 | 0.4494 |
| someone else is using my account | Suspicious Activity and Account Security | Suspicious Activity and Account Security | ✓ | 0.713 | 0.4768 |
| is the museum wheelchair accessible? | Accessibility and Special Assistance at Experiences | Accessibility and Special Assistance at Experiences | ✓ | 0.57 | 0.3964 |
| is CultPass available in Curitiba? | Where CultPass Is Available and How Experiences Are Chosen | Where CultPass Is Available and How Experiences Are Chosen | ✓ | 1.0 | 0.6962 |
| a staff member harassed me at the show | Reporting a Problem or Safety Incident at a Venue | Reporting a Problem or Safety Incident at a Venue | ✓ | 0.462 | 0.3577 |
| I want to talk to a real person | When and How Support Escalates to a Human Agent | When and How Support Escalates to a Human Agent | ✓ | 0.201 | 0.2834 |
| how do I book an event | How to Reserve a Spot for an Event | How to Reserve a Spot for an Event | ✓ | 0.956 | 0.5354 |
| how many experiences do I get per month | What's Included in a CultPass Subscription | How the Monthly Experience Quota Works | ✓ | 0.844 | 0.5238 |
| how can I cancel my subscription | How to Cancel or Pause a Subscription | How to Cancel or Pause a Subscription | ✓ | 1.0 | 0.5845 |
| how much does premium cost compared to basic | CultPass Plans: Basic vs Premium | CultPass Plans: Basic vs Premium | ✓ | 0.611 | 0.4111 |
| I never got the confirmation email for my booking | Not Receiving Emails or Notifications | Not Receiving Emails or Notifications | ✓ | 0.69 | 0.4395 |
| the venue moved my concert to another date and I can't go | Event Cancelled or Rescheduled by the Partner | Event Cancelled or Rescheduled by the Partner | ✓ | 0.423 | 0.3724 |
| can I still get my credit back if I cancel tomorrow's show? | Cancelling a Reservation and the No-Show Policy | Cancelling a Reservation and the No-Show Policy | ✓ | 0.757 | 0.4637 |
| there are bookings on my account that I never made | Suspicious Activity and Account Security | Double Charge or Unrecognized Charge | ✓ | 0.692 | 0.406 |
| I want my money back for this month | Refund Policy and Refund Requests | Refund Policy and Refund Requests | ✓ | 0.592 | 0.4044 |
| I can't sign in anymore | How to Handle Login Issues? | How to Handle Login Issues? | ✓ | 0.523 | 0.3794 |
| I can't log in to my Cultpass account. | How to Handle Login Issues? | Not Receiving Emails or Notifications | ✓ | 0.837 | 0.5094 |
| Can you recommend a good pizza recipe? | (off-topic) | Transferring or Sharing a Reservation | - | 0.0 | 0.1081 |
| What's the weather forecast in Rio tomorrow? | (off-topic) | Where CultPass Is Available and How Experiences Are Chosen | - | 0.061 | 0.242 |
| Do you sell airline tickets to Lisbon? | (off-topic) | Transferring or Sharing a Reservation | - | 0.212 | 0.2338 |
| Can you help me with my tax return? | (off-topic) | Refund Policy and Refund Requests | - | 0.08 | 0.2007 |
| Who won the football match last night? | (off-topic) | QR Code Not Scanning at the Venue | - | 0.08 | 0.1237 |
| Is my car insurance still valid? | (off-topic) | Payment Failed or Card Declined | - | 0.225 | 0.2721 |
| Write me a poem about the ocean | (off-topic) | Double Charge or Unrecognized Charge | - | 0.08 | 0.022 |

## bm25_only

| query | expected | top-1 | hit@3 | confidence | cosine |
|---|---|---|---|---|---|
| I can't log in, the password reset email never arrives | How to Handle Login Issues? | How to Handle Login Issues? | ✓ | 0.747 | None |
| my card was declined when renewing | Payment Failed or Card Declined | Payment Failed or Card Declined | ✓ | 0.63 | None |
| I was charged twice this month | Double Charge or Unrecognized Charge | Double Charge or Unrecognized Charge | ✓ | 0.639 | None |
| can I get my money back for last month? | Refund Policy and Refund Requests | Double Charge or Unrecognized Charge | ✓ | 0.427 | None |
| how do I change the card I pay with | Updating Your Payment Method or Billing Date | Gift Cards and Corporate Plans | ✓ | 0.434 | None |
| I got a gift card code, where do I put it | Gift Cards and Corporate Plans | Gift Cards and Corporate Plans | ✓ | 0.71 | None |
| what's the difference between basic and premium | CultPass Plans: Basic vs Premium | Upgrading or Downgrading Your Plan | ✓ | 0.571 | None |
| I want to upgrade my plan | Upgrading or Downgrading Your Plan | Upgrading or Downgrading Your Plan | ✓ | 0.59 | None |
| do unused experiences roll over to next month? | How the Monthly Experience Quota Works | How the Monthly Experience Quota Works | ✓ | 0.686 | None |
| how do I restart my paused membership | Reactivating a Paused or Cancelled Subscription | Reactivating a Paused or Cancelled Subscription | ✓ | 0.63 | None |
| I need to cancel my booking for Saturday | Cancelling a Reservation and the No-Show Policy | Accessibility and Special Assistance at Experiences | ✗ | 0.498 | None |
| the event I want is full | Event Sold Out and the Waitlist | Transferring or Sharing a Reservation | ✗ | 0.33 | None |
| why is there an extra fee on this event | Premium Experiences and Extra Fees | Premium Experiences and Extra Fees | ✓ | 0.592 | None |
| can my friend go in my place? | Transferring or Sharing a Reservation | Transferring or Sharing a Reservation | ✓ | 0.577 | None |
| the concert was cancelled by the venue, what happens now | Event Cancelled or Rescheduled by the Partner | Event Cancelled or Rescheduled by the Partner | ✓ | 0.455 | None |
| the entrance scanner won't read my ticket | QR Code Not Scanning at the Venue | QR Code Not Scanning at the Venue | ✓ | 0.569 | None |
| the app keeps crashing on my android | App Crashes, Freezes or Won't Load | App Crashes, Freezes or Won't Load | ✓ | 0.565 | None |
| I'm not getting any notifications | Not Receiving Emails or Notifications | Not Receiving Emails or Notifications | ✓ | 0.474 | None |
| how do I update my email address | Changing Your Email Address or Profile Details | Changing Your Email Address or Profile Details | ✓ | 0.643 | None |
| my account says it is suspended | Account Blocked or Suspended | Account Blocked or Suspended | ✓ | 0.616 | None |
| please delete all my data | Deleting Your Account and Personal Data Requests | Deleting Your Account and Personal Data Requests | ✓ | 0.626 | None |
| someone else is using my account | Suspicious Activity and Account Security | How to Handle Login Issues? | ✗ | 0.458 | None |
| is the museum wheelchair accessible? | Accessibility and Special Assistance at Experiences | Accessibility and Special Assistance at Experiences | ✓ | 0.412 | None |
| is CultPass available in Curitiba? | Where CultPass Is Available and How Experiences Are Chosen | Where CultPass Is Available and How Experiences Are Chosen | ✓ | 0.486 | None |
| a staff member harassed me at the show | Reporting a Problem or Safety Incident at a Venue | QR Code Not Scanning at the Venue | ✓ | 0.505 | None |
| I want to talk to a real person | When and How Support Escalates to a Human Agent | Transferring or Sharing a Reservation | ✗ | 0.33 | None |
| how do I book an event | How to Reserve a Spot for an Event | Transferring or Sharing a Reservation | ✓ | 0.414 | None |
| how many experiences do I get per month | What's Included in a CultPass Subscription | What's Included in a CultPass Subscription | ✓ | 0.534 | None |
| how can I cancel my subscription | How to Cancel or Pause a Subscription | How to Cancel or Pause a Subscription | ✓ | 0.405 | None |
| how much does premium cost compared to basic | CultPass Plans: Basic vs Premium | CultPass Plans: Basic vs Premium | ✓ | 0.507 | None |
| I never got the confirmation email for my booking | Not Receiving Emails or Notifications | How to Reserve a Spot for an Event | ✓ | 0.544 | None |
| the venue moved my concert to another date and I can't go | Event Cancelled or Rescheduled by the Partner | Updating Your Payment Method or Billing Date | ✗ | 0.499 | None |
| can I still get my credit back if I cancel tomorrow's show? | Cancelling a Reservation and the No-Show Policy | Cancelling a Reservation and the No-Show Policy | ✓ | 0.646 | None |
| there are bookings on my account that I never made | Suspicious Activity and Account Security | Deleting Your Account and Personal Data Requests | ✗ | 0.367 | None |
| I want my money back for this month | Refund Policy and Refund Requests | Double Charge or Unrecognized Charge | ✓ | 0.427 | None |
| I can't sign in anymore | How to Handle Login Issues? | How to Handle Login Issues? | ✓ | 0.434 | None |
| I can't log in to my Cultpass account. | How to Handle Login Issues? | How to Handle Login Issues? | ✓ | 0.463 | None |
| Can you recommend a good pizza recipe? | (off-topic) | How to Reserve a Spot for an Event | - | 0.0 | None |
| What's the weather forecast in Rio tomorrow? | (off-topic) | How to Reserve a Spot for an Event | - | 0.0 | None |
| Do you sell airline tickets to Lisbon? | (off-topic) | QR Code Not Scanning at the Venue | - | 0.338 | None |
| Can you help me with my tax return? | (off-topic) | Where CultPass Is Available and How Experiences Are Chosen | - | 0.418 | None |
| Who won the football match last night? | (off-topic) | App Crashes, Freezes or Won't Load | - | 0.329 | None |
| Is my car insurance still valid? | (off-topic) | Payment Failed or Card Declined | - | 0.322 | None |
| Write me a poem about the ocean | (off-topic) | Double Charge or Unrecognized Charge | - | 0.307 | None |

Columns: *hybrid + category* is what the workflow runs (the classifier's category boosts articles tagged for it; each query carries the category a correct classification would give). *hybrid* is pure retrieval with no metadata. *bm25_only* is the offline fallback.

Reading: the category boost is what makes hard paraphrases such as "I can't log in to my Cultpass account" (the embedding model ranks the login article low for it) land on the right article. 'I want to talk to a real person' is escalated by the wants-human guardrail before the gate is consulted, so its low score is harmless. BM25-only ranks reasonably but cannot separate off-topic questions by score, which is why it is documented as a degraded mode.
