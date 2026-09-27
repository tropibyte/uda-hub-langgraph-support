# UDA-Hub evaluation report

Labelled tickets: 32 (evaluation/tickets.jsonl). Each routing strategy ran on its own fresh copy of the databases with the live LLM and MCP tools.

| metric | rules_first | llm_supervisor |
|---|---|---|
| tickets | 32 | 32 |
| category_accuracy | 0.969 | 1.0 |
| route_accuracy | 1.0 | 1.0 |
| outcome_accuracy | 1.0 | 1.0 |
| team_accuracy | 1.0 | 1.0 |
| escalation_precision | 1.0 | 1.0 |
| escalation_recall | 1.0 | 1.0 |
| article_hit_rate | 1.0 | 1.0 |
| article_cited_rate | 1.0 | 1.0 |
| tool_accuracy | 1.0 | 1.0 |
| grounded_resolution_rate | 1.0 | 1.0 |
| avg_confidence | 0.969 | 0.969 |
| avg_judge_score | 1.0 | 1.0 |
| avg_latency_s | 12.37 | 13.24 |
| avg_llm_calls | 3.56 | 4.28 |
| all_checks_passed | 31 | 32 |

## Per ticket

| id | variant | category | route | status | conf | checks | tools |
|---|---|---|---|---|---|---|---|
| T01 | rules_first | reservation_booking | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T02 | rules_first | subscription_management | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T03 | rules_first | technical_issue | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T04 | rules_first | technical_issue | knowledge_resolver | resolved | 0.937 | ✓✓✓✓✓✓ |  |
| T05 | rules_first | subscription_management | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T06 | rules_first | reservation_change | account_specialist | resolved | 0.9 | ✓✓✓✓✓✓ | list_reservations |
| T07 | rules_first | general_inquiry | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T08 | rules_first | technical_issue | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T09 | rules_first | general_inquiry | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T10 | rules_first | general_inquiry | knowledge_resolver | resolved | 0.918 | ✓✓✓✓✓✓ |  |
| T11 | rules_first | subscription_management | account_specialist | resolved | 0.985 | ✓✓✓✓✓✓ | get_customer_profile |
| T12 | rules_first | reservation_change | account_specialist | resolved | 0.961 | ✓✓✓✓✓✓ | cancel_reservation, list_reservations |
| T13 | rules_first | reservation_booking | account_specialist | resolved | 0.9 | ✓✓✓✓✓✓ | reserve_experience, search_experiences |
| T14 | rules_first | reservation_booking | account_specialist | resolved | 0.9 | ✓✓✓✓✓✓ | search_experiences |
| T15 | rules_first | subscription_management | billing_specialist | resolved | 1.0 | ✓✓✓✓✓✓ | resume_subscription |
| T16 | rules_first | subscription_management | billing_specialist | resolved | 1.0 | ✓✓✓✓✓✓ | pause_subscription |
| T17 | rules_first | billing_payment | billing_specialist | escalated | None | ✓✓✓✓✓✓ | get_customer_profile, submit_refund_request |
| T18 | rules_first | billing_payment | billing_specialist | resolved | 1.0 | ✓✓✓✓✓✓ | get_customer_profile |
| T19 | rules_first | login_access | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T20 | rules_first | account_security | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T21 | rules_first | account_security | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T22 | rules_first | safety_incident | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T23 | rules_first | privacy_request | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T24 | rules_first | account_security | escalation | escalated | None | ✗✓✓✓✓✓ |  |
| T25 | rules_first | refund_request | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T26 | rules_first | other | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T27 | rules_first | other | knowledge_resolver | resolved | 0.938 | ✓✓✓✓✓✓ |  |
| T28 | rules_first | login_access | knowledge_resolver | resolved | 0.921 | ✓✓✓✓✓✓ | search_knowledge_base |
| T29 | rules_first | reservation_booking | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T30 | rules_first | reservation_change | account_specialist | resolved | 0.927 | ✓✓✓✓✓✓ | cancel_reservation, list_reservations |
| T31 | rules_first | subscription_management | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T32 | rules_first | reservation_change | account_specialist | resolved | 1.0 | ✓✓✓✓✓✓ | list_reservations, search_knowledge_base |
| T01 | llm_supervisor | reservation_booking | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T02 | llm_supervisor | subscription_management | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T03 | llm_supervisor | technical_issue | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T04 | llm_supervisor | technical_issue | knowledge_resolver | resolved | 0.937 | ✓✓✓✓✓✓ |  |
| T05 | llm_supervisor | subscription_management | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T06 | llm_supervisor | reservation_change | knowledge_resolver | resolved | 0.891 | ✓✓✓✓✓✓ |  |
| T07 | llm_supervisor | general_inquiry | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T08 | llm_supervisor | technical_issue | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T09 | llm_supervisor | general_inquiry | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T10 | llm_supervisor | general_inquiry | knowledge_resolver | resolved | 0.902 | ✓✓✓✓✓✓ |  |
| T11 | llm_supervisor | subscription_management | account_specialist | resolved | 0.986 | ✓✓✓✓✓✓ | get_customer_profile |
| T12 | llm_supervisor | reservation_change | account_specialist | resolved | 0.961 | ✓✓✓✓✓✓ | cancel_reservation, list_reservations |
| T13 | llm_supervisor | reservation_booking | account_specialist | resolved | 0.9 | ✓✓✓✓✓✓ | reserve_experience, search_experiences |
| T14 | llm_supervisor | reservation_booking | account_specialist | resolved | 0.9 | ✓✓✓✓✓✓ | search_experiences |
| T15 | llm_supervisor | subscription_management | billing_specialist | resolved | 1.0 | ✓✓✓✓✓✓ | resume_subscription |
| T16 | llm_supervisor | subscription_management | billing_specialist | resolved | 1.0 | ✓✓✓✓✓✓ | pause_subscription |
| T17 | llm_supervisor | refund_request | billing_specialist | escalated | None | ✓✓✓✓✓✓ | submit_refund_request |
| T18 | llm_supervisor | billing_payment | billing_specialist | resolved | 1.0 | ✓✓✓✓✓✓ | get_customer_profile |
| T19 | llm_supervisor | login_access | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T20 | llm_supervisor | account_security | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T21 | llm_supervisor | account_security | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T22 | llm_supervisor | safety_incident | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T23 | llm_supervisor | privacy_request | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T24 | llm_supervisor | other | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T25 | llm_supervisor | refund_request | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T26 | llm_supervisor | other | escalation | escalated | None | ✓✓✓✓✓✓ |  |
| T27 | llm_supervisor | other | knowledge_resolver | resolved | 0.938 | ✓✓✓✓✓✓ | search_knowledge_base |
| T28 | llm_supervisor | login_access | knowledge_resolver | resolved | 0.95 | ✓✓✓✓✓✓ |  |
| T29 | llm_supervisor | reservation_booking | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T30 | llm_supervisor | reservation_change | account_specialist | resolved | 0.927 | ✓✓✓✓✓✓ | cancel_reservation, list_reservations |
| T31 | llm_supervisor | subscription_management | knowledge_resolver | resolved | 1.0 | ✓✓✓✓✓✓ |  |
| T32 | llm_supervisor | reservation_change | account_specialist | resolved | 1.0 | ✓✓✓✓✓✓ | list_reservations, search_knowledge_base |

checks = category, first route, outcome, tools, escalation team, expected article (✓ = pass or n/a).

## Misses

- **T24 (rules_first)** `This is ridiculous, it's the third time I'm writing. I want to speak t` -> category=account_security, route=escalation, status=escalated, tools=[]. Reply: Hi Cathy, I'm very sorry for the repeated issues with your account security. I've escalated your case to our Trust & Safety team, who will review it urgently. You can expect to hear back within 4 busi
