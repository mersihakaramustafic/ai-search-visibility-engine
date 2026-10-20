"""Test brands: the same rows and aliases as the brands table."""

from __future__ import annotations

from visibility_extractor.matching import Brand

BRANDS = [
    Brand("attio", "Attio", "attio.com", ("Attio", "Attio CRM"), False),
    Brand("close", "Close", "close.com", ("Close", "Close CRM", "Close.com", "Close.io"), False),
    Brand("freshsales", "Freshsales", "freshworks.com",
          ("Freshsales", "Freshsales CRM", "Freshworks CRM", "Freshworks Sales"), False),
    Brand("hubspot", "HubSpot", "hubspot.com",
          ("HubSpot", "Hubspot", "HubSpot CRM", "HubSpot Sales Hub", "HubSpot Smart CRM"), True),
    Brand("intercom", "Intercom", "intercom.com", ("Intercom", "Intercom Fin", "Fin AI Agent"), False),
    Brand("monday_crm", "monday CRM", "monday.com",
          ("monday CRM", "Monday CRM", "monday.com CRM", "monday sales CRM"), False),
    Brand("pipedrive", "Pipedrive", "pipedrive.com", ("Pipedrive", "Pipedrive CRM"), False),
    Brand("salesforce", "Salesforce", "salesforce.com",
          ("Salesforce", "Salesforce CRM", "Salesforce Sales Cloud", "Sales Cloud", "SFDC"), False),
    Brand("zendesk", "Zendesk", "zendesk.com", ("Zendesk", "Zendesk Suite", "Zendesk Sell"), False),
    Brand("zoho_crm", "Zoho CRM", "zoho.com", ("Zoho CRM", "Zoho", "Zoho CRM Plus"), False),
]
