import os.path
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Scopes: Read-only for now (expand later for drafts/send)
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",  # For drafting
    "https://www.googleapis.com/auth/gmail.send"      # For sending
]

def fetch_unread_emails(max_results=5):
    """Fetches and returns a list of the last N unread emails as dicts."""
    creds = None
    # Load existing tokens if available
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    
    # If no valid credentials, prompt user login
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = 'true'

            creds = flow.run_local_server(port=8080)
        # Save tokens for next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    
    try:
        # Build Gmail API client
        service = build("gmail", "v1", credentials=creds)
        
        # Fetch unread messages (last N)
        results = service.users().messages().list(userId="me", q="is:unread", maxResults=max_results).execute()
        messages = results.get("messages", [])
        
        if not messages:
            print("No unread messages found.")
            return []
        
        emails = []
        for msg in messages:
            msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
            subject = next((header["value"] for header in msg_data["payload"]["headers"] if header["name"] == "Subject"), "No Subject")
            body = msg_data["snippet"]  # Use snippet for simple body; expand to full payload if needed
            emails.append({
                "thread_id": msg_data["threadId"],
                "subject": subject,
                "body": body
            })
        
        return emails
    
    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

if __name__ == "__main__":
    emails = fetch_unread_emails()
    for email in emails:
        print(f"Thread {email['thread_id']}: {email['subject']} - {email['body'][:50]}...")
