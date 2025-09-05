import os.path
import os
import base64  # For decoding email bodies
from bs4 import BeautifulSoup  # For stripping HTML to plain text

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
        try:
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        except ValueError as e:
            print(f"Invalid token file: {e} - Re-authorizing...")
            os.remove("token.json")  # Delete invalid token
    
    # If no valid credentials, prompt user login
    if not creds or not creds.valid or not creds.refresh_token:
        flow = InstalledAppFlow.from_client_secrets_file(
            "credentials.json", SCOPES
        )
        os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = 'true'
        # Ensure offline access and force consent for refresh_token
        creds = flow.run_local_server(port=8080, access_type='offline', prompt='consent')
        # Save new tokens
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    else:
        # Refresh if expired
        if creds.expired:
            creds.refresh(Request())
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
            msg_data = service.users().messages().get(userId="me", id=msg["id"], format="full").execute()
            headers = msg_data["payload"]["headers"]
            
            # Extract subject
            subject = next((header["value"] for header in headers if header["name"] == "Subject"), "No Subject")
            
            # Extract sender (From)
            from_addr = next((header["value"] for header in headers if header["name"] == "From"), "default@sender.com")
            
            # Extract full body (prefer plain text; if HTML, strip tags)
            body = ""
            if 'parts' in msg_data['payload']:
                for part in msg_data['payload']['parts']:
                    if part['mimeType'] == 'text/plain':
                        data = part['body'].get('data', '')
                        if data:
                            body += base64.urlsafe_b64decode(data).decode('utf-8')
                        break  # Use plain text if available
                    elif part['mimeType'] == 'text/html' and not body:  # Fallback to HTML if no plain text
                        data = part['body'].get('data', '')
                        if data:
                            html = base64.urlsafe_b64decode(data).decode('utf-8')
                            soup = BeautifulSoup(html, 'html.parser')
                            body += soup.get_text()  # Strip HTML to plain text
            else:
                data = msg_data['payload']['body'].get('data', '')
                if data:
                    body += base64.urlsafe_b64decode(data).decode('utf-8')
            
            emails.append({
                "thread_id": msg_data["threadId"],
                "subject": subject,
                "body": body,
                "from": from_addr  # Added sender email
            })
        
        return emails
    
    except HttpError as error:
        print(f"An error occurred: {error}")
        return []

if __name__ == "__main__":
    emails = fetch_unread_emails()
    for email in emails:
        print(f"Thread {email['thread_id']}: {email['subject']} - {email['body'][:50]}...")
