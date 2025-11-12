import os
import requests

#gets the secret key from the .env file
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = os.getenv("SENDGRID_FROM_EMAIL")

def mail_sender(to, subject, html):
  url = "https://api.sendgrid.com/v3/mail/send"
  response = requests.post(url,headers={"Authorization": f"Bearer {SENDGRID_API_KEY}","Content-Type": "application/json"},
        json={"personalizations": [{"to": [{"email": to}]}],"from": {"email": SENDGRID_FROM_EMAIL},"to":{"email": SENDGRID_FROM_EMAIL},"subject": subject,"content": [{"type": "text/html", "value": html}]} )
  print("SendGrid Response:", response.status_code, response.text)
  return response.status_code
