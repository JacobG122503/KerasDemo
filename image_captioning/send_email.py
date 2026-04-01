
import smtplib
from email.mime.text import MIMEText
import sys

def send_email(subject, body, to_email):
    """Sends an email using a local sendmail-compatible MTA."""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'cluster-notifier@example.com'
    msg['To'] = to_email

    try:
        # Use the sendmail command
        with smtplib.SMTP('localhost') as s:
            s.send_message(msg)
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Error sending email: {e}", file=sys.stderr)
        # Fallback for systems without a local MTA, just print to console
        print("--- EMAIL ---")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Body: {body}")
        print("--- END EMAIL ---")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python send_email.py <to_email> <subject> '<body>'", file=sys.stderr)
        sys.exit(1)
    
    to_email = sys.argv[1]
    subject = sys.argv[2]
    body = sys.argv[3]
    send_email(to_email, subject, body)
