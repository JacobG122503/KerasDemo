
import mimetypes
import os
import shutil
import smtplib
import subprocess
import sys
from email.message import EmailMessage


def send_email(subject, body, to_email, attachment_path=None):
    """Sends an email using a local sendmail-compatible MTA."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = "cluster-notifier@example.com"
    msg["To"] = to_email
    msg.set_content(body)

    if attachment_path and os.path.isfile(attachment_path):
        mime_type, _ = mimetypes.guess_type(attachment_path)
        if mime_type:
            maintype, subtype = mime_type.split("/", 1)
        else:
            maintype, subtype = "application", "octet-stream"

        with open(attachment_path, "rb") as attachment_file:
            msg.add_attachment(
                attachment_file.read(),
                maintype=maintype,
                subtype=subtype,
                filename=os.path.basename(attachment_path),
            )

    sendmail_path = shutil.which("sendmail") or "/usr/sbin/sendmail"

    try:
        if os.path.exists(sendmail_path):
            subprocess.run(
                [sendmail_path, "-t", "-oi"],
                input=msg.as_bytes(),
                check=True,
            )
        else:
            with smtplib.SMTP("localhost") as smtp_client:
                smtp_client.send_message(msg)
        print(f"Email sent to {to_email}")
    except Exception as exc:
        print(f"Error sending email: {exc}", file=sys.stderr)
        print("--- EMAIL ---")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Body: {body}")
        if attachment_path:
            print(f"Attachment: {attachment_path}")
        print("--- END EMAIL ---")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print(
            "Usage: python send_email.py <to_email> <subject> '<body>' [attachment_path]",
            file=sys.stderr,
        )
        sys.exit(1)

    to_email = sys.argv[1]
    subject = sys.argv[2]
    body = sys.argv[3]
    attachment_path = sys.argv[4] if len(sys.argv) == 5 else None
    send_email(subject, body, to_email, attachment_path)
