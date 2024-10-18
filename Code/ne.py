import smtplib

# Define email addresses
sender_email = "codeinechoniner@gmail.com"
receiver_email = "jeromgladsun@gmail.com"

# Create SMTP session
smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
smtp_server.starttls()
smtp_server.login(sender_email, "yhqjkahoevcixcth")

# Send email
message = "This is a test email sent from Python script."
smtp_server.sendmail(sender_email, receiver_email, message)
smtp_server.quit()
