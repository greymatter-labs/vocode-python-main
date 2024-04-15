
from googleapiclient.discovery import build
import datetime

credentials = {
    "scope": "https://www.googleapis.com/auth/calendar",
    "token_type": "Bearer",
    "expiry_date": 1712737888818,
    "access_token": "ya29.a0Ad52N39ovPP7iZG3n7lA8U4SH-7bj99yjFCGgCTRDDa9k3OdWr1S4t1eHQ4xoHzNVIS-HuXreTZRtTW_y9cZIvIswR8PBQB4aubKuBqDEyVGb_HgVKn8dAcB0NjK7cIXXDDmDZJKx0w41BURl0xP2v1wmn4xNIelvYTPaCgYKATMSARASFQHGX2Min71-UNOQ-srt6ME0Fl6Ekw0171",
    "refresh_token": "1//05rveMvzFqsHdCgYIARAAGAUSNwF-L9IrtR7mK0cLsrjwQCjYdG3vW7bAs-jTEGoFIiI29IwfmtSVxdKyvXlC5nDk6GSL3eHFzpo"
}

with build('calendar', 'v3', credentials=credentials) as gcal:
    now = datetime.datetime.now(datetime.timezone.utc).astimezone()
    two_weeks = datetime.timedelta(weeks=2)
    res =gcal.freebusy().query(body={
        "items": [
            {
                "id": "primary",
            },
        ],
        "timeMax": (now + two_weeks).isoformat(),
        "timeMin": now.isoformat(),
    })
    print(res)