from ast import List
from googleapiclient.discovery import build
import google.oauth2.credentials
from datetime import date, timedelta, datetime
from typing import Optional, TypedDict

class GcalInterval(TypedDict):
    start: str # UTC ISO
    end: str # UTC ISO

class OauthCredentials(TypedDict):
    access_token: str
    refresh_token: str
    token_uri: str
    client_id: str
    client_secret: str
    scopes: Optional[List[str]]
    scope: Optional[str]


# should account for business hours
def busy_to_free(all_busy_times: List[GcalInterval], start_of_day: str, end_of_day: str) -> X:
    pass

def get_availability_for_day(availability: X, day: str) -> List[GcalInterval]:
    pass


# Unlike normal oauth credentials, Google's python library only accepts `scopes` not `scope`
def get_google_scopes(oauth_credentials: OauthCredentials) -> Optional[List[str]]:
    if oauth_credentials.scopes:
        return oauth_credentials.scopes
    if oauth_credentials.scope:
        return [oauth_credentials.scope]
    return None



# returns something like
# [{'start': '2024-04-11T16:00:00Z', 'end': '2024-04-11T16:30:00Z'}, {'start': '2024-04-12T16:00:00Z', 'end': '2024-04-12T16:30:00Z'}, {'start': '2024-04-12T17:00:00Z', 'end': '2024-04-13T02:30:00Z'}, {'start': '2024-04-15T19:30:00Z', 'end': '2024-04-15T20:00:00Z'}]
def get_gcal_busy_times(oauth_credentials: OauthCredentials) -> List[dict]:
    google_credentials = google.oauth2.credentials.Credentials(**{
        "scopes": get_google_scopes(oauth_credentials),
        "token": oauth_credentials.access_token,
        "refresh_token": oauth_credentials.refresh_token
    })

    with build('calendar', 'v3', credentials=google_credentials) as gcal:
        now = datetime.datetime.now(datetime.timezone.utc).astimezone()
        two_weeks = datetime.timedelta(weeks=2)
        res = gcal.freebusy().query(body={
            "items": [
                {
                    "id": "primary",
                },
            ],
            "timeMax": (now + two_weeks).isoformat(),
            "timeMin": now.isoformat(),
        }).execute()
        cal_holds = res['calendars']['primary']['busy']
        return cal_holds
    