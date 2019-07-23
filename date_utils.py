
from datetime import date, datetime, timedelta, tzinfo
from time import mktime, strptime


#Taken from https://stackoverflow.com/questions/1101508/how-to-parse-dates-with-0400-timezone-string-in-python/23122493#23122493 
class FixedOffset(tzinfo):
    """Fixed offset in minutes: `time = utc_time + utc_offset`."""
    def __init__(self, offset):
        self.__offset = timedelta(minutes=offset)
        hours, minutes = divmod(offset, 60)
        #NOTE: the last part is to remind about deprecated POSIX GMT+h timezones
        #  that have the opposite sign in the name;
        #  the corresponding numeric value is not used e.g., no minutes
        self.__name = '<%+03d%02d>%+d' % (hours, minutes, -hours)
    def utcoffset(self, dt=None):
        return self.__offset
    def tzname(self, dt=None):
        return self.__name
    def dst(self, dt=None):
        return timedelta(0)
    def __repr__(self):
        return 'FixedOffset(%d)' % (self.utcoffset().total_seconds() / 60) 


#Taken from https://stackoverflow.com/questions/2331592/why-does-datetime-datetime-utcnow-not-contain-timezone-information
# A UTC class.
class UTC(tzinfo):
    """UTC"""
    def utcoffset(self, dt):
        return timedelta(0)
    def tzname(self, dt):
        return "UTC"
    def dst(self, dt):
        return timedelta(0)
    def __repr__(self):
        return 'UTC'


def convert_commit_date(date_as_string):
    native_date_str, _, offset_str = date_as_string.rpartition(' ')
    native_date = datetime.strptime(native_date_str, '%a %b %d %H:%M:%S %Y')
    offset = int(offset_str[-4:-2])*60 + int(offset_str[-2:])
    if offset_str[0] == "-":
        offset = -offset
    date = native_date.replace(tzinfo=FixedOffset(offset))
    return date


def convert_bug_report_timestamp(timestamp_as_string):
    date = datetime.utcfromtimestamp(float(timestamp_as_string))
    date = date.replace(tzinfo=UTC())
    return date


# see https://stackoverflow.com/questions/8777753/converting-datetime-date-to-utc-timestamp-in-python
# section "How to convert an aware datetime object to POSIX timestamp"
def datetime_to_timestamp(dt, epoch=datetime(1970, 1, 1, tzinfo=UTC())):
    """Convert offset-aware datetime to timestamp

    Parameters
    ----------
    dt : datetime
        Offset-aware datetime, i.e. datetime with non-None tzinfo.

    epoch : datetime, optional
        Offset-aware epoch, by default Unix epoch (used as static variable).

    Returns
    -------
    float
        POSIX timestamp corresponding to the datetime instance, that is
        seconds since 1970-01-01 00:00:00 UTC
    """
    #assert dt.tzinfo is not None and dt.utcoffset() is not None
    try:
        # Python 3.3+
        dt.timestamp()
    except:
        # Python 2.x
        return (dt - epoch).total_seconds()
