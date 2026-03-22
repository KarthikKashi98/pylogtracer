"""
time_resolver.py
=================
Resolves relative and ambiguous time references in user questions
into concrete datetime strings before passing to the LangGraph router.

Python handles time math — reliable, testable, zero LLM cost.
LLM handles intent — picks the right tool using resolved timestamps.

Patterns handled:
    Relative day:
        "today"         → today 00:00:00 to 23:59:59
        "yesterday"     → yesterday 00:00:00 to 23:59:59
        "last night"    → yesterday 20:00:00 to today 06:00:00

    Relative time (anchored to today unless day mentioned):
        "10am"          → today 10:00:00 to 10:59:59
        "10:30am"       → today 10:30:00 to 10:59:59
        "2pm"           → today 14:00:00 to 14:59:59
        "at 10"         → today 10:00:00 to 10:59:59

    Combined:
        "yesterday 10am"     → yesterday 10:00:00 to 10:59:59
        "yesterday at 2pm"   → yesterday 14:00:00 to 14:59:59
        "this morning"       → today 06:00:00 to 12:00:00
        "this afternoon"     → today 12:00:00 to 18:00:00
        "this evening"       → today 18:00:00 to 22:00:00

    Relative duration:
        "an hour ago"        → now-1h to now
        "2 hours ago"        → now-2h to now
        "30 minutes ago"     → now-30m to now
        "last 30 minutes"    → now-30m to now

    Absolute (passed through, no change):
        "2024-03-01"              → unchanged
        "2024-03-01 10:00:00"     → unchanged
        "March 1"                 → 2024-03-01 (current year)

Usage:
    from utils.time_resolver import TimeResolver
    resolver = TimeResolver()
    result   = resolver.resolve("what errors happened at 10am?")
    # result = {
    #     "original_question" : "what errors happened at 10am?",
    #     "enriched_question" : "what errors happened at 10am? [RESOLVED: ...]",
    #     "from_dt"           : "2024-03-01 10:00:00",
    #     "to_dt"             : "2024-03-01 10:59:59",
    #     "date"              : "2024-03-01",
    #     "resolved"          : True
    # }
"""

import re
from datetime import datetime, timedelta, date
from typing import Optional, Dict


# ── Month name map ────────────────────────────────────────────────
MONTHS = {
    "january": 1,  "jan": 1,
    "february": 2, "feb": 2,
    "march": 3,    "mar": 3,
    "april": 4,    "apr": 4,
    "may": 5,
    "june": 6,     "jun": 6,
    "july": 7,     "jul": 7,
    "august": 8,   "aug": 8,
    "september": 9,"sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11,"nov": 11,
    "december": 12,"dec": 12,
}


class TimeResolver:
    """
    Resolves relative time references in questions to absolute timestamps.

    Args:
        now: Override current time (useful for testing). Default = datetime.now()
    """

    def __init__(self, now: Optional[datetime] = None):
        self._now = now or datetime.now()

    # ─────────────────────────────────────────────────────────────
    # PUBLIC
    # ─────────────────────────────────────────────────────────────

    def resolve(self, question: str) -> Dict:
        """
        Resolve time references in a question.

        Returns:
            {
                "original_question": str,
                "enriched_question": str,   ← question + resolved timestamps appended
                "from_dt":          str | None,
                "to_dt":            str | None,
                "date":             str | None,
                "resolved":         bool        ← True if any time was resolved
            }
        """
        lower = question.lower()

        result = {
            "original_question": question,
            "enriched_question": question,
            "from_dt":           None,
            "to_dt":             None,
            "date":              None,
            "resolved":          False,
        }

        # Try each resolver in priority order
        # Higher priority resolvers run first
        resolved = (
            self._resolve_absolute_datetime(lower)   or   # "2024-03-01 10:00:00"
            self._resolve_absolute_date(lower)        or   # "2024-03-01", "March 1"
            self._resolve_duration_ago(lower)         or   # "2 hours ago"
            self._resolve_last_duration(lower)        or   # "last 30 minutes"
            self._resolve_time_of_day_named(lower)    or   # "this morning/afternoon/evening/night"
            self._resolve_last_night(lower)           or   # "last night"
            self._resolve_yesterday_with_time(lower)  or   # "yesterday at 10am"
            self._resolve_today_with_time(lower)      or   # "today at 10am"
            self._resolve_yesterday(lower)            or   # "yesterday" alone
            self._resolve_today(lower)                or   # "today" alone
            self._resolve_clock_time(lower)               # "10am", "2:30pm"
        )

        if resolved:
            result.update(resolved)
            result["resolved"] = True
            result["enriched_question"] = self._enrich(question, resolved)

        return result

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — resolvers (each returns dict or None)
    # ─────────────────────────────────────────────────────────────

    def _resolve_absolute_datetime(self, text: str) -> Optional[Dict]:
        """2024-03-01 10:00:00 or 2024-03-01T10:00:00"""
        pattern = r'(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)'
        m = re.search(pattern, text)
        if m:
            dt_str = f"{m.group(1)} {m.group(2)}"
            if len(m.group(2)) == 5:
                dt_str += ":00"
            return {
                "from_dt": dt_str,
                "to_dt":   dt_str,
                "date":    m.group(1),
            }
        return None

    def _resolve_absolute_date(self, text: str) -> Optional[Dict]:
        """2024-03-01 or March 1 or Mar 1"""
        # ISO format
        m = re.search(r'\b(\d{4}-\d{2}-\d{2})\b', text)
        if m:
            return {
                "from_dt": None,
                "to_dt":   None,
                "date":    m.group(1),
            }

        # "March 1" or "Mar 1" or "1 March"
        month_pattern = '|'.join(MONTHS.keys())
        m = re.search(
            rf'\b({month_pattern})\s+(\d{{1,2}})\b|\b(\d{{1,2}})\s+({month_pattern})\b',
            text
        )
        if m:
            if m.group(1):
                month_name, day = m.group(1), int(m.group(2))
            else:
                day, month_name = int(m.group(3)), m.group(4)
            month_num = MONTHS[month_name]
            year      = self._now.year
            try:
                dt  = date(year, month_num, day)
                return {
                    "from_dt": None,
                    "to_dt":   None,
                    "date":    dt.strftime("%Y-%m-%d"),
                }
            except ValueError:
                pass
        return None

    def _resolve_duration_ago(self, text: str) -> Optional[Dict]:
        """'2 hours ago', 'an hour ago', '30 minutes ago', '5 mins ago'"""
        m = re.search(
            r'\b(an?|\d+)\s+(hour|hr|minute|min|second|sec)s?\s+ago\b',
            text
        )
        if m:
            amount_str = m.group(1)
            unit       = m.group(2)
            amount     = 1 if amount_str in ("a", "an") else int(amount_str)

            if unit in ("hour", "hr"):
                delta = timedelta(hours=amount)
            elif unit in ("minute", "min"):
                delta = timedelta(minutes=amount)
            else:
                delta = timedelta(seconds=amount)

            from_dt = self._now - delta
            return {
                "from_dt": self._fmt(from_dt),
                "to_dt":   self._fmt(self._now),
                "date":    from_dt.strftime("%Y-%m-%d"),
            }
        return None

    def _resolve_last_duration(self, text: str) -> Optional[Dict]:
        """'last 30 minutes', 'last 2 hours', 'past hour'"""
        m = re.search(
            r'\b(?:last|past)\s+(\d+|an?)\s+(hour|hr|minute|min|second|sec)s?\b',
            text
        )
        if m:
            amount_str = m.group(1)
            unit       = m.group(2)
            amount     = 1 if amount_str in ("a", "an") else int(amount_str)

            if unit in ("hour", "hr"):
                delta = timedelta(hours=amount)
            elif unit in ("minute", "min"):
                delta = timedelta(minutes=amount)
            else:
                delta = timedelta(seconds=amount)

            from_dt = self._now - delta
            return {
                "from_dt": self._fmt(from_dt),
                "to_dt":   self._fmt(self._now),
                "date":    from_dt.strftime("%Y-%m-%d"),
            }
        return None

    def _resolve_time_of_day_named(self, text: str) -> Optional[Dict]:
        """'this morning', 'this afternoon', 'this evening'"""
        today = self._now.date()
        if "this morning" in text or "this dawn" in text:
            return self._day_range(today, 6, 12)
        if "this afternoon" in text:
            return self._day_range(today, 12, 18)
        if "this evening" in text:
            return self._day_range(today, 18, 22)
        if "tonight" in text:
            return self._day_range(today, 20, 24)
        return None

    def _resolve_last_night(self, text: str) -> Optional[Dict]:
        """'last night'"""
        if "last night" in text:
            yesterday = (self._now - timedelta(days=1)).date()
            today     = self._now.date()
            return {
                "from_dt": f"{yesterday} 20:00:00",
                "to_dt":   f"{today} 06:00:00",
                "date":    str(yesterday),
            }
        return None

    def _resolve_yesterday_with_time(self, text: str) -> Optional[Dict]:
        """'yesterday at 10am', 'yesterday 2:30pm'"""
        if "yesterday" not in text:
            return None
        yesterday = (self._now - timedelta(days=1)).date()
        time_info = self._extract_clock_time(text)
        if time_info:
            from_dt = datetime.combine(yesterday, time_info["from_time"])
            to_dt   = datetime.combine(yesterday, time_info["to_time"])
            return {
                "from_dt": self._fmt(from_dt),
                "to_dt":   self._fmt(to_dt),
                "date":    str(yesterday),
            }
        return None

    def _resolve_today_with_time(self, text: str) -> Optional[Dict]:
        """'today at 10am', 'today 2:30pm'"""
        if "today" not in text:
            return None
        today     = self._now.date()
        time_info = self._extract_clock_time(text)
        if time_info:
            from_dt = datetime.combine(today, time_info["from_time"])
            to_dt   = datetime.combine(today, time_info["to_time"])
            return {
                "from_dt": self._fmt(from_dt),
                "to_dt":   self._fmt(to_dt),
                "date":    str(today),
            }
        return None

    def _resolve_yesterday(self, text: str) -> Optional[Dict]:
        """'yesterday' alone — full day"""
        if "yesterday" in text:
            yesterday = (self._now - timedelta(days=1)).date()
            return {
                "from_dt": None,
                "to_dt":   None,
                "date":    str(yesterday),
            }
        return None

    def _resolve_today(self, text: str) -> Optional[Dict]:
        """'today' alone — full day"""
        if "today" in text:
            return {
                "from_dt": None,
                "to_dt":   None,
                "date":    str(self._now.date()),
            }
        return None

    def _resolve_clock_time(self, text: str) -> Optional[Dict]:
        """
        '10am', '2pm', '10:30am', '14:00' — anchored to TODAY.
        This is the key fix: no date context = defaults to today.
        """
        today     = self._now.date()
        time_info = self._extract_clock_time(text)
        if time_info:
            from_dt = datetime.combine(today, time_info["from_time"])
            to_dt   = datetime.combine(today, time_info["to_time"])
            return {
                "from_dt": self._fmt(from_dt),
                "to_dt":   self._fmt(to_dt),
                "date":    str(today),
            }
        return None

    # ─────────────────────────────────────────────────────────────
    # PRIVATE — helpers
    # ─────────────────────────────────────────────────────────────

    def _extract_clock_time(self, text: str) -> Optional[Dict]:
        """
        Extract a clock time from text and return from_time / to_time.
        "10am"    → from=10:00, to=10:59
        "10:30am" → from=10:30, to=10:59
        "14:00"   → from=14:00, to=14:59
        """
        from datetime import time as dtime

        # HH:MMam/pm
        m = re.search(r'\b(\d{1,2}):(\d{2})\s*(am|pm)?\b', text)
        if m:
            h, minute = int(m.group(1)), int(m.group(2))
            meridiem  = m.group(3)
            h = self._to_24h(h, meridiem)
            if 0 <= h <= 23:
                return {
                    "from_time": dtime(h, minute, 0),
                    "to_time":   dtime(h, 59, 59),
                }

        # HHam/pm (no minutes)
        m = re.search(r'\b(\d{1,2})\s*(am|pm)\b', text)
        if m:
            h        = int(m.group(1))
            meridiem = m.group(2)
            h = self._to_24h(h, meridiem)
            if 0 <= h <= 23:
                return {
                    "from_time": dtime(h, 0, 0),
                    "to_time":   dtime(h, 59, 59),
                }

        # "at 10" or "at 14" — no am/pm
        m = re.search(r'\bat\s+(\d{1,2})\b', text)
        if m:
            h = int(m.group(1))
            if 0 <= h <= 23:
                return {
                    "from_time": dtime(h, 0, 0),
                    "to_time":   dtime(h, 59, 59),
                }

        return None

    def _to_24h(self, h: int, meridiem: Optional[str]) -> int:
        """Convert 12h to 24h."""
        if meridiem is None:
            return h
        meridiem = meridiem.lower()
        if meridiem == "am":
            return 0 if h == 12 else h
        else:  # pm
            return 12 if h == 12 else h + 12

    def _day_range(self, day: date, from_h: int, to_h: int) -> Dict:
        """Return from/to covering hours from_h to to_h on given day."""
        from datetime import time as dtime
        if to_h == 24:
            to_dt = datetime.combine(day + timedelta(days=1), dtime(0, 0, 0))
        else:
            to_dt = datetime.combine(day, dtime(to_h, 0, 0))
        return {
            "from_dt": self._fmt(datetime.combine(day, dtime(from_h, 0, 0))),
            "to_dt":   self._fmt(to_dt),
            "date":    str(day),
        }

    def _fmt(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    def _enrich(self, question: str, resolved: Dict) -> str:
        """Append resolved timestamps to question for LLM router context."""
        parts = [f"today={self._now.strftime('%Y-%m-%d')}"]
        if resolved.get("date"):
            parts.append(f"date={resolved['date']}")
        if resolved.get("from_dt"):
            parts.append(f"from_dt={resolved['from_dt']}")
        if resolved.get("to_dt"):
            parts.append(f"to_dt={resolved['to_dt']}")
        hint = " | ".join(parts)
        return f"{question} [RESOLVED: {hint}]"
