"""Brazilian operational calendar utilities.

Encapsulates the logic discovered during EDA:

* Sunday is a non-operational day.
* National federal holidays are non-operational.
* Carnival (Monday and Tuesday before Ash Wednesday) is non-operational
  even though it's a "facultative" holiday in Brazil.
* Corpus Christi (Thursday 60 days after Easter) is non-operational
  for the same reason.

Easter is computed via the Anonymous Gregorian algorithm so we avoid
adding external dependencies. The algorithm is valid for years
1583-4099.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, timedelta

import holidays


def compute_easter(year: int) -> date:
    """Return the date of Easter Sunday for the given year.

    Uses the Anonymous Gregorian (Gauss) algorithm. Pure arithmetic,
    no external libraries.

    Args:
        year: A year in [1583, 4099]. Outside this range the result
            is not guaranteed correct.

    Returns:
        The date of Easter Sunday.

    Example:
        >>> compute_easter(2024).isoformat()
        '2024-03-31'
        >>> compute_easter(2017).isoformat()
        '2017-04-16'
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def build_br_operational_holidays(years: Iterable[int]) -> set[date]:
    """Return the set of non-operational dates for Brazilian shipping.

    Combines:
      * National federal holidays from the ``holidays`` library.
      * Carnival Monday and Tuesday (48 and 47 days before Easter).
      * Corpus Christi (60 days after Easter, always a Thursday).

    Args:
        years: Iterable of years to cover.

    Returns:
        A set of ``date`` objects representing non-operational days.
    """
    years_list = list(years)
    op_holidays: set[date] = set()

    # National federal holidays (Labor Day, Independence Day, etc.)
    op_holidays.update(holidays.country_holidays("BR", years=years_list).keys())

    for year in years_list:
        easter = compute_easter(year)
        # Carnival: the Monday and Tuesday before Ash Wednesday.
        # Ash Wednesday = Easter - 46 days, so Carnival = -48 (Mon) and -47 (Tue).
        op_holidays.add(easter - timedelta(days=48))
        op_holidays.add(easter - timedelta(days=47))
        # Corpus Christi: 60 days after Easter (always a Thursday).
        op_holidays.add(easter + timedelta(days=60))

    return op_holidays


def get_black_friday(year: int) -> date:
    """Return the date of Black Friday for the given year.

    Black Friday is defined as the Friday immediately following the
    fourth Thursday of November (i.e., the day after US Thanksgiving).
    This is the canonical commercial definition adopted globally,
    including by Brazilian retailers.

    Note:
        We do NOT use "last Friday of November". In years where Nov 30
        is a Friday (like 2018), the real Black Friday is one week earlier.

    Args:
        year: A year (any).

    Returns:
        The date of Black Friday.

    Example:
        >>> get_black_friday(2017).isoformat()
        '2017-11-24'
        >>> get_black_friday(2018).isoformat()
        '2018-11-23'
        >>> get_black_friday(2019).isoformat()
        '2019-11-29'
    """
    # Find the fourth Thursday of November
    thursdays_found = 0
    d = date(year, 11, 1)
    while thursdays_found < 4:
        if d.weekday() == 3:  # Thursday
            thursdays_found += 1
            if thursdays_found == 4:
                break
        d += timedelta(days=1)
    # Black Friday = the day after the fourth Thursday
    return d + timedelta(days=1)


def get_dia_dos_namorados(year: int) -> date:
    """Return the date of Dia dos Namorados (Brazilian Valentine's Day).

    Always falls on June 12th, regardless of weekday.

    Args:
        year: A year (any).

    Returns:
        The date of Dia dos Namorados.
    """
    return date(year, 6, 12)
