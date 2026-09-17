"""
Phase 32: balance-check helper. `modal billing report` reports CUMULATIVE
SPEND over a period, not a remaining-balance figure directly (Modal's CLI
has no direct "remaining credits" command) -- so remaining balance is
tracked here as an anchored delta: the user stated $7.00 remaining at the
exact point phase 31 was stopped, which corresponds to a known total
month-to-date spend figure captured at that same moment (see ANCHOR_SPEND
below, taken from `modal billing report --for "this month"` immediately
after that stop). Every subsequent check re-runs the same report and
subtracts the anchor to get new spend since that point, then subtracts that
from $7.00.

SAFETY_MARGIN = $1.50 per the brief -- if projected remaining balance after
subtracting the anchor falls below this, do not launch another paid run.
"""
import json
import subprocess

ANCHOR_SPEND = 22.29818302   # modal billing report --for "this month" total, at the moment the user
                              # said "$7 left" (captured at the start of this phase, before any new spend)
ANCHOR_REMAINING = 7.00
SAFETY_MARGIN = 1.50


def get_month_to_date_spend():
    # --json avoids a real bug hit during development: --csv's underlying
    # rich-table renderer wraps long Cost values across two lines even in
    # CSV mode when stdout is captured at a narrow width, silently
    # corrupting the numbers (e.g. "6.31566992" split into "6.31" and
    # "566992" on separate lines). --json is not width-wrapped.
    out = subprocess.run(
        ["/Users/haseeb/.pyenv/versions/3.11.6/bin/python3", "-m", "modal", "billing", "report",
         "--for", "this month", "-r", "d", "--json"],
        capture_output=True, text=True, check=True,
    )
    rows = json.loads(out.stdout)
    return sum(float(r["Cost"]) for r in rows)


def check_balance(label=""):
    spend = get_month_to_date_spend()
    remaining = ANCHOR_REMAINING - (spend - ANCHOR_SPEND)
    print(f"[{label}] month-to-date spend=${spend:.4f}  estimated remaining=${remaining:.4f}")
    can_proceed = remaining >= SAFETY_MARGIN
    print(f"  {'OK to proceed' if can_proceed else 'STOP -- below $1.50 safety margin'}")
    return {"label": label, "spend": spend, "remaining": remaining, "can_proceed": can_proceed}


if __name__ == "__main__":
    check_balance("manual check")
