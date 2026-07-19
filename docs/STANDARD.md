# Bar-Data Integrity Standard

**Status:** v1.0 (public edition)
**Companion:** this is the method behind the [`tapetruth`](../README.md) engine.

---

## 1. What this standard is

A doctrine for keeping corporate-action data and OHLCV bars — raw, adjusted, daily,
intraday — **correct, continuously verifiable, and honestly annotated**. It generalizes
cleanly across any equity data pipeline: nothing below depends on a specific broker,
vendor, or exchange, only on the shape of the problem.

The one-sentence version: **records are hypotheses; the tape is truth; every correction is
a measured, load-time view; and agreement is proven against independent sources, never
assumed.**

## 2. The core doctrine

**D1 — Tape is truth.** Corporate-action records (any source: a vendor feed, a regulatory
filing, a text-extraction pipeline) are *claims*. A claim is applied only where your own
as-traded price series — immune to adjustment-factor bugs by construction — confirms it:
the right ratio, at the right date, with next-bar persistence. Claims the tape contradicts
are refuted; claims the tape locates elsewhere are snapped to where the tape actually moves.

**D2 — Append-only records, view-level corrections.** The underlying records are never
mutated in place. Every fix (de-duplication, date-snapping, refutation, exclusion) happens
at load time inside one canonical chain, and every decision is auditable. Fixes are
therefore reversible, replayable, and self-documenting.

**D3 — One chain, one chokepoint.** Every consumer of corporate-action data should run
through the same collapse chain; every writer of corrected bars should run through the same
sanity guard. The single most common way this class of bug reappears is "a fix applied to
one code path but not another" — a rule that doesn't live at a shared chokepoint doesn't
really exist.

**D4 — External reconciliation.** Internal self-consistency proves nothing about
correctness. A pipeline should validate against **independent external sources** — a broker
feed, a reference vendor (rented for calibration, not depended on permanently), and
internal invariants. A defect should be measured against an external source before repair,
and the repair accepted only once re-measurement shows the divergence closed.

**D5 — Vendors point, they don't write.** A rented reference feed tells you WHERE to look.
Every fact you actually store should be confirmed by your own tape or a public record. When
a subscription or feed relationship ends, nothing in your pipeline should depend on
retained third-party data.

## 3. The twelve rules

| # | Rule | Rationale | In `tapetruth` |
|---|------|-----------|----------------|
| 1 | Duplicate events collapse by date-cluster; canonical pick uses source rank, with the measured price-gap overriding rank when available | Multiple sources logging the same real event at slightly different dates (announcement vs. filing vs. market-effective date) compounds a single event's ratio N times if every row is multiplied in | `chain.collapse_duplicate_actions` |
| 2 | Contradictory same-date and near-date claims resolve by the measured gap | A split and its exact inverse (or two different ratios) logged for the same real event cancel to a no-op, or compound into a fictitious combined factor, if both survive | `chain.collapse_same_date_conflicts`, `chain.collapse_near_date_conflicts` |
| 3 | Actions dated after a symbol's final bar are inapplicable | No boundary exists in the data past the last bar — applying the factor anyway rescales the *entire* series, not just the bars after a real boundary | `chain.drop_post_coverage_actions` |
| 4 | Vendor mislabels confirmed by a public record are excluded by citation, never by tape | A spin-off's value-separation price drop can be numerically indistinguishable from a real split of the same ratio — the tape can't tell the difference; only a cited public record can | `chain.drop_recorded_mislabels` |
| 5 | Snap misdated large actions to the unique tape boundary | A wrong date breaks every date-anchored downstream consumer (a refuter kills a real-but-misdated split; a vintage classifier misreads the boundary) — one cause, several symptoms | `chain.snap_actions_to_tape` + `locator.find_unique_boundary` |
| 6 | Refute lone phantom claims the tape contradicts, but never on a pre-adjusted vintage | A claim with no tape support is very likely fabricated or misattributed — unless the series arrived already adjusted upstream, in which case the price contribution is legitimately suppressed and only a vintage flag can tell the two apart | `chain.refute_phantom_actions` (vintage-aware via an injectable hook) |
| 7 | Any static implausible-ratio cap needs a tape-confirmation escape hatch | Real ratios in illiquid or thinly-covered names can exceed any static "plausible" threshold — a cap without an escape kills genuine events | Doctrine — `refute_phantom_actions` never refutes a claim the tape confirms, at any ratio; this repository does not ship a separate static ratio cap |
| 8 | Daily high/low fields inside action windows should be validated (and, where possible, repaired) against an independent sub-daily source | A per-field vintage fix corrects *when* a field was adjusted, but not bad individual trade prints (odd-lot, out-of-sequence, error prints) that leak into a high or low; a real field is never a wild multiple away from its own close | `guard.apply_bad_print_guard` |
| 9 | Repair only inside evidence-scoped jurisdictions | A magnitude test that can't distinguish a real defect from an ordinary feed-definition gap must be scoped to where defects are actually proven to occur, or it silently rewrites correct data at scale | `guard`'s dense-coverage jurisdiction (`dense_min_bars`) |
| 10 | Write-invariants are enforced at the repair chokepoint *and* validated at rest | The measurement-sanity bound (is this truth value itself plausible?) and the output contract bound (what every downstream consumer is promised) are different things — a repair should respect both | `guard`'s band-clamp (chokepoint); validating "at rest" is your pipeline's job downstream of this engine |
| 11 | Absence detection should use anti-joins against an independent expected-set, never a step's own bookkeeping | A step that only checks its own record of what it received can't detect a hole its own bookkeeping doesn't know about | Doctrine — outside this engine's scope (tapetruth verifies data you already have; it does not monitor an ingestion pipeline for silent gaps) |
| 12 | Every writer should be atomic; every output file should be verified after write | A bare, non-atomic write over a file being concurrently read can silently truncate it | Doctrine — outside this engine's scope (tapetruth operates on in-memory DataFrames; how you persist the result is your pipeline's concern) |

## 4. Standing machinery (recommended cadence)

| Machinery | Recommended cadence | Catches |
|---|---|---|
| Full collapse chain (`chain.collapse_all`) | Every time you (re)build adjustment factors or adjusted bars | Rules 1–6 |
| Bad-print guard (`guard.apply_bad_print_guard`) | Every write of corrected/adjusted bars | Rules 8–10 |
| A corpus-level invariant check (residual-out-of-band count, factor blow-ups, coverage) | Scheduled (e.g. weekly) plus after any substrate change | Structural regressions across the whole dataset |
| Reconciliation against a broker feed (`reconcile`) | On-demand, and after any substrate change | Agreement against an independent, typically free/low-cost feed |
| Reconciliation against a reference vendor (`reconcile`) | While a calibration subscription is active (see D5) | Agreement against a second, independent, typically paid feed |

## 5. Acceptance metrics

What "correct" means, numerically, on the included synthetic gauntlet
(`python -m tapetruth.demo`):

- **Detection floor:** at least 80% of planted defects, across the eight `defect`-category
  classes, correctly caught and repaired or excluded.
- **False-positive ceiling:** at most 2% of clean-symbol bars incorrectly altered.
- **Legitimate-preservation:** the three `legitimate`-category classes (things that look
  like defects but are real) should be preserved at a near-100% rate — an engine that's
  merely aggressive isn't the same as one that's correct.
- **Known gaps reported, never hidden:** any defect class the current engine does not catch
  is scored and reported honestly, excluded from the headline detection number, not folded
  in as a false "pass."

For your own external-reconciliation work (`tapetruth.reconcile`): the target is a high
pass rate (EXACT + MINOR + EXACT_ADJUSTED_EQUIV + EXPLAINED) on the *comparable* set,
with `NOT_COMPARABLE` symbols reported but excluded from the denominator, and zero
unexplained `MISMATCH` on any symbol you'd consider "pinned" or load-bearing.

## 6. The honest-classes principle

A reconciler that can only say "match" or "mismatch" hides information. `tapetruth.reconcile`
uses six classes instead of a boolean:

- **EXACT** — the two series agree everywhere they overlap, within tight tolerance.
- **MINOR** — small, bounded disagreement (rounding, day-count conventions) — worth
  knowing, not worth alarming over.
- **EXACT_ADJUSTED_EQUIV** — the two series are parallel but offset by a constant: every
  period-over-period step agrees, they just anchor their cumulative product to a different
  starting vintage. A real, common, benign pattern — not a disagreement about any actual
  event.
- **EXPLAINED** — a real disagreement exists, but it has a specific, citable, accepted
  reason (a known scope difference, a documented one-sided gap).
- **NOT_COMPARABLE** — not enough overlapping data to say anything at all. Reported, never
  silently hidden, never counted as a failure.
- **MISMATCH** — a real, unexplained disagreement. The only class that means "something
  might actually be wrong."

## 7. Vendor / reference-feed posture

Treat any rented, paid reference feed as a **time-boxed calibration instrument**, not a
permanent dependency: rent it, reconcile against it, convert every finding into a
tape-confirmed or public-record-confirmed fix, re-measure until the divergence closes,
then let the subscription lapse. What survives is a pipeline whose correctness depends on
your own tape and public records — never on continued access to someone else's feed.

## 8. Known gaps

Tracked honestly, not hidden — a standard that hides its own limitations isn't one you can
trust:

1. **Isolated bad-close prints are not currently caught.** The bad-print guard validates
   open/high/low *against* the close of the same row, but trusts the close itself. An
   isolated, internally-self-consistent bad close (the whole candle moves together, so
   geometry checks can't distinguish it from a real move) with a full next-day reversal
   currently survives. This is modeled explicitly as the `bad_close_vspike` class in the
   included gauntlet, and is reported there — honestly, at its true (near-zero) detection
   rate — rather than silently excluded from the benchmark.
2. **The bad-print guard's truth-confirmation and jurisdiction thresholds are tuned, not
   derived.** Like the broader tick-cleaning literature this method draws on (see §9), the
   specific tolerance values are empirically chosen, not theoretically proven optimal —
   they are exposed as `GuardConfig`/`ChainConfig` fields precisely so you can retune them
   against your own data's characteristics.
3. **Absence/coverage monitoring (rules 11–12) is out of this engine's scope.** tapetruth
   verifies data you already have; it does not watch an ingestion pipeline for silent gaps
   or enforce atomic writes. Those remain your pipeline's responsibility.

## 9. Academic lineage

This method's general shape — discipline one data stream against an independent one,
tolerance-banded, never simply assumed — has real academic precedent, though the specific
corporate-action application here (tape-confirmed de-duplication, boundary-snapping,
vintage-aware phantom refutation, honest-class reconciliation) is, to the best of the
authors' knowledge, not previously published as a general, open, reusable tool (see the
prior-art discussion in the project's release notes for the adversarial literature sweep
behind that claim).

- **CRSP price/share adjustment-factor methodology** (the `CFACPR`/`CFACSHR` two-factor
  system) is the closest methodological ancestor — and has itself documented,
  acknowledged gaps in exactly this problem space. Glasscock, R., "Stock Price Adjustment
  Factors in COMPUSTAT and CRSP" (2016) documents that CRSP/COMPUSTAT adjustment factors
  are "inaccurate in the presence of property dividend, spin-off, and rights offering
  events" — the identical defect family (price-factor vs. share-factor divergence around
  non-split corporate actions) that rule 4 targets. Even the closest thing the industry has
  to a gold standard acknowledges the problem is unsolved in general.
- **Barndorff-Nielsen, O.E., Hansen, P.R., Lunde, A., Shephard, N., "Realized kernels in
  practice: trades and quotes,"** *The Econometrics Journal*, 12(3) (2009) — the standard
  tick-data-cleaning taxonomy (rules disciplining one stream against an independent one,
  tolerance-banded) is the closest academic kin to the bad-print guard's design. The paper
  is explicit that its own rules depend on ad hoc tuning parameters — the same honest
  limitation acknowledged in §8 above.

---

*Change protocol: this standard should be updated before the code that implements it, and
the code before any claim of conformance. A rule not reflected in an actual, runnable
mechanism (or explicitly marked "doctrine" / "known gap") is not a rule anyone can rely on.*
