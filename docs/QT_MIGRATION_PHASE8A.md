# Phase 8A: near-duplicate review

The Near duplicates page observes `ProjectProcessingController`. It reads the central
`duplicates` stage's path groups immediately and enriches them when the `ranking`
stage publishes `AutomaticDuplicateGroup` dictionaries. Opening the route, sorting,
filtering, and review actions do not run pHash, dHash, ORB, keeper ranking, or any
other image analysis. If ranking fails, group membership stays visible but keeper
actions wait for a successful central retry.

`DuplicateReviewController` projects stage payloads into `DuplicateGroupModel` and
a `PhotoListModel` for the selected group. Both grids are virtualized. Visible group
cards acquire bounded `ThumbnailService` leases; the detail grid and existing
Inspector use the same preview infrastructure as other library routes. The
recommendation and explanation come directly from the ranking result. Manual
selection preserves the ranking result's keeper count.

The group's backend ID is a digest of its sorted member paths. It can change when
membership changes. The `duplicate_group_reviews` table in the existing Room 36
app-state database stores a user's keeper paths, per-member `KEEP`/`REMOVE` intents,
and explicit reviewed flag. The table is separate from `image_decisions` because
that row's `review_state` is also used by Blur review; duplicate actions never
overwrite a Blur choice. A conflicting explicit Blur choice blocks bulk confirmation
with a safe message. `REMOVE` here is review intent only: no source file is moved or
deleted. Restoring an alternative clears its duplicate `REMOVE` intent and returns
the group to Needs review.

Each saved group includes a signature of member paths, file sizes, and modification
times. Changed or missing files invalidate saved keeper/review choices. The one
bounded review worker checks live revisions before writing. Project switches and
processing retries cancel old operations, clear models and Inspector state, and
discard stale results by session and folder. All Qt model updates run on their
owning thread.

This phase does not execute Trash, add Favorites, export photos, or change duplicate
detection and ranking algorithms. Those workflows are deferred.
