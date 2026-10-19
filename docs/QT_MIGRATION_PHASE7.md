# Phase 7: Known and Unknown People review

`PeopleController` observes `ProjectProcessingController` and reads its completed
`people` stage payload. Opening either route never starts face detection,
embedding, or clustering. The two pages use virtualized `GridView` models and
reuse `PhotoGrid` and `PhotoInspector` for detail photos. Project-session changes
clear both models, selection, pending previews, and review UI before new results
arrive. During processing or a stage failure, both pages show the stage's true
status; the Phase 6 unknown-face safety limit remains in force.

Known faces are grouped by persisted face-database person ID. The cards show one
stable representative, chosen by largest cached face box and then path/box order,
plus unique photo and face counts. Unknown cards use the representative chosen
by the existing People stage. The People aggregate contains no per-face quality
score, so Phase 7 does not invent a new ranking algorithm.

Naming reads existing embeddings from `FaceScanCache` using current file
size/mtime and the cluster's face boxes. One bounded review worker commits all
vectors atomically through `SQLiteFaceDatabase.assign_embeddings`, either to a
new person or a selected existing person. Names are trimmed, checked for length
and control characters, and matched case-insensitively; Unicode names work.
Nearest-known similarity is shown only as an optional suggestion and never
auto-assigns. Successful review patches the current People presentation without
restarting upstream stages. The face database signature changes after the write,
so the prior People aggregate cache key becomes stale on the next project run;
per-photo face and embedding caches remain reusable.

The stage's integer unknown-cluster ID is transient. Phase 7 derives an ignored
review key from sorted normalized photo paths and face boxes. That key survives
cache reuse and unchanged reopens. Changed cluster membership or face bounds
produce a new key; an edited photo with the same path and face bounds can retain
its ignored key. Ignored keys are stored in the existing application state
database under the project folder. Ignore/restore updates only that row and
does not touch source photos or embeddings. The default queue hides ignored
clusters; Ignored and All filters expose them.

`FacePreviewService` uses the bounded thumbnail scheduler: two workers, at most
64 pending requests, a 16 MiB/96-entry cache, and up to 256-pixel output. It
decodes the same orientation-aware, 2048-pixel image coordinate space used for
face detection, clamps boxes, adds 18% context on each side, and discards the
decode after making the crop. Visible delegates hold cancellable preview leases;
project changes invalidate their requests. No full-resolution image is kept in
the card model.

Current limits: if quadratic unknown clustering exceeds Phase 6's safety cap,
the People stage remains unavailable, including recognized known matches in
that stage's discarded payload. Cluster membership changes can change an
ignored key. Manual cluster splitting, arbitrary known-person merging, export,
and physical photo deletion are deferred.
