# Phase 8D — Safe photo export

`ExportController` is one project-scoped Qt presentation owner. Library, Favorites,
Blur, duplicate detail, and People detail pass their existing photo model to
`prepareSelected`; Trash deliberately has no export action. Selected identities
must still belong to both that model and the active master Library model. Selection
is temporary UI state and is kept after export. Export never writes review,
Favorites, People, or Trash metadata and never starts project processing.

The controller snapshots source size, nanosecond modification time, and file identity on one
bounded `room36-export` worker. A plan is immutable after preparation. The user
chooses Copy or Hard link and a destination in a shared QML dialog, then confirms.
Before each placement the worker opens the source and checks that same handle
against the snapshotted revision and identity. Copy checks it again after streaming;
Hard link checks the created link and current source path. Changed or missing files fail individually; other files
continue. Cancellation stops before the next file, keeps completed outputs, and
reports completed and failed counts. Progress counts output placements.

Copy streams from the verified source handle into the destination handle created
with `O_CREAT | O_EXCL`. It never closes the exclusive handle and reopens the
destination by pathname for writing. It flushes the output and preserves source
mode/timestamps through the handle where the platform supports those operations.
Failure cleanup checks the destination file identity first; a path rebound to
another file is never removed by Room36. Hard link uses `os.link` and never silently copies on
failure. Hard links need a compatible source and destination filesystem, generally
the same drive on Windows. They share underlying file data: editing through either
path affects both; removing one path leaves data accessible through the other.
Hard links are not backup copies.

Output names use `photo.jpg`, `photo_1.jpg`, and so on. Exclusive creation for
Copy and atomic link creation for Hard link reject an existing name, including a
name created between collision checks. Export does not overwrite destinations.
The chosen destination and every actual person output directory are rejected
if they equal or lie inside the project, escape the chosen destination, or resolve
there through a symlink/junction. The worker preflights the root once before
iterating the batch. No destination directory is created
until confirmed execution. Person folder names strip Windows-invalid characters,
trailing spaces/periods, and reserved device names. Distinct people whose names
sanitize alike receive deterministic suffixes.

`Export person` uses the selected known person's current People projection.
`Export all people` uses every known person in the current project, independent
of search filtering. The same photo can be placed in several person folders, but
only once within one person's folder. The dialog distinguishes unique source
photos from output placements and shows unique source bytes.

Project/model or processing-session changes invalidate unconfirmed plans. An
already-confirmed batch may finish for its original project; its progress and
result are discarded from a different active project's UI. Destination and method
cannot change while a batch runs; its completion summary uses the immutable
execution snapshot. Export requires no
project rescan because source files are not changed. The one worker and bounded
queues prevent one-thread-per-photo behavior. Current limitations: no symbolic
links, no export from Trash, and no automatic retry or rollback of partial output.
