# Phase 8D — Safe photo export

`ExportController` is one project-scoped Qt presentation owner. Library, Favorites,
Blur, duplicate detail, and People detail pass their existing photo model to
`prepareSelected`; Trash deliberately has no export action. Selected identities
must still belong to both that model and the active master Library model. Selection
is temporary UI state and is kept after export. Export never writes review,
Favorites, People, or Trash metadata and never starts project processing.

The controller snapshots source size and nanosecond modification time on one
bounded `room36-export` worker. A plan is immutable after preparation. The user
chooses Copy or Hard link and a destination in a shared QML dialog, then confirms.
Before each placement the worker checks the source is still a regular file with
the snapshotted revision. Changed or missing files fail individually; other files
continue. Cancellation stops before the next file, keeps completed outputs, and
reports completed and failed counts. Progress counts output placements.

Copy uses `shutil.copy2`. Hard link uses `os.link` and never silently copies on
failure. Hard links need a compatible source and destination filesystem, generally
the same drive on Windows. They share underlying file data: editing through either
path affects both; removing one path leaves data accessible through the other.
Hard links are not backup copies.

Output names use `photo.jpg`, `photo_1.jpg`, and so on. Exclusive creation for
Copy and atomic link creation for Hard link reject an existing name, including a
name created between collision checks. Export does not overwrite destinations.
The chosen destination is rejected when it equals or lies inside the project,
including through a symlinked ancestor. No destination directory is created
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
result are discarded from a different active project's UI. Export requires no
project rescan because source files are not changed. The one worker and bounded
queues prevent one-thread-per-photo behavior. Current limitations: no symbolic
links, no export from Trash, and no automatic retry or rollback of partial output.
