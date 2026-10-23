# Active-project filesystem watcher

`ProjectFilesystemWatcher` owns one Qt `QFileSystemWatcher` for the active
project. It watches the root, accessible subdirectories, and the root's parent.
It never watches individual photos. One bounded worker enumerates directories;
Qt watch registration and all presentation changes remain on the Qt thread.
After successful rediscovery it rebuilds the directory watch set, including new
nested folders. Failed registrations are logged, exposed as a degraded state,
and trigger a low-frequency fallback refresh request.

Directory notifications restart one 900 ms quiet-period timer. The watcher
only calls `ProjectController.refreshActiveProject()`. Discovery coalesces
requests during an active refresh into at most one follow-up. The normal
cache-first processing pipeline handles added, removed, renamed, modified, and
restored photos; the watcher neither interprets review decisions nor decodes
images. An explicit same-project refresh absorbs notifications received before
its discovery starts, avoiding a redundant refresh after Room36 Trash moves.
Events during discovery can request one bounded follow-up when needed.

Switching projects removes old watches and invalidates old enumeration results.
Notifications from the previous project's directories cannot refresh the new
project. A parent-directory watch can detect an unavailable root reappearing.
Shutdown stops timers, removes watches, and joins the enumeration worker.

An OS-level restore is not a Room36 undo operation. Restored files may regain
valid persisted REMOVE review intent when path, size, and modification time
match, but never regain a prior temporary Trash batch selection. Sending them
to Trash again requires new selection and explicit confirmation. The user can
instead choose Cancel removal. Export destinations remain outside the project,
so export does not trigger project reconciliation.
