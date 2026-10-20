# Phase 8B: Favorites

Favorites are project metadata in the application-state SQLite database's
`favorite_photos(folder, path)` table. Both keys use the existing normalized
`photo_identity` convention. Favorites never use `image_decisions`, Blur review,
duplicate group reviews, People results, or processing cache signatures.

One `FavoritesController` owns the current project's favorite set and a
`PhotoListModel` containing only paths in the master discovered photo model.
`ProjectController.projectOpening` clears the previous project immediately;
`projectOpened` starts a metadata load after discovery. The Favorites page
reuses `PhotoGrid`, bounded thumbnails, and the existing Inspector. Visible
cards and Inspector hearts subscribe to favorite/pending signals, so a saved
change updates them and the sidebar count without image analysis.

One bounded worker serializes SQLite operations. Repeated toggles of the same
photo are disabled until its write finishes; presentation changes only after
successful persistence. Each result carries its folder and generation, so an
old project's completion cannot update the current project. Database failures
leave the visible state unchanged and show a local safe message.

The Inspector keeps the current preview and photo metadata if that favorite is
removed from its grid, and Previous/Next skip photos no longer present in the
Favorites model. Same-path external edits preserve a favorite. Missing paths
stay in metadata but are absent from the current model and count until they
reappear in discovery.

A photo may currently be both a favorite and have a Blur or Duplicate REMOVE
intent; neither workflow rewrites the other. Future Trash review can address
that conflict before any physical action. This phase performs no deletion or
export.
