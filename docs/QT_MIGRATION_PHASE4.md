# Phase 4: Library shell

The application Loader owns Home/Auth/Loading/Library/Gallery. LibraryShell owns
its internal currentRoute. Changing an internal route keeps the shell, sidebar,
and top bar mounted. All current destinations share LibraryRoutePlaceholder;
its route/title/message bindings change without allocating feature pages.
Later implementations replace the content Loader selection only.

## Contracts

- ProjectGrid -> HomePage (or development ScreenPreview) emits
  projectActivated(projectId, projectName). App.openProject rejects absent IDs,
  stores activeProjectId/activeProjectName, and selects the library root route.
- App.projectModel is an optional adapter input; no project database was added.
  Production Home stays empty until supplied. New-project/menu actions are
  forwarded as signals, with no fake creation or persistence.
- LibraryShell receives projectId/projectName independently of the preview model.
  Context survives internal route changes. Home destroys the shell; opening a
  subsequent project supplies new context and starts at the library route.
- showRoute(routeId), or assigning currentRoute, normalizes unknown IDs to library.
  Unknown root routes normalize to home.
- LibraryShell.routeCounts is an optional map keyed by route ID. Replace the map
  when updating metadata. Missing/negative/non-numeric counts are hidden; zero
  is a valid displayed count. No backend counts are inferred.
- Sort/View emit route-aware requests; App adds the active project ID. Preferences
  emits settingsRequested. These signals have no feature implementation yet.

## Routes and design

LibraryRoutes.js is the shared declarative navigation metadata:
library (All), blurry (Blurry photos), duplicates (Near duplicates), favorites
(Favorites), knownPeople (Known people), unknownPeople (Unknown people), trash
(Trash). These categories follow the existing blur/people design references.
All destinations explicitly identify future content. No photo grid is included.

The sidebar uses SidebarFilterItem, the existing gradient palette, white logo,
and existing icons. Home and Preferences are actions, not internal content routes.
The top bar shows project/page context plus Home, Sort by, and View. The approved
Library references justify these controls; no search/profile/export workflow was
invented. The workspace uses Theme.libraryBackground (black).

Metrics define a 430 px sidebar at widths >=1600 and 320 px below, a 148 px top
bar, and 48 px content margins. Sidebar navigation can scroll when constrained.
Buttons retain keyboard activation, focus treatment, and accessible names.

## Development and validation

ROOM36_START_PAGE=library opens the shell without project context for inspection.
ROOM36_HOME_PREVIEW_DATA=1 enables the existing opt-in Home sample projects;
activating a card exercises Home -> Library -> Home without permanent storage.
Production Quick Controls style selection remains Basic unless explicitly overridden.

Run the five Phase 1-3 unittest modules plus ui.test_qml_library_shell with
QT_QPA_PLATFORM=offscreen, QML_DISABLE_DISK_CACHE=1, QT_SHADER_CACHE_DISABLE=1,
and QT_QPA_FONTDIR pointing to assets/fonts on Windows. Use python -B. Do not set
QT_QUICK_CONTROLS_STYLE merely for the tests: production setup selects it.

Tests cover all routes/fallbacks, instance persistence, project activation and
replacement, metadata, request signals, keyboard activation, and geometry at
1920x1080, 1440x900, and 1280x720. Qt/QML warnings are captured and fail tests;
fresh-process startup checks exercise both Basic and an explicit Fusion override.

No backend algorithms or bridge APIs changed. Photo models, thumbnail providers,
scans, review pages, people workflows, export, and project persistence remain out
of scope. Sort/View/Preferences are interface contracts only.
