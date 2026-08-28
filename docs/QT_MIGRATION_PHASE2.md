# Qt Migration Phase 2

Phase 2 adds the reusable Room 36 QML design-system foundation. It does not
implement production pages such as LoadingPage, AuthPage, HomePage, or
LibraryShell.

## Component Tree

```text
qml/
├── App.qml
├── theme/
│   ├── Theme.qml
│   ├── Typography.qml
│   ├── Metrics.qml
│   ├── Animations.qml
│   └── Effects.qml
├── components/
│   ├── PrimaryButton.qml
│   ├── PillButton.qml
│   ├── IconButton.qml
│   ├── SearchField.qml
│   ├── ToolbarDropdownButton.qml
│   ├── SidebarFilterItem.qml
│   ├── SelectionCheckbox.qml
│   ├── ScoreBar.qml
│   ├── PersonAvatar.qml
│   ├── PhotoCard.qml
│   └── ProjectCard.qml
└── dev/
    └── ComponentGallery.qml
```

`App.qml` hosts `qml/dev/ComponentGallery.qml` for this phase. The gallery
shows the palette, typography, reusable controls, card delegates, and a small
bridge debug section with folder, status, progress, Start Scan, and Cancel.

## Asset Cleanup

Runtime assets are limited to the curated categories:

```text
assets/icons/
assets/branding/
assets/backgrounds/
assets/fonts/
```

Removed duplicate or raw Illustrator leftovers:

- `assets/backgrounds/loading screen - background mesh.svg`
- `assets/backgrounds/page 1 - header -  export button rectangle.svg`

Kept:

- `assets/backgrounds/loading_mesh.svg`

Full-screen Illustrator exports remain design references only and are not
loaded by runtime QML.

## Design Reference

`design_reference/README.md` documents the expected reference screen pairs:

- `loading.svg` / `loading.png`
- `signup.svg` / `signup.png`
- `home.svg` / `home.png`
- `blurry_photos.svg` / `blurry_photos.png`
- `known_people.svg` / `known_people.png`
- `unknown_people.svg` / `unknown_people.png`

These files are visual ground truth only.

## Design Tokens

`Theme.qml` preserves the exact Room 36 palette:

```text
mainBlue   #00008E
midBlue    #4B44E0
lightBlue  #A4A4FF
neonGreen  #D7F205
white      #FFFFFF
black      #000000
```

Semantic colors are defined centrally as palette derivatives or opacity
variants, including `workspaceBackground`, `panelFill`, `mutedText`,
`hoverFill`, `pressedFill`, `selectedFill`, `separator`, `focusRing`, and
`selectedGlow`. Components should consume these names instead of raw hex values.

## Typography

`Typography.qml` loads the runtime Britanica fonts and keeps fallback behavior
through a hidden QML `Text` probe. The actual Qt family names are distinct:

```text
Britanica Thin.ttf     -> Britanica-Thin
Britanica Regular.ttf  -> Britanica-Regular
Britanica Bold.ttf     -> Britanica-Bold
```

Semantic size tokens are exposed as `display`, `pageTitle`, `sectionTitle`,
`cardTitle`, `body`, `button`, `caption`, and `detail`. Runtime UI uses real
QML text, not SVG text.

## Metrics

`Metrics.qml` keeps the 1920x1080 Illustrator artboard as a reference and adds
semantic tokens for photo cards, toolbar controls, sidebar rows, inspector
elements, and auth controls. These values are reference measurements, not fixed
screen coordinates.

Examples:

- `photoCardReferenceWidth`, `photoCardAspectRatio`, `photoCardFooterHeight`
- `toolbarDropdownWidth`, `toolbarExportButtonWidth`, `toolbarButtonHeight`
- `sidebarPreferredWidth`, `sidebarNavRowHeight`, `sidebarCheckboxSize`
- `scoreBarWidth`, `personAvatarSize`, `notesFieldHeight`
- `authFieldWidth`, `authPrimaryButtonHeight`, `authSocialButtonSize`

## Animations And Effects

`Animations.qml` centralizes restrained timings:

```text
hover      110 ms
press       90 ms
selection  160 ms
popup      160 ms
panel      200 ms
page       220 ms
```

`Effects.qml` uses Qt 6 `QtQuick.Effects` and `MultiEffect`. The reusable glow
is white at 50% opacity with zero offset and soft blur. Components enable it
only for selected, focused, or hovered surfaces.

## Asset Path Convention

QML resolves assets through `Theme.qml` helpers:

```qml
theme.iconUrl("search.svg")
theme.brandingUrl("logo_blue.svg")
theme.backgroundUrl("loading_mesh.svg")
theme.fontUrl("Britanica Regular.ttf")
```

The helpers use `Qt.resolvedUrl("../../assets/...")` from `qml/theme/Theme.qml`,
so source-tree launch does not depend on the current working directory.
PyInstaller packaging is intentionally unchanged in this phase.

## Tests

Added `ui/test_qml_design_system.py` smoke coverage for:

- QML root load with zero QML warnings
- Critical component instantiation
- Runtime asset path resolution
- Removed raw runtime assets
- Exact theme palette values
- Actual Britanica Qt family names

Updated `ui/test_qt_bridge.py` to create a `QApplication` with
`QT_QPA_PLATFORM=offscreen`, allowing bridge and QML smoke tests to run in the
same process.

## Limitations

- The component gallery uses placeholder/sample data only.
- Sorting, filtering, menus, and photo/project models are not connected to the
  backend yet.
- The transitional bridge still exposes only the Phase 1 scan state.
- PyInstaller resource packaging should be revisited after production QML pages
  are introduced.

## Recommended Phase 3 Work

- Build production QML pages from the design references using these components.
- Introduce a real library shell and screen routing.
- Bind photo/project/person delegates to Qt-facing models.
- Add visual regression checks against curated reference screenshots.
- Extend packaging tests once QML resources stabilize.
