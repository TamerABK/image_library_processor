# Room 36 Design Reference

The files in `design_reference/screens/` are visual ground truth for the QML
migration. They are not runtime assets and must not be loaded as complete screen
SVGs from QML.

Expected reference pairs:

- `loading.svg` / `loading.png`
- `signup.svg` / `signup.png`
- `home.svg` / `home.png`
- `blurry_photos.svg` / `blurry_photos.png`
- `known_people.svg` / `known_people.png`
- `unknown_people.svg` / `unknown_people.png`

Runtime UI should be rebuilt with native QML surfaces, text, reusable controls,
and curated assets from `assets/icons/`, `assets/branding/`,
`assets/backgrounds/`, and `assets/fonts/`.
