.pragma library

// Navigation metadata only. No result rows, sample counts or person identities.
var entries = [
    { routeId: "library", label: "All", title: "Library", icon: "view.svg", section: "Organize",
      message: "" },
    { routeId: "blurry", label: "Blurry photos", title: "Blurry photos", icon: "landscape.svg", section: "",
      message: "Blurry photo review will be added in Phase 6." },
    { routeId: "duplicates", label: "Near duplicates", title: "Near duplicates", icon: "view_medium.svg", section: "",
      message: "Duplicate review will be added in a later feature phase." },
    { routeId: "favorites", label: "Favorites", title: "Favorites", icon: "favorite.svg", section: "",
      message: "Favorites will be connected in a later feature phase." },
    { routeId: "knownPeople", label: "Known people", title: "Known people", icon: "profile.svg", section: "People",
      message: "Known people content will be added in Phase 7." },
    { routeId: "unknownPeople", label: "Unknown people", title: "Unknown people", icon: "profile.svg", section: "",
      message: "Unknown people review will be added in Phase 7." },
    { routeId: "trash", label: "Trash", title: "Trash", icon: "trash.svg", section: "Settings",
      message: "Trash content will be connected in a later feature phase." }
]

function entry(routeId) {
    for (var i = 0; i < entries.length; ++i) {
        if (entries[i].routeId === routeId) {
            return entries[i]
        }
    }
    return entries[0]
}

function normalize(routeId) {
    return entry(routeId).routeId
}
