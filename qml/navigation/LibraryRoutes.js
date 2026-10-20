.pragma library

// Navigation metadata only. No result rows, sample counts or person identities.
var entries = [
    { routeId: "library", label: "All", title: "Library", icon: "sidebar_view.svg", section: "Organize",
      message: "" },
    { routeId: "blurry", label: "Blurry photos", title: "Blurry photos", icon: "sidebar_landscape.svg", section: "",
      message: "Blurry photo review will be added in Phase 6." },
    { routeId: "duplicates", label: "Near duplicates", title: "Near duplicates", icon: "sidebar_view_medium.svg", section: "",
      message: "" },
    { routeId: "favorites", label: "Favorites", title: "Favorites", icon: "star_shape.svg", section: "",
      message: "" },
    { routeId: "knownPeople", label: "Known people", title: "Known people", icon: "sidebar_profile.svg", section: "People",
      message: "Known people content will be added in Phase 7." },
    { routeId: "unknownPeople", label: "Unknown people", title: "Unknown people", icon: "sidebar_profile.svg", section: "",
      message: "Unknown people review will be added in Phase 7." },
    { routeId: "trash", label: "Trash", title: "Trash", icon: "trash.svg", section: "Settings",
      message: "" }
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
