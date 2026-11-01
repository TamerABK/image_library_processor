.pragma library

// Navigation metadata only. No result rows, sample counts or person identities.
var entries = [
    { routeId: "library", label: "All", title: "Library", iconName: "grid", section: "Organize",
      message: "" },
    { routeId: "blurry", label: "Blurry photos", title: "Blurry photos", iconName: "image", section: "",
      message: "Blurry photo review will be added in Phase 6." },
    { routeId: "duplicates", label: "Near duplicates", title: "Near duplicates", iconName: "image_multiple", section: "",
      message: "" },
    { routeId: "favorites", label: "Favorites", title: "Favorites", iconName: "star", section: "",
      message: "" },
    { routeId: "knownPeople", label: "Known people", title: "Known people", iconName: "people", section: "People",
      message: "Known people content will be added in Phase 7." },
    { routeId: "unknownPeople", label: "Unknown people", title: "Unknown people", iconName: "people", section: "",
      message: "Unknown people review will be added in Phase 7." },
    { routeId: "trash", label: "Trash", title: "Trash", iconName: "delete", section: "",
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
