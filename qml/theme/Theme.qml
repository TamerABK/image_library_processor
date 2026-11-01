import QtQuick

QtObject {
    readonly property color mainBlue: "#00008E"
    readonly property color midBlue: "#4B44E0"
    readonly property color lightBlue: "#A4A4FF"
    readonly property color neonGreen: "#D7F205"
    readonly property color warningRed: "#D92D20"
    readonly property color warningYellow: "#B77900"
    readonly property color white: "#FFFFFF"
    readonly property color black: "#000000"

    readonly property color appBackground: white
    readonly property color homeBackground: white
    readonly property color authBackground: white
    readonly property color libraryBackground: white
    readonly property color libraryNavFill: Qt.rgba(0, 0, 142 / 255, 0.32)
    readonly property color galleryBackground: Qt.rgba(246 / 255, 246 / 255, 1, 1)
    readonly property color workspaceBackground: appBackground
    readonly property color panelFill: white
    readonly property color panelRaisedFill: Qt.rgba(1, 1, 1, 0.94)
    readonly property color mutedText: Qt.rgba(0, 0, 0, 0.58)
    readonly property color disabledText: Qt.rgba(0, 0, 0, 0.34)
    readonly property color disabledFill: Qt.rgba(0, 0, 0, 0.07)
    readonly property color hoverFill: Qt.rgba(75 / 255, 68 / 255, 224 / 255, 0.08)
    readonly property color pressedFill: Qt.rgba(0, 0, 142 / 255, 0.16)
    readonly property color selectedFill: Qt.rgba(164 / 255, 164 / 255, 255 / 255, 0.28)
    readonly property color separator: Qt.rgba(0, 0, 0, 0.12)
    readonly property color focusRing: Qt.rgba(215 / 255, 242 / 255, 5 / 255, 0.76)
    readonly property color selectedGlow: Qt.rgba(1, 1, 1, 0.50)
    readonly property color imagePlaceholder: Qt.rgba(75 / 255, 68 / 255, 224 / 255, 0.10)
    readonly property color overlayScrim: Qt.rgba(0, 0, 0, 0.38)
    readonly property color footerFill: mainBlue

    readonly property url assetsRoot: Qt.resolvedUrl("../../assets/")

    function assetUrl(relativePath) {
        return Qt.resolvedUrl("../../assets/" + relativePath)
    }

    function iconUrl(fileName) {
        // Compatibility for existing controls while their SVGs are moved to the
        // shared Fluent icon component. Brand marks remain in assets/icons.
        const icons = {
            "home.svg": "home_20_regular.svg",
            "home_blue.svg": "home_20_regular.svg",
            "settings.svg": "settings_20_regular.svg",
            "settings_home.svg": "settings_20_regular.svg",
            "profile.svg": "person_20_regular.svg",
            "sidebar_profile.svg": "people_20_regular.svg",
            "sidebar_view.svg": "image_multiple_20_regular.svg",
            "sidebar_view_medium.svg": "image_multiple_20_regular.svg",
            "sidebar_landscape.svg": "image_20_regular.svg",
            "landscape.svg": "image_24_regular.svg",
            "portrait.svg": "image_20_regular.svg",
            "star_shape.svg": "star_20_regular.svg",
            "star_shape_filled.svg": "star_20_filled.svg",
            "favorite.svg": "star_20_regular.svg",
            "favorite_filled.svg": "star_20_filled.svg",
            "trash.svg": "delete_20_regular.svg",
            "known_person_trash.svg": "delete_20_regular.svg",
            "pen.svg": "edit_20_regular.svg",
            "known_person_pen.svg": "edit_20_regular.svg",
            "sort.svg": "arrow_sort_20_regular.svg",
            "view.svg": "grid_20_regular.svg",
            "view_small.svg": "grid_20_regular.svg",
            "view_medium.svg": "grid_20_regular.svg",
            "view_large.svg": "grid_20_regular.svg",
            "project_open.svg": "folder_open_24_regular.svg",
            "import.svg": "arrow_download_20_regular.svg",
            "import_home.svg": "arrow_download_20_regular.svg",
            "export.svg": "arrow_upload_20_regular.svg",
            "subscription.svg": "premium_20_regular.svg",
            "search.svg": "search_20_regular.svg",
            "menu_more.svg": "more_horizontal_20_regular.svg",
            "chevron_down.svg": "chevron_down_16_regular.svg",
            "chevron_right.svg": "chevron_right_16_regular.svg",
            "date.svg": "calendar_20_regular.svg",
            "last_scan.svg": "clock_20_regular.svg",
            "file_type.svg": "image_20_regular.svg",
            "fullscreen.svg": "view_desktop_20_regular.svg"
        }
        return icons[fileName] ? fluentIconUrl(icons[fileName]) : assetUrl("icons/" + fileName)
    }

    function fluentIconUrl(fileName) {
        return assetUrl("icons/fluent/" + fileName)
    }

    function brandingUrl(fileName) {
        return assetUrl("branding/" + fileName)
    }

    function backgroundUrl(fileName) {
        return assetUrl("backgrounds/" + fileName)
    }

    function fontUrl(fileName) {
        return assetUrl("fonts/" + fileName)
    }
}
