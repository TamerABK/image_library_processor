import QtQuick

QtObject {
    readonly property color mainBlue: "#00008E"
    readonly property color midBlue: "#4B44E0"
    readonly property color lightBlue: "#A4A4FF"
    readonly property color neonGreen: "#D7F205"
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
        return assetUrl("icons/" + fileName)
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
