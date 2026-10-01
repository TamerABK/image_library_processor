pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Controls
import "../components" as Components
import "../theme" as Room36Theme
import "../navigation/LibraryRoutes.js" as Routes

Rectangle {
    id: root
    objectName: "librarySidebar"

    property string currentRoute: "library"
    // Replace the map when metadata changes. Missing/negative counts stay hidden.
    property var routeCounts: ({})
    signal routeRequested(string routeId)
    signal homeRequested()
    signal settingsRequested()

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    implicitWidth: metrics.sidebarPreferredWidth
    radius: metrics.controlRadiusLarge
    gradient: Gradient {
        GradientStop { position: 0; color: theme.lightBlue }
        GradientStop { position: 0.5; color: theme.midBlue }
        GradientStop { position: 1; color: theme.mainBlue }
    }

    Image {
        id: logo
        anchors.top: parent.top
        anchors.topMargin: root.height >= 900 ? metrics.homeTopMargin : metrics.spacingLarge
        anchors.horizontalCenter: parent.horizontalCenter
        width: Math.min(metrics.sidebarNavRowWidth, root.width - metrics.spacingLarge)
        height: width / (metrics.homeLogoWidth / metrics.homeLogoHeight)
        source: theme.brandingUrl("logo_white.svg")
        fillMode: Image.PreserveAspectFit
    }

    ScrollView {
        id: scroll
        objectName: "librarySidebarScroll"
        anchors.top: logo.bottom
        anchors.bottom: parent.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: metrics.spacingLarge
        contentWidth: availableWidth
        clip: true

        Column {
            width: Math.min(scroll.availableWidth, metrics.sidebarNavRowWidth)
            x: (scroll.availableWidth - width) / 2
            spacing: metrics.spacingTiny

            Repeater {
                model: Routes.entries

                delegate: Column {
                    id: entryDelegate
                    required property var modelData
                    width: parent.width
                    spacing: metrics.spacingSmall

                    Text {
                        width: parent.width
                        topPadding: metrics.spacingMedium
                        bottomPadding: metrics.spacingTiny
                        visible: entryDelegate.modelData.section.length > 0
                        text: entryDelegate.modelData.section
                        color: theme.white
                        font.family: typography.sectionTitleFamily
                        font.pixelSize: typography.caption
                    }

                    Components.SidebarFilterItem {
                        objectName: "libraryNav_" + entryDelegate.modelData.routeId
                        width: parent.width
                        darkSurface: true
                        label: entryDelegate.modelData.label
                        iconSource: theme.iconUrl(entryDelegate.modelData.icon)
                        selected: root.currentRoute === entryDelegate.modelData.routeId
                        count: {
                            const value = root.routeCounts[entryDelegate.modelData.routeId]
                            return typeof value === "number" && isFinite(value) && value >= 0
                                ? Math.floor(value) : -1
                        }
                        onClicked: root.routeRequested(entryDelegate.modelData.routeId)
                    }
                }
            }

            Components.SidebarFilterItem {
                objectName: "librarySidebarHome"
                width: parent.width
                darkSurface: true
                label: "Home"
                iconSource: theme.iconUrl("home.svg")
                Accessible.checkable: false
                onClicked: root.homeRequested()
            }

            Components.SidebarFilterItem {
                objectName: "libraryPreferences"
                enabled: false
                width: parent.width
                darkSurface: true
                label: "Preferences"
                iconSource: theme.iconUrl("settings.svg")
                Accessible.checkable: false
                onClicked: root.settingsRequested()
            }
        }
    }
}
