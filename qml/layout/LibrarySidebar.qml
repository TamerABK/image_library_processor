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
    property string projectName: ""
    property int photoCount: -1
    property real firstScanSeconds: -1
    // Replace the map when metadata changes. Missing/negative counts stay hidden.
    property var routeCounts: ({})
    signal routeRequested(string routeId)
    signal homeRequested()
    signal settingsRequested()

    function durationLabel(seconds) {
        if (seconds < 0) return "Pending"
        const whole = Math.round(seconds)
        return whole < 60 ? whole + "s" : Math.floor(whole / 60) + "m " + (whole % 60) + "s"
    }

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

    // Fill only the left arcs so the window edge is square; keep the right arcs.
    Rectangle {
        anchors.left: parent.left
        anchors.top: parent.top
        anchors.bottom: parent.bottom
        width: root.radius
        gradient: Gradient {
            GradientStop { position: 0; color: theme.lightBlue }
            GradientStop { position: 0.5; color: theme.midBlue }
            GradientStop { position: 1; color: theme.mainBlue }
        }
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
        anchors.bottom: bottomPanel.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: metrics.spacingLarge
        contentWidth: availableWidth
        clip: true

        Column {
            width: Math.min(scroll.availableWidth, metrics.sidebarNavRowWidth)
            x: (scroll.availableWidth - width) / 2
            spacing: metrics.spacingTiny

            Components.SidebarFilterItem {
                objectName: "libraryHomeButton"
                width: parent.width
                darkSurface: true
                label: "Home"
                iconName: "home"
                Accessible.checkable: false
                onClicked: root.homeRequested()
            }

            Repeater {
                model: Routes.entries.filter(function(entry) { return entry.routeId !== "trash" })

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
                        iconName: entryDelegate.modelData.iconName
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

        }
    }

    Column {
        id: bottomPanel
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: metrics.spacingLarge
        spacing: metrics.spacingTiny

        Components.SidebarFilterItem {
            objectName: "libraryNav_trash"
            width: parent.width
            darkSurface: true
            label: "Trash"
            iconName: "delete"
            selected: root.currentRoute === "trash"
            count: typeof root.routeCounts.trash === "number" && root.routeCounts.trash >= 0
                ? root.routeCounts.trash : -1
            onClicked: root.routeRequested("trash")
        }
        Components.SidebarFilterItem {
            objectName: "libraryPreferences"
            width: parent.width
            darkSurface: true
            label: "Settings"
            iconName: "settings"
            Accessible.checkable: false
            onClicked: root.settingsRequested()
        }
        Rectangle {
            objectName: "projectDetailsCard"
            width: parent.width
            height: 118
            radius: 14
            color: theme.midBlue
            border.width: 1
            border.color: theme.lightBlue
            Column {
                anchors.fill: parent
                anchors.margins: 10
                spacing: 5
                Text {
                    width: parent.width
                    text: root.projectName || "Project details"
                    color: theme.white
                    font.family: typography.boldFamily
                    font.weight: Font.Bold
                    font.pixelSize: typography.button
                    elide: Text.ElideRight
                }
                Text {
                    text: "Photos  " + (root.photoCount >= 0 ? root.photoCount : "—")
                    color: theme.white
                    font.family: typography.boldFamily
                    font.weight: Font.Bold
                    font.pixelSize: typography.caption
                }
                Text {
                    text: "First scan  " + root.durationLabel(root.firstScanSeconds)
                    color: theme.white
                    font.family: typography.boldFamily
                    font.weight: Font.Bold
                    font.pixelSize: typography.caption
                }
                Text {
                    text: "Blurry  " + (typeof root.routeCounts.blurry === "number" && root.routeCounts.blurry >= 0
                        ? root.routeCounts.blurry : "—")
                    color: theme.white
                    font.family: typography.boldFamily
                    font.weight: Font.Bold
                    font.pixelSize: typography.caption
                }
            }
        }
    }
}
