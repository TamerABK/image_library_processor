import QtQuick
import QtQuick.Layouts
import "../components" as Components
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "libraryTopBar"

    property string projectName: ""
    property string pageTitle: "Library"
    property bool libraryControlsEnabled: true
    signal homeRequested()
    signal sortRequested(string option)
    signal viewSizeRequested(string option)

    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    Room36Theme.Metrics { id: metrics }

    implicitHeight: metrics.libraryTopBarHeight

    RowLayout {
        anchors.fill: parent
        anchors.margins: metrics.spacingLarge
        spacing: metrics.spacingMedium

        Components.IconButton {
            objectName: "libraryHomeButton"
            source: theme.iconUrl("home_blue.svg")
            accessibleLabel: "Back to Home"
            onClicked: root.homeRequested()
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.minimumWidth: 0
            spacing: metrics.spacingTiny

            Text {
                objectName: "libraryProjectName"
                Layout.fillWidth: true
                text: root.projectName.length > 0 ? root.projectName : "Library"
                color: theme.midBlue
                font.family: typography.captionFamily
                font.pixelSize: typography.caption
                elide: Text.ElideRight
            }

            Text {
                objectName: "libraryPageTitle"
                Layout.fillWidth: true
                text: root.pageTitle
                color: theme.mainBlue
                font.family: typography.pageTitleFamily
                font.pixelSize: typography.sectionTitle
                elide: Text.ElideRight
            }
        }

        Components.ToolbarDropdownButton {
            objectName: "librarySortButton"
            enabled: root.libraryControlsEnabled
            label: "Sort by"
            iconSource: theme.iconUrl("sort.svg")
            normalFill: theme.neonGreen
            options: ["Name A–Z", "Name Z–A"]
            onOptionSelected: function(option) { root.sortRequested(option) }
        }

        Components.ToolbarDropdownButton {
            objectName: "libraryViewButton"
            enabled: root.libraryControlsEnabled
            label: "View"
            iconSource: theme.iconUrl("view.svg")
            normalFill: theme.neonGreen
            options: ["Large", "Medium", "Small"]
            onOptionSelected: function(option) { root.viewSizeRequested(option) }
        }
    }
}
