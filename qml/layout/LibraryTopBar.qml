import QtQuick
import QtQuick.Layouts
import "../components" as Components
import "../theme" as Room36Theme

Item {
    id: root
    objectName: "libraryTopBar"

    property string projectName: ""
    property string pageTitle: "Library"
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
            source: theme.iconUrl("home.svg")
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
                color: theme.lightBlue
                font.family: typography.captionFamily
                font.pixelSize: typography.caption
                elide: Text.ElideRight
            }

            Text {
                objectName: "libraryPageTitle"
                Layout.fillWidth: true
                text: root.pageTitle
                color: theme.white
                font.family: typography.pageTitleFamily
                font.pixelSize: typography.sectionTitle
                elide: Text.ElideRight
            }
        }

        Components.ToolbarDropdownButton {
            objectName: "librarySortButton"
            label: "Sort by"
            iconSource: theme.iconUrl("sort.svg")
            normalFill: theme.neonGreen
            options: ["Date", "Type", "Portrait", "Landscape"]
            onOptionSelected: function(option) { root.sortRequested(option) }
        }

        Components.ToolbarDropdownButton {
            objectName: "libraryViewButton"
            label: "View"
            iconSource: theme.iconUrl("view.svg")
            normalFill: theme.neonGreen
            options: ["Large", "Medium", "Small"]
            onOptionSelected: function(option) { root.viewSizeRequested(option) }
        }
    }
}
