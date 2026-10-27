pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme

Flickable {
    id: root

    property var projectModel: null
    property bool newProjectEnabled: true
    property string viewSize: "Medium"
    readonly property real preferredWidth: viewSize === "Small" ? 310
        : viewSize === "Large" ? 510 : metrics.projectCardReferenceWidth
    readonly property real preferredHeight: viewSize === "Small" ? 254
        : viewSize === "Large" ? 418 : metrics.projectCardReferenceHeight
    readonly property int columnCount: Math.max(
        1,
        Math.floor((width + metrics.projectCardHorizontalGap) / (preferredWidth + metrics.projectCardHorizontalGap))
    )
    readonly property real cardWidth: Math.min(preferredWidth,
        Math.max(1, (width - (columnCount - 1) * metrics.projectCardHorizontalGap) / columnCount))
    readonly property int occupiedColumns: Math.min(columnCount, 1 + projectRepeater.count)

    signal newProjectRequested()
    signal projectActivated(var projectId, string projectName)
    signal projectMenuRequested(var projectId)

    Room36Theme.Metrics { id: metrics }

    contentWidth: width
    contentHeight: grid.implicitHeight
    clip: true
    boundsBehavior: Flickable.StopAtBounds

    GridLayout {
        id: grid
        width: root.occupiedColumns * root.cardWidth
            + (root.occupiedColumns - 1) * metrics.projectCardHorizontalGap
        columns: root.columnCount
        columnSpacing: metrics.projectCardHorizontalGap
        rowSpacing: metrics.projectCardHorizontalGap

        NewProjectCard {
            objectName: "newProjectCard"
            enabled: root.newProjectEnabled
            viewSize: root.viewSize
            Layout.preferredWidth: root.cardWidth
            Layout.preferredHeight: root.preferredHeight
            onClicked: root.newProjectRequested()
        }

        Repeater {
            id: projectRepeater
            model: root.projectModel ? root.projectModel : 0

            delegate: ProjectCard {
                objectName: "projectCard_" + index
                required property int index
                required property var model

                // Qt detaches role objects while resetting/removing delegates.
                property var row: model || ({})
                property var projectIdentifier: row && row.projectId !== undefined ? row.projectId : index

                viewSize: root.viewSize
                Layout.preferredWidth: root.cardWidth
                Layout.preferredHeight: root.preferredHeight
                source: row && row.thumbnailUrl !== undefined ? row.thumbnailUrl : ""
                menuEnabled: false
                title: row && row.name !== undefined ? row.name : ""
                photoCount: row && row.photoCount !== undefined ? row.photoCount : -1
                subtitle: row && row.lastOpened !== undefined ? row.lastOpened : ""
                onClicked: root.projectActivated(projectIdentifier, title)
                onMenuRequested: root.projectMenuRequested(projectIdentifier)
            }
        }
    }
}
