pragma ComponentBehavior: Bound

import QtQuick
import QtQuick.Layouts
import "../theme" as Room36Theme

Flickable {
    id: root

    property var projectModel: null
    readonly property int columnCount: Math.max(
        1,
        Math.floor((width + metrics.projectCardHorizontalGap) / (metrics.projectCardReferenceWidth + metrics.projectCardHorizontalGap))
    )

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
        width: root.width
        columns: root.columnCount
        columnSpacing: metrics.projectCardHorizontalGap
        rowSpacing: metrics.projectCardHorizontalGap

        NewProjectCard {
            Layout.preferredWidth: metrics.projectCardReferenceWidth
            Layout.preferredHeight: metrics.projectCardReferenceHeight
            onClicked: root.newProjectRequested()
        }

        Repeater {
            model: root.projectModel ? root.projectModel : 0

            delegate: ProjectCard {
                objectName: "projectCard_" + index
                required property int index
                required property var model

                property var row: model
                property var projectIdentifier: row.projectId !== undefined ? row.projectId : index

                Layout.preferredWidth: metrics.projectCardReferenceWidth
                Layout.preferredHeight: metrics.projectCardReferenceHeight
                source: row.thumbnailUrl !== undefined ? row.thumbnailUrl : ""
                title: row.name !== undefined ? row.name : ""
                photoCount: row.photoCount !== undefined ? row.photoCount : -1
                subtitle: row.lastOpened !== undefined ? row.lastOpened : ""
                onClicked: root.projectActivated(projectIdentifier, title)
                onMenuRequested: root.projectMenuRequested(projectIdentifier)
            }
        }
    }
}
