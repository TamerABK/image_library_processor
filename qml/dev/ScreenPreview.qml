import QtQuick
import "../pages" as Pages

Item {
    id: root
    objectName: "screenPreview"

    property string pageId: "home"

    signal newProjectRequested()
    signal projectActivated(var projectId, string projectName)
    signal projectMenuRequested(var projectId)
    signal searchChanged(string text)
    signal sortChanged(string value)
    signal viewSizeChanged(string value)

    ListModel {
        id: previewProjects

        ListElement { projectId: "saras-wedding"; name: "Sara's wedding"; photoCount: 286; thumbnailUrl: ""; lastOpened: "" }
        ListElement { projectId: "africa-trip"; name: "Africa trip"; photoCount: 184; thumbnailUrl: ""; lastOpened: "" }
        ListElement { projectId: "fall"; name: "Fall"; photoCount: 92; thumbnailUrl: ""; lastOpened: "" }
        ListElement { projectId: "karens-wedding"; name: "Karen's wedding"; photoCount: 341; thumbnailUrl: ""; lastOpened: "" }
        ListElement { projectId: "summer-trip"; name: "Summer trip"; photoCount: 128; thumbnailUrl: ""; lastOpened: "" }
        ListElement { projectId: "joshs-wedding"; name: "Josh's wedding"; photoCount: 219; thumbnailUrl: ""; lastOpened: "" }
        ListElement { projectId: "lamas-wedding"; name: "Lama's wedding"; photoCount: 157; thumbnailUrl: ""; lastOpened: "" }
    }

    Pages.HomePage {
        anchors.fill: parent
        projectModel: previewProjects
        onNewProjectRequested: root.newProjectRequested()
        onProjectActivated: function(projectId, projectName) { root.projectActivated(projectId, projectName) }
        onProjectMenuRequested: function(projectId) { root.projectMenuRequested(projectId) }
        onSearchChanged: function(text) { root.searchChanged(text) }
        onSortChanged: function(value) { root.sortChanged(value) }
        onViewSizeChanged: function(value) { root.viewSizeChanged(value) }
    }
}
